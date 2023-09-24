# curses.py
# A simple terminal worker UI
# Supports audio alerts on low VRAM / RAM and toggling worker maintenance mode.
import contextlib
import curses
import locale
import os
import re
import sys
import textwrap
import threading
import time
from collections import deque
from math import trunc

import pkg_resources
import psutil
import requests

from worker.logger import config, logger
from worker.stats import bridge_stats
from worker.utils.gpuinfo import GPUInfo


class DequeOutputCollector:
    def __init__(self):
        self.deque = deque()

    def write(self, s):
        if s != "\n":
            self.deque.append(s.strip())

    def set_size(self, size):
        while len(self.deque) > size:
            self.deque.popleft()

    def flush(self):
        pass

    def isatty(self):
        # No, we are not a TTY
        return False

    def close(self):
        pass


class TerminalUI:
    LOGURU_REGEX = re.compile(
        r"(\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d).*\| "
        r"(INIT_OK|INIT_ERR|INIT_WARN|INIT|DEBUG|INFO|WARNING|ERROR).*\| (.*) - (.*)$",
    )
    KUDOS_REGEX = re.compile(r".*average kudos per hour: (\d+)")
    JOBDONE_REGEX = re.compile(r".*(Generation for id.*finished successfully|Finished interrogation.*)")

    ART = {
        "top_left": "╓",
        "top_right": "╖",
        "bottom_left": "╙",
        "bottom_right": "╜",
        "horizontal": "─",
        "vertical": "║",
        "left-join": "╟",
        "right-join": "╢",
        "progress": "▓",
    }

    # Refresh interval in seconds to call API for remote worker stats
    REMOTE_STATS_REFRESH = 5

    # Refresh interval in seconds for API calls to get overall ai horde stats
    REMOTE_HORDE_STATS_REFRESH = 60

    COLOUR_RED = 1
    COLOUR_GREEN = 2
    COLOUR_YELLOW = 3
    COLOUR_BLUE = 4
    COLOUR_MAGENTA = 5
    COLOUR_CYAN = 6
    COLOUR_WHITE = 7

    DELIM = "::::"

    # Number of seconds between audio alerts
    ALERT_INTERVAL = 5

    JUNK = [
        "Result = False",
        "Result = True",
        "Try again with a different prompt and/or seed.",
    ]

    CLIENT_AGENT = "terminalui:1:db0"

    def __init__(self, bridge_data):
        self.should_stop = False
        self.bridge_data = bridge_data

        self.dreamer_worker = False
        self.scribe_worker = False
        self.alchemy_worker = False

        # We check the name rather the type directly to avoid bad things
        # happening if we import the Kobold class
        if bridge_data.__class__.__name__ == "InterrogationBridgeData":
            self.alchemy_worker = True
        elif bridge_data.__class__.__name__ == "StableDiffusionBridgeData":
            self.dreamer_worker = True
        elif bridge_data.__class__.__name__ == "KoboldAIBridgeData":
            self.scribe_worker = True

        self.model_manager = None

        if hasattr(self.bridge_data, "scribe_name") and self.scribe_worker:
            self.worker_name = self.bridge_data.scribe_name
        else:
            self.worker_name = self.bridge_data.worker_name
        if hasattr(self.bridge_data, "horde_url"):
            self.url = self.bridge_data.horde_url
        elif hasattr(self.bridge_data, "kai_url"):
            self.url = self.bridge_data.kai_url
        self._worker_info_thread = None
        self._horde_stats_thread = None
        self.main = None
        self.width = 0
        self.height = 0
        self.status_height = 17
        self.show_module = False
        self.show_debug = False
        self.last_key = None
        self.pause_log = False
        self.input = DequeOutputCollector()
        self.output = DequeOutputCollector()
        self.worker_id = None
        threading.Thread(target=self.load_worker_id, daemon=True).start()
        self.last_stats_refresh = time.time() - (TerminalUI.REMOTE_STATS_REFRESH - 3)
        self.last_horde_stats_refresh = time.time() - (TerminalUI.REMOTE_HORDE_STATS_REFRESH - 3)
        self.maintenance_mode = False
        self.gpu = GPUInfo()
        self.gpu.samples_per_second = 5
        self.commit_hash = self.get_commit_hash()
        self.cpu_average = []
        self.audio_alerts = False
        self.last_audio_alert = 0
        self.stdout = DequeOutputCollector()
        self._bck_stdout = sys.stdout
        self.stderr = DequeOutputCollector()
        self._bck_stderr = sys.stderr
        self.reset_stats()
        self.download_label = ""
        self.download_current = None
        self.download_total = None
        if not self.scribe_worker:
            from hordelib.settings import UserSettings
            from hordelib.shared_model_manager import SharedModelManager

            self.model_manager = SharedModelManager
            UserSettings.download_progress_callback = self.download_progress

    def initialise(self):
        # Suppress stdout / stderr
        sys.stderr = self.stderr
        sys.stdout = self.stdout
        # Remove all loguru sinks
        logger.remove()
        handlers = [sink for sink in config["handlers"] if isinstance(sink["sink"], str)]
        # Re-initialise loguru
        newconfig = {"handlers": handlers}
        logger.configure(**newconfig)
        # Add our own handler
        logger.add(self.input, level="DEBUG")
        locale.setlocale(locale.LC_ALL, "")
        self.initialise_main_window()
        self.resize()

    def download_progress(self, desc, current, total):
        """Called by the model manager when downloading files to update us on progress"""
        # Just save what we're passed for later rendering
        self.download_label = desc
        self.download_current = current
        self.download_total = total

    def load_log(self):
        self.load_log_queue()

    def parse_log_line(self, line):
        if regex := TerminalUI.LOGURU_REGEX.match(line):
            if not self.show_debug and regex.group(2) == "DEBUG":
                return None
            if regex.group(2) == "ERROR":
                self.error_count += 1
            elif regex.group(2) == "WARNING":
                self.warning_count += 1
            return f"{regex.group(2)}::::{regex.group(1)}::::{regex.group(3)}::::{regex.group(4)}"
        return None

    def load_log_queue(self):
        lines = list(self.input.deque)
        self.input.deque.clear()
        for line in lines:
            ignore = False
            for skip in TerminalUI.JUNK:
                if skip.lower() in line.lower():
                    ignore = True
            if ignore:
                continue
            log_line = self.parse_log_line(line)
            if not log_line:
                continue
            self.output.write(log_line)
            if regex := TerminalUI.KUDOS_REGEX.match(line):
                self.kudos_per_hour = int(regex.group(1))
            if regex := TerminalUI.JOBDONE_REGEX.match(line):
                self.jobs_done += 1

        self.output.set_size(self.height)

    def initialise_main_window(self):
        # getch doesn't block
        self.main.nodelay(True)
        # Hide cursor
        curses.curs_set(0)
        # Define colours
        curses.start_color()
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_BLUE, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        curses.init_pair(6, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLACK)

    def resize(self):
        # Determine terminal size
        curses.update_lines_cols()
        # Determine terminal size
        self.height, self.width = self.main.getmaxyx()

    def print(self, win, y, x, text, colour=None):
        # Ensure we're going to fit
        height, width = win.getmaxyx()
        if y < 0 or x < 0 or x + len(text) > width or y > height:
            return
        # Always highlight certain text
        if text == "Pending":
            colour = curses.color_pair(TerminalUI.COLOUR_YELLOW)
        with contextlib.suppress(curses.error):
            if not colour:
                win.addstr(y, x, text)
            else:
                win.addstr(y, x, text, colour)

    def draw_line(self, win, y, label):
        height, width = win.getmaxyx()
        self.print(
            win,
            y,
            0,
            TerminalUI.ART["left-join"] + TerminalUI.ART["horizontal"] * (width - 2) + TerminalUI.ART["right-join"],
        )
        self.print(win, y, 2, label)

    def draw_box(self, y, x, width, height):  # noqa: ARG002
        # An attempt to work cross platform, box() doesn't.

        # Draw the top border
        self.print(
            self.main,
            0,
            0,
            TerminalUI.ART["top_left"] + TerminalUI.ART["horizontal"] * (width - 2) + TerminalUI.ART["top_right"],
        )

        # Draw the side borders
        for y in range(1, height - 1):
            self.print(self.main, y, 0, TerminalUI.ART["vertical"])
            self.print(self.main, y, width - 1, TerminalUI.ART["vertical"])

        # Draw the bottom border
        self.print(
            self.main,
            height - 1,
            0,
            TerminalUI.ART["bottom_left"] + TerminalUI.ART["horizontal"] * (width - 2),
        )
        self.print(self.main, height - 1, width - 1, TerminalUI.ART["bottom_right"])

    def seconds_to_timestring(self, seconds):
        if isinstance(seconds, str):
            return seconds
        hours = int(seconds // 3600)
        days = hours // 24
        hours %= 24
        result = ""
        if days:
            result += f"{days}d "
        if hours:
            result += f"{hours}h "
        if minutes := int((seconds % 3600) // 60):
            result += f"{minutes}m"
        return result

    def get_uptime(self):
        hours = int((time.time() - self.start_time) // 3600)
        minutes = int(((time.time() - self.start_time) % 3600) // 60)
        seconds = int((time.time() - self.start_time) % 60)
        return f"{hours}:{minutes:02}:{seconds:02}"

    def reset_stats(self):
        bridge_stats.reset()
        self.start_time = time.time()
        self.jobs_done = 0
        self.kudos_per_hour = 0
        self.pop_time = 0
        self.jobs_per_hour = "Pending"
        self.total_kudos = "Pending"
        self.total_worker_kudos = "Pending"
        self.total_uptime = "Pending"
        self.avg_kudos_per_job = "unknown"
        self.threads = "Pending"
        self.total_failed_jobs = "Pending"
        self.total_models = "Pending"
        self.total_jobs = "Pending"
        self.queued_requests = "Pending"
        self.worker_count = "Pending"
        self.thread_count = "Pending"
        self.queued_mps = "Pending"
        self.last_minute_mps = "Pending"
        self.queue_time = "Pending"
        self.error_count = 0
        self.warning_count = 0

    def print_switch(self, y, x, label, switch):
        colour = curses.color_pair(TerminalUI.COLOUR_CYAN) if switch else curses.color_pair(TerminalUI.COLOUR_WHITE)
        self.print(self.main, y, x, label, colour)
        return x + len(label) + 2

    def get_free_ram(self):
        mem = psutil.virtual_memory().available
        percent = 100 - trunc(psutil.virtual_memory().percent)
        mem /= 1048576
        unit = "MB"
        if mem >= 1024:
            mem /= 1024
            unit = "GB"
        mem = trunc(mem)
        return f"{mem} {unit} ({percent}%)"

    def get_cpu_usage(self):
        cpu = psutil.cpu_percent()
        self.cpu_average.append(cpu)
        self.cpu_average = self.cpu_average[-(self.gpu.samples_per_second * 60 * 5) :]
        avg_cpu = trunc(sum(self.cpu_average) / len(self.cpu_average))
        cpu = f"{trunc(cpu)}%".ljust(3)
        return f"{cpu} ({avg_cpu}%)"

    def print_status(self):
        # This is the design template: (80 columns)
        # ╔═AIDream-01════════════════════════════════════════════════(0.10.10)══000000═╗
        # ║   Uptime: 0:14:35     Jobs Completed: 6       Avg Kudos Per Job: 103        ║
        # ║   Models: 174         Kudos Per Hour: 5283        Jobs Per Hour: 524966     ║
        # ║  Threads: 3                 Warnings: 9999               Errors: 100        ║
        # ║ CPU Load: 99% (99%)         Free RAM: 2 GB (99%)      Job Fetch: 2.32s      ║
        # ╟─NVIDIA GeForce RTX 3090─────────────────────────────────────────────────────╢
        # ║   Load: 100% (90%)        VRAM Total: 24576MiB        Fan Speed: 100%       ║
        # ║   Temp: 100C (58C)         VRAM Used: 16334MiB          PCI Gen: 5          ║
        # ║  Power: 460W (178W)        VRAM Free: 8241MiB         PCI Width: 32x        ║
        # ╟─Worker Total────────────────────────────────────────────────────────────────╢
        # ║                         Worker Kudos: 9385297        Total Jobs: 701138     ║
        # ║                         Total Uptime: 34d 19h 14m   Jobs Failed: 972        ║
        # ╟─Entire Horde────────────────────────────────────────────────────────────────╢
        # ║                          Jobs Queued: 99999          Queue Time: 99m        ║
        # ║                        Total Workers: 1000        Total Threads: 1000       ║
        # ║ Downloading some-file-with-long-name.safetensors: ######################### ║
        # ║   (m)aintenance  (s)ource  (d)ebug  (p)ause log  (a)lerts  (r)eset  (q)uit  ║
        # ╙─────────────────────────────────────────────────────────────────────────────╜

        # Define three colums centres
        col_left = 12
        col_mid = self.width // 2
        col_right = self.width - 12

        # How many GPUs are we using?
        num_gpus = self.gpu.get_num_gpus()

        # Define rows on which sections start
        row_local = 0
        row_gpu = row_local + 5
        row_total = row_gpu + (4 * num_gpus)
        row_horde = row_total + 3
        self.status_height = row_horde + 6

        def label(y, x, label):
            self.print(self.main, y, x - len(label) - 1, label)

        self.draw_box(0, 0, self.width, self.status_height)
        self.draw_line(self.main, row_gpu, "")
        self.draw_line(self.main, row_total, "Worker Total")
        self.draw_line(self.main, row_horde, "Entire Horde")
        self.print(self.main, row_local, 2, f"{self.worker_name}")
        self.print(self.main, row_local, self.width - 8, f"{self.commit_hash[:6]}")
        self.print(self.main, row_local, self.width - 19, f"({self.get_hordelib_version()})")

        label(row_local + 1, col_left, "Uptime:")
        label(row_local + 2, col_left, "Models:")
        label(row_local + 3, col_left, "Threads:")
        label(row_local + 4, col_left, "CPU Load:")
        label(row_local + 1, col_mid, "Jobs Completed:")
        label(row_local + 2, col_mid, "Kudos Per Hour:")
        label(row_local + 3, col_mid, "Warnings:")
        label(row_local + 4, col_mid, "Free RAM:")
        label(row_local + 1, col_right, "Avg Kudos Per Job:")
        label(row_local + 2, col_right, "Jobs Per Hour:")
        label(row_local + 3, col_right, "Errors:")
        label(row_local + 4, col_right, "Job Fetch:")

        tmp_row_gpu = row_gpu
        for gpu_i in range(num_gpus):
            label(tmp_row_gpu + 1, col_left, "Load:")
            label(tmp_row_gpu + 2, col_left, "Temp:")
            label(tmp_row_gpu + 3, col_left, "Power:")
            label(tmp_row_gpu + 1, col_mid, "VRAM Total:")
            label(tmp_row_gpu + 2, col_mid, "VRAM Used:")
            label(tmp_row_gpu + 3, col_mid, "VRAM Free:")
            label(tmp_row_gpu + 1, col_right, "Fan Speed:")
            label(tmp_row_gpu + 2, col_right, "PCI Gen:")
            label(tmp_row_gpu + 3, col_right, "PCI Width:")
            tmp_row_gpu += 4

        label(row_total + 1, col_mid, "Worker Kudos:")
        label(row_total + 2, col_mid, "Total Uptime:")
        label(row_total + 1, col_right, "Total Jobs:")
        label(row_total + 2, col_right, "Jobs Failed:")

        label(row_horde + 1, col_mid, "Jobs Queued:")
        label(row_horde + 2, col_mid, "Total Workers:")
        label(row_horde + 1, col_right, "Queue Time:")
        label(row_horde + 2, col_right, "Total Threads:")

        self.print(self.main, row_local + 1, col_left, f"{self.get_uptime()}")
        self.print(self.main, row_local + 1, col_mid, f"{self.jobs_done}")
        self.print(self.main, row_local + 1, col_right, f"{self.avg_kudos_per_job}")

        self.print(self.main, row_local + 2, col_left, f"{self.total_models}")
        self.print(self.main, row_local + 2, col_mid, f"{self.kudos_per_hour}")
        self.print(self.main, row_local + 2, col_right, f"{self.jobs_per_hour}")

        self.print(self.main, row_local + 3, col_left, f"{self.threads}")
        self.print(self.main, row_local + 3, col_mid, f"{self.warning_count}")
        self.print(self.main, row_local + 3, col_right, f"{self.error_count}")
        self.print(self.main, row_local + 4, col_right, f"{self.pop_time} s")

        # Add some warning colours to free ram
        ram = self.get_free_ram()
        ram_colour = curses.color_pair(TerminalUI.COLOUR_WHITE)
        if re.match(r"\d{3,4} MB", ram):
            ram_colour = curses.color_pair(TerminalUI.COLOUR_MAGENTA)
        elif re.match(r"(\d{1,2}) MB", ram):
            if self.audio_alerts and time.time() - self.last_audio_alert > TerminalUI.ALERT_INTERVAL:
                self.last_audio_alert = time.time()
                curses.beep()
            ram_colour = curses.color_pair(TerminalUI.COLOUR_RED)

        self.print(self.main, row_local + 4, col_left, f"{self.get_cpu_usage()}")
        self.print(self.main, row_local + 4, col_mid, f"{self.get_free_ram()}", ram_colour)

        gpus = []
        for gpu_i in range(num_gpus):
            gpus.append(self.gpu.get_info(gpu_i))
        for gpu_i, gpu in enumerate(gpus):
            if gpu:
                # Add some warning colours to free vram
                vram_colour = curses.color_pair(TerminalUI.COLOUR_WHITE)
                if re.match(r"\d\d\d MB", gpu["vram_free"]):
                    vram_colour = curses.color_pair(TerminalUI.COLOUR_MAGENTA)
                elif re.match(r"(\d{1,2}) MB", gpu["vram_free"]):
                    if self.audio_alerts and time.time() - self.last_audio_alert > TerminalUI.ALERT_INTERVAL:
                        self.last_audio_alert = time.time()
                        curses.beep()
                    vram_colour = curses.color_pair(TerminalUI.COLOUR_RED)

                gpu_name = gpu["product"]
                if num_gpus > 1:
                    gpu_name = f"{gpu_name} #{gpu_i}"
                self.draw_line(self.main, row_gpu, gpu_name)

                self.print(self.main, row_gpu + 1, col_left, f"{gpu['load']:4} ({gpu['avg_load']})")
                self.print(self.main, row_gpu + 1, col_mid, f"{gpu['vram_total']}")
                self.print(self.main, row_gpu + 1, col_right, f"{gpu['fan_speed']}")

                self.print(self.main, row_gpu + 2, col_left, f"{gpu['temp']:4} ({gpu['avg_temp']})")
                self.print(self.main, row_gpu + 2, col_mid, f"{gpu['vram_used']}")
                self.print(self.main, row_gpu + 2, col_right, f"{gpu['pci_gen']}")

                self.print(self.main, row_gpu + 3, col_left, f"{gpu['power']:4} ({gpu['avg_power']})")
                self.print(self.main, row_gpu + 3, col_mid, f"{gpu['vram_free']}", vram_colour)
                self.print(self.main, row_gpu + 3, col_right, f"{gpu['pci_width']}")

                row_gpu += 4

        self.print(self.main, row_total + 1, col_mid, f"{self.total_kudos}")
        self.print(self.main, row_total + 1, col_right, f"{self.total_jobs}")

        self.print(self.main, row_total + 2, col_mid, f"{self.seconds_to_timestring(self.total_uptime)}")
        self.print(self.main, row_total + 2, col_right, f"{self.total_failed_jobs}")

        self.print(self.main, row_horde + 1, col_mid, f"{self.queued_requests}")
        self.print(self.main, row_horde + 1, col_right, f"{self.seconds_to_timestring(self.queue_time)}")

        self.print(self.main, row_horde + 2, col_mid, f"{self.worker_count}")
        self.print(self.main, row_horde + 2, col_right, f"{self.thread_count}")

        inputs = [
            "(m)aintenance",
            "(s)ource",
            "(d)ebug",
            "(p)ause log",
            "(a)udio alerts",
            "(r)eset",
            "(q)uit",
        ]
        x = self.width - len("  ".join(inputs)) - 2
        y = row_horde + 4
        x = self.print_switch(y, x, inputs[0], self.maintenance_mode)
        x = self.print_switch(y, x, inputs[1], self.show_module)
        x = self.print_switch(y, x, inputs[2], self.show_debug)
        x = self.print_switch(y, x, inputs[3], self.pause_log)
        x = self.print_switch(y, x, inputs[4], self.audio_alerts)
        x = self.print_switch(y, x, inputs[5], False)
        x = self.print_switch(y, x, inputs[6], False)

        # Display download progress bar if any
        if self.download_total:
            y = row_horde + 3
            x = 2
            self.print(self.main, y, x, self.download_label)
            x += len(self.download_label) + 1
            percentage_done = self.download_current / self.download_total
            done_in_chars = round((self.width - (x + 2)) * percentage_done)
            progress = TerminalUI.ART["progress"] * done_in_chars
            colour = curses.color_pair(TerminalUI.COLOUR_GREEN)
            self.print(self.main, y, x, progress, colour)
        else:
            y = row_horde + 3
            x = 2
            clearline = " " * (self.width - (x + 2))
            self.print(self.main, y, x, clearline)

    def fit_output_to_term(self, output):
        # How many lines of output can we fit, after line wrapping?
        termrows = self.height - self.status_height
        linecount = 0
        maxrows = 0
        for i, fullline in enumerate(reversed(output)):
            line = fullline.split(TerminalUI.DELIM)[-1:][0]
            # 21 is the timestamp length
            linecount += len(textwrap.wrap(line, self.width - 21))
            if self.show_module:
                linecount += 1
            if linecount > termrows:
                maxrows = i
                break
        # Truncate the output so it fits
        return output[-maxrows:]

    def print_log(self):
        if not self.pause_log:
            self.load_log()
        output = list(self.output.deque)
        if not output:
            return

        output = self.fit_output_to_term(output)

        y = self.status_height
        inputrow = 0
        last_timestamp = ""
        while y < self.height and inputrow < len(output):
            # Print any log info we have
            cat, nextwhen, source, msg = output[inputrow].split(TerminalUI.DELIM)
            colour = TerminalUI.COLOUR_WHITE
            if cat == "DEBUG":
                colour = TerminalUI.COLOUR_WHITE
            elif cat == "ERROR":
                colour = TerminalUI.COLOUR_RED
            elif cat in ["INIT"]:
                colour = TerminalUI.COLOUR_WHITE
            elif cat in ["INIT_OK"]:
                colour = TerminalUI.COLOUR_GREEN
                msg = f"OK: {msg}"
            elif cat in ["INIT_WARN"]:
                colour = TerminalUI.COLOUR_YELLOW
                msg = f"Warning: {msg}"
            elif cat in ["INIT_ERR"]:
                colour = TerminalUI.COLOUR_RED
                msg = f"Error: {msg}"
            elif cat == "WARNING":
                colour = TerminalUI.COLOUR_YELLOW

            # Timestamp
            when = nextwhen if nextwhen != last_timestamp else ""
            last_timestamp = nextwhen
            length = len(last_timestamp) + 2
            self.print(self.main, y, 1, f"{when}", curses.color_pair(TerminalUI.COLOUR_GREEN))

            # Source file name
            if self.show_module:
                self.print(self.main, y, length, f"{source}", curses.color_pair(TerminalUI.COLOUR_GREEN))
                y += 1
                if y > self.height:
                    break

            # Actual log message
            text = textwrap.wrap(msg, self.width - length)
            for line in text:
                self.print(self.main, y, length, line, curses.color_pair(colour))
                y += 1
                if y > self.height:
                    break
            inputrow += 1

    def load_worker_id(self):
        try:
            while not self.worker_id:
                if not self.worker_name:
                    logger.warning("Still waiting to determine worker name")
                    time.sleep(5)
                    continue
                workers_url = f"{self.url}/api/v2/workers"
                try:
                    r = requests.get(workers_url, headers={"client-agent": TerminalUI.CLIENT_AGENT}, timeout=5)
                except requests.exceptions.Timeout:
                    logger.warning("Timeout while waiting for worker ID from API")
                except requests.exceptions.RequestException as ex:
                    logger.error(f"Failed to get worker ID {ex}")
                if r.ok:
                    worker_json = r.json()
                    self.worker_id = next(
                        (item["id"] for item in worker_json if item["name"] == self.worker_name),
                        None,
                    )
                    if self.worker_id:
                        logger.warning(f"Found worker ID {self.worker_id}")
                    else:
                        pass
                        # # Our worker is not yet in the worker results from the API (cache delay)
                        # logger.warning("Waiting for the AI Horde to acknowledge this worker to fetch worker ID")
                else:
                    logger.warning(f"Failed to get worker ID {r.status_code}")
                time.sleep(5)
        except Exception as ex:
            logger.warning(str(ex))

    def set_maintenance_mode(self, enabled):
        if not self.bridge_data.api_key or not self.worker_id:
            return
        header = {"apikey": self.bridge_data.api_key, "client-agent": TerminalUI.CLIENT_AGENT}
        payload = {"maintenance": enabled}
        if enabled:
            logger.warning("Attempting to enable maintenance mode.")
        else:
            logger.warning("Attempting to disable maintenance mode.")
        worker_URL = f"{self.url}/api/v2/workers/{self.worker_id}"
        res = requests.put(worker_URL, json=payload, headers=header)
        if not res.ok:
            logger.error(f"Maintenance mode failed: {res.text}")

    def get_remote_worker_info(self):
        try:
            if not self.worker_id:
                return
            worker_URL = f"{self.url}/api/v2/workers/{self.worker_id}"
            try:
                r = requests.get(worker_URL, headers={"client-agent": TerminalUI.CLIENT_AGENT}, timeout=5)
            except requests.exceptions.Timeout:
                logger.warning("Worker info API failed to respond in time")
                return
            except requests.exceptions.RequestException:
                logger.warning("Worker info API request failed {ex}")
                return
            if not r.ok:
                logger.warning(f"Calling Worker information API failed ({r.status_code})")
                return
            data = r.json()

            self.maintenance_mode = data.get("maintenance_mode", False)
            self.total_worker_kudos = data.get("kudos_details", {}).get("generated", 0)
            if self.total_worker_kudos is not None:
                self.total_worker_kudos = int(self.total_worker_kudos)
            self.total_jobs = data.get("requests_fulfilled", 0)
            self.total_kudos = int(data.get("kudos_rewards", 0))
            self.threads = data.get("threads", 0)
            self.total_uptime = data.get("uptime", 0)
            self.total_failed_jobs = data.get("uncompleted_jobs", 0)
            if self.scribe_worker and data.get("models"):
                self.total_models = data.get("models")[0]
        except Exception as ex:
            logger.warning(str(ex))

    def get_remote_horde_stats(self):
        try:
            url = f"{self.url}/api/v2/status/performance"
            try:
                r = requests.get(url, headers={"client-agent": TerminalUI.CLIENT_AGENT}, timeout=10)
            except requests.exceptions.Timeout:
                pass
            except requests.exceptions.RequestException:
                return
            if not r.ok:
                logger.warning(f"Calling AI Horde stats API failed ({r.status_code})")
                return
            data = r.json()
            self.queued_requests = int(data.get("queued_requests", 0))
            self.worker_count = int(data.get("worker_count", 1))
            self.thread_count = int(data.get("thread_count", 0))
            self.queued_mps = int(data.get("queued_megapixelsteps", 0))
            self.last_minute_mps = int(data.get("past_minute_megapixelsteps", 0))
            self.queue_time = (self.queued_mps / self.last_minute_mps) * 60
        except Exception as ex:
            logger.warning(str(ex))

    def update_stats(self):
        # Total models
        if self.model_manager and self.model_manager.manager:
            if self.dreamer_worker:
                self.total_models = len(self.model_manager.manager.get_loaded_models_names(mm_include="compvis"))
            elif self.alchemy_worker:
                self.total_models = len(
                    self.model_manager.manager.get_loaded_models_names(
                        mm_include=["clip", "blip", "codeformer", "gfpgan", "esrgan"],
                    ),
                )
            elif self.scribe_worker:
                self.total_models = "See KAI"
        # Recent job pop times
        if "pop_time_avg_5_mins" in bridge_stats.stats:
            self.pop_time = bridge_stats.stats["pop_time_avg_5_mins"]
        if "jobs_per_hour" in bridge_stats.stats:
            self.jobs_per_hour = bridge_stats.stats["jobs_per_hour"]
        if "avg_kudos_per_job" in bridge_stats.stats:
            self.avg_kudos_per_job = bridge_stats.stats["avg_kudos_per_job"]

        if time.time() - self.last_stats_refresh > TerminalUI.REMOTE_STATS_REFRESH:
            self.last_stats_refresh = time.time()
            if (self._worker_info_thread and not self._worker_info_thread.is_alive()) or not self._worker_info_thread:
                self._worker_info_thread = threading.Thread(target=self.get_remote_worker_info, daemon=True).start()

        if time.time() - self.last_horde_stats_refresh > TerminalUI.REMOTE_HORDE_STATS_REFRESH:
            self.last_horde_stats_refresh = time.time()
            if (self._horde_stats_thread and not self._horde_stats_thread.is_alive()) or not self._horde_stats_thread:
                self._horde_stats_thread = threading.Thread(target=self.get_remote_horde_stats, daemon=True).start()

    def get_commit_hash(self):
        head_file = os.path.join(".git", "HEAD")
        if not os.path.exists(head_file):
            return ""
        try:
            with open(head_file, "r") as f:
                head_contents = f.read().strip()

            if not head_contents.startswith("ref:"):
                return head_contents

            ref_path = os.path.join(".git", *head_contents[5:].split("/"))

            with open(ref_path, "r") as f:
                return f.read().strip()

        except Exception:
            return ""

    def get_input(self):
        x = self.main.getch()
        self.last_key = x
        if x == curses.KEY_RESIZE:
            self.resize()
        elif x == ord("d") or x == ord("D"):
            self.show_debug = not self.show_debug
        elif x == ord("s") or x == ord("S"):
            self.show_module = not self.show_module
        elif x == ord("a") or x == ord("A"):
            self.audio_alerts = not self.audio_alerts
        elif x == ord("r") or x == ord("R"):
            self.reset_stats()
        elif x == ord("q") or x == ord("Q"):
            return True
        elif x == ord("m") or x == ord("M"):
            self.maintenance_mode = not self.maintenance_mode
            self.set_maintenance_mode(self.maintenance_mode)
        elif x == ord("p") or x == ord("P"):
            self.pause_log = not self.pause_log

        return None

    def poll(self):
        if self.get_input():
            return True
        self.main.erase()
        self.update_stats()
        self.print_status()
        self.print_log()
        self.main.refresh()
        return None

    def main_loop(self, stdscr):
        if not stdscr:
            self.stop()
            logger.error("Failed to initialise curses")
            return

        self.main = stdscr
        while True:
            if self.should_stop:
                return
            try:
                self.initialise()
                while True:
                    if self.should_stop:
                        return
                    if self.poll():
                        return
                    time.sleep(1 / self.gpu.samples_per_second)
            except KeyboardInterrupt:
                return
            except Exception as exc:
                logger.error(str(exc))

    def run(self):
        self.should_stop = False
        curses.wrapper(self.main_loop)

    def stop(self):
        self.should_stop = True
        # Restore the terminal
        sys.stdout = self._bck_stdout
        sys.stderr = self._bck_stderr

        curses.nocbreak()
        self.main.keypad(False)
        curses.echo()
        curses.endwin()

    def get_hordelib_version(self):
        try:
            return pkg_resources.get_distribution("hordelib").version
        except pkg_resources.DistributionNotFound:
            return "Unknown"


if __name__ == "__main__":
    print("Enable the terminal UI in bridgeData.yaml")
