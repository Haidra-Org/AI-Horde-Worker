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

import psutil
import requests
import yaml
from nataili.util.logger import config, logger
from pynvml.smi import nvidia_smi


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


class GPUInfo:
    def __init__(self):
        self.avg_load = []
        self.avg_temp = []
        self.avg_power = []
        # Average period in samples, default 10 samples per second, period 5 minutes
        self.samples_per_second = 10
        # Look out for device env var hack
        self.device = int(os.getenv("CUDA_VISIBLE_DEVICES", 0))

    # Return a value from the given dictionary supporting dot notation
    def get(self, data, key, default=""):
        # Handle nested structures
        path = key.split(".")

        if len(path) == 1:
            # Simple case
            return data[key] if key in data else default
        # Nested case
        walkdata = data
        for element in path:
            if element in walkdata:
                walkdata = walkdata[element]
            else:
                walkdata = ""
                break
        return walkdata

    def _get_gpu_data(self):
        with contextlib.suppress(Exception):
            nvsmi = nvidia_smi.getInstance()
            data = nvsmi.DeviceQuery()
            return data.get("gpu", [None])[self.device]

    def _mem(self, raw):
        unit = "GB"
        mem = raw / 1024
        if mem < 1:
            unit = "MB"
            raw *= 1024
        return f"{int(mem)} {unit}"

    def get_info(self):
        data = self._get_gpu_data()
        if not data:
            return None

        # Calculate averages
        try:
            gpu_util = int(self.get(data, "utilization.gpu_util", 0))
        except ValueError:
            gpu_util = 0

        try:
            gpu_temp = int(self.get(data, "temperature.gpu_temp", 0))
        except ValueError:
            gpu_temp = 0

        try:
            gpu_power = int(self.get(data, "power_readings.power_draw", 0))
        except ValueError:
            gpu_power = 0

        self.avg_load.append(gpu_util)
        self.avg_temp.append(gpu_temp)
        self.avg_power.append(gpu_power)
        self.avg_load = self.avg_load[-(self.samples_per_second * 60 * 5) :]
        self.avg_power = self.avg_power[-(self.samples_per_second * 60 * 5) :]
        self.avg_temp = self.avg_temp[-(self.samples_per_second * 60 * 5) :]
        avg_load = int(sum(self.avg_load) / len(self.avg_load))
        avg_power = int(sum(self.avg_power) / len(self.avg_power))
        avg_temp = int(sum(self.avg_temp) / len(self.avg_temp))

        return {
            "product": self.get(data, "product_name", "unknown"),
            "pci_gen": self.get(data, "pci.pci_gpu_link_info.pcie_gen.current_link_gen", "?"),
            "pci_width": self.get(data, "pci.pci_gpu_link_info.link_widths.current_link_width", "?"),
            "fan_speed": f"{self.get(data, 'fan_speed')}{self.get(data, 'fan_speed_unit')}",
            "vram_total": f"{self._mem(self.get(data, 'fb_memory_usage.total', '0'))}",
            "vram_used": f"{self._mem(self.get(data, 'fb_memory_usage.used', '0'))}",
            "vram_free": f"{self._mem(self.get(data, 'fb_memory_usage.free', '0'))}",
            "load": f"{gpu_util}{self.get(data, 'utilization.unit')}",
            "temp": f"{gpu_temp}{self.get(data, 'temperature.unit')}",
            "power": f"{gpu_power}{self.get(data, 'power_readings.unit')}",
            "avg_load": f"{avg_load}{self.get(data, 'utilization.unit')}",
            "avg_temp": f"{avg_temp}{self.get(data, 'temperature.unit')}",
            "avg_power": f"{avg_power}{self.get(data, 'power_readings.unit')}",
        }


class TerminalUI:
    REGEX = re.compile(r"(INIT|DEBUG|INFO|WARNING|ERROR).*(\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d).*\| (.*) - (.*)$")
    LOGURU_REGEX = re.compile(
        r"(\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d).*\| (INIT|INIT_OK|DEBUG|INFO|WARNING|ERROR).*\| (.*) - (.*)$",
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
    }

    # Refresh interval in seconds to call API for remote worker stats
    REMOTE_STATS_REFRESH = 30

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
    ]

    CLIENT_AGENT = "terminalui:1:db0"

    def __init__(self, worker_name=None, apikey=None, url="https://stablehorde.net"):
        self.url = url
        self.main = None
        self.width = 0
        self.height = 0
        self.status_height = 17
        self.show_module = False
        self.show_debug = False
        self.last_key = None
        self.pause_log = False
        self.use_log_file = False
        self.log_file = None
        self.input = DequeOutputCollector()
        self.output = DequeOutputCollector()
        self.worker_name = worker_name
        self.apikey = apikey
        self.worker_id = self.load_worker_id()
        self.start_time = time.time()
        self.last_stats_refresh = 0
        self.jobs_done = 0
        self.kudos_per_hour = 0
        self.jobs_per_hour = 0
        self.maintenance_mode = False
        self.total_kudos = 0
        self.total_worker_kudos = 0
        self.total_uptime = 0
        self.performance = "unknown"
        self.threads = 0
        self.total_failed_jobs = 0
        self.total_models = 0
        self.total_jobs = 0
        self.queued_requests = 0
        self.worker_count = 0
        self.thread_count = 0
        self.queued_mps = 0
        self.last_minute_mps = 0
        self.queue_time = 0
        self.gpu = GPUInfo()
        self.gpu.samples_per_second = 5
        self.error_count = 0
        self.warning_count = 0
        self.commit_hash = self.get_commit_hash()
        self.cpu_average = []
        self.audio_alerts = False
        self.last_audio_alert = 0
        self.stdout = DequeOutputCollector()
        self.stderr = DequeOutputCollector()

    def initialise(self):
        # Suppress stdout / stderr
        sys.stderr = self.stderr
        sys.stdout = self.stdout
        if self.use_log_file:
            self.open_log()
        else:
            # Remove all loguru sinks
            logger.remove()
            handlers = [sink for sink in config["handlers"] if type(sink["sink"]) is str]
            # Re-initialise loguru
            newconfig = {"handlers": handlers}
            logger.configure(**newconfig)
            # Add our own handler
            logger.add(self.input, level="DEBUG")
        locale.setlocale(locale.LC_ALL, "")
        self.initialise_main_window()
        self.resize()
        self.get_remote_worker_info()

    def open_log(self):
        # We try a couple of times, log rotiation, etc
        for _ in range(2):
            try:
                self.log_file = open("logs/bridge.log", "rt", encoding="utf-8", errors="ignore")  # noqa: SIM115
                # FIXME @jug this probably could be reworked
                self.log_file.seek(0, os.SEEK_END)
                break
            except OSError:
                time.sleep(1)

    def load_log(self):
        if self.use_log_file:
            while line := self.log_file.readline():
                self.input.write(line)
        self.load_log_queue()

    def parse_log_line(self, line):
        if self.use_log_file:
            if regex := TerminalUI.REGEX.match(line):
                if not self.show_debug and regex.group(1) == "DEBUG":
                    return None
                if regex.group(1) == "ERROR":
                    self.error_count += 1
                elif regex.group(1) == "WARNING":
                    self.warning_count += 1
                return f"{regex.group(1)}::::{regex.group(2)}::::{regex.group(3)}::::{regex.group(4)}"
            return None

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
                self.jobs_per_hour = int(3600 / ((time.time() - self.start_time) / self.jobs_done))

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

    def print_switch(self, y, x, label, switch):
        colour = curses.color_pair(TerminalUI.COLOUR_CYAN) if switch else curses.color_pair(TerminalUI.COLOUR_WHITE)
        self.print(self.main, y, x, label, colour)
        return x + len(label) + 2

    def get_free_ram(self):
        mem = psutil.virtual_memory().free
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
        # ╔═AIDream-01══════════════════════════════════════════════════════════════════╗
        # ║   Uptime: 0:14:35     Jobs Completed: 6             Performance: 0.3 MPS    ║
        # ║   Models: 174         Kudos Per Hour: 5283        Jobs Per Hour: 524966     ║
        # ║  Threads: 3                 Warnings: 9999               Errors: 100        ║
        # ║ CPU Load: 99% (99%)         Free RAM: 2 GB (99%)                            ║
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
        # ║                                                                             ║
        # ║            (m)aintenance  (s)ource  (d)ebug  (p)ause log  (a)lerts  (q)uit  ║
        # ╙─────────────────────────────────────────────────────────────────────────────╜

        # Define three colums centres
        col_left = 12
        col_mid = self.width // 2
        col_right = self.width - 12

        # Define rows on which sections start
        row_local = 0
        row_gpu = row_local + 5
        row_total = row_gpu + 4
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

        label(row_local + 1, col_left, "Uptime:")
        label(row_local + 2, col_left, "Models:")
        label(row_local + 3, col_left, "Threads:")
        label(row_local + 4, col_left, "CPU Load:")
        label(row_local + 1, col_mid, "Jobs Completed:")
        label(row_local + 2, col_mid, "Kudos Per Hour:")
        label(row_local + 3, col_mid, "Warnings:")
        label(row_local + 4, col_mid, "Free RAM:")
        label(row_local + 1, col_right, "Performance:")
        label(row_local + 2, col_right, "Jobs Per Hour:")
        label(row_local + 3, col_right, "Errors:")

        label(row_gpu + 1, col_left, "Load:")
        label(row_gpu + 2, col_left, "Temp:")
        label(row_gpu + 3, col_left, "Power:")
        label(row_gpu + 1, col_mid, "VRAM Total:")
        label(row_gpu + 2, col_mid, "VRAM Used:")
        label(row_gpu + 3, col_mid, "VRAM Free:")
        label(row_gpu + 1, col_right, "Fan Speed:")
        label(row_gpu + 2, col_right, "PCI Gen:")
        label(row_gpu + 3, col_right, "PCI Width:")

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
        self.print(self.main, row_local + 1, col_right, f"{self.performance}")

        self.print(self.main, row_local + 2, col_left, f"{self.total_models}")
        self.print(self.main, row_local + 2, col_mid, f"{self.kudos_per_hour}")
        self.print(self.main, row_local + 2, col_right, f"{self.jobs_per_hour}")

        self.print(self.main, row_local + 3, col_left, f"{self.threads}")
        self.print(self.main, row_local + 3, col_mid, f"{self.warning_count}")
        self.print(self.main, row_local + 3, col_right, f"{self.error_count}")

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

        gpu = self.gpu.get_info()
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

            self.draw_line(self.main, row_gpu, gpu["product"])

            self.print(self.main, row_gpu + 1, col_left, f"{gpu['load']:4} ({gpu['avg_load']})")
            self.print(self.main, row_gpu + 1, col_mid, f"{gpu['vram_total']}")
            self.print(self.main, row_gpu + 1, col_right, f"{gpu['fan_speed']}")

            self.print(self.main, row_gpu + 2, col_left, f"{gpu['temp']:4} ({gpu['avg_temp']})")
            self.print(self.main, row_gpu + 2, col_mid, f"{gpu['vram_used']}")
            self.print(self.main, row_gpu + 2, col_right, f"{gpu['pci_gen']}")

            self.print(self.main, row_gpu + 3, col_left, f"{gpu['power']:4} ({gpu['avg_power']})")
            self.print(self.main, row_gpu + 3, col_mid, f"{gpu['vram_free']}", vram_colour)
            self.print(self.main, row_gpu + 3, col_right, f"{gpu['pci_width']}")

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
            elif cat == "INIT" or cat == "INIT_OK":
                colour = TerminalUI.COLOUR_MAGENTA
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
        if not self.worker_name:
            return None
        workers_url = f"{self.url}/api/v2/workers"
        try:
            r = requests.get(workers_url, headers={"client-agent": TerminalUI.CLIENT_AGENT})
        except requests.exceptions.RequestException:
            return None
        if r.ok:
            worker_json = r.json()
            for item in worker_json:
                if item["name"] == self.worker_name:
                    return item["id"]
            return None
        return None

    def set_maintenance_mode(self, enabled):
        if not self.apikey or not self.worker_id:
            return
        header = {"apikey": self.apikey, "client-agent": TerminalUI.CLIENT_AGENT}
        payload = {"maintenance": enabled, "name": self.worker_name}
        worker_URL = f"{self.url}/api/v2/workers/{self.worker_id}"
        requests.put(worker_URL, json=payload, headers=header)

    def get_remote_worker_info(self):
        if not self.worker_id:
            self.worker_id = self.load_worker_id()
        if not self.worker_id:
            return
        worker_URL = f"{self.url}/api/v2/workers/{self.worker_id}"
        try:
            r = requests.get(worker_URL, headers={"client-agent": TerminalUI.CLIENT_AGENT})
        except requests.exceptions.RequestException:
            return
        if not r.ok:
            return
        data = r.json()

        worker_type = data.get("type", "unknown")
        self.maintenance_mode = data.get("maintenance_mode", False)
        self.total_worker_kudos = data.get("kudos_details", {}).get("generated", 0)
        if self.total_worker_kudos is not None:
            self.total_worker_kudos = int(self.total_worker_kudos)
        self.total_jobs = data.get("requests_fulfilled", 0)
        self.total_kudos = int(data.get("kudos_rewards", 0))
        perf = data.get("performance", "0").replace("No requests fulfilled yet", "0")
        self.threads = data.get("threads", 0)
        self.total_uptime = data.get("uptime", 0)
        self.total_failed_jobs = data.get("uncompleted_jobs", 0)

        if worker_type == "image":
            self.performance = perf.replace("megapixelsteps per second", "MPS")
            self.total_models = len(data.get("models", []))
        elif worker_type == "interrogation":
            self.performance = perf.replace("seconds per form", "SPF")
            self.total_models = 0

    def get_remote_horde_stats(self):
        url = f"{self.url}/api/v2/status/performance"
        try:
            r = requests.get(url, headers={"client-agent": TerminalUI.CLIENT_AGENT})
        except requests.exceptions.RequestException:
            return
        if not r.ok:
            return
        data = r.json()
        self.queued_requests = int(data.get("queued_requests", 0))
        self.worker_count = int(data.get("worker_count", 1))
        self.thread_count = int(data.get("thread_count", 0))
        self.queued_mps = int(data.get("queued_megapixelsteps", 0))
        self.last_minute_mps = int(data.get("past_minute_megapixelsteps", 0))
        self.queue_time = (self.queued_mps / self.last_minute_mps) * 60

    def update_stats(self):
        if time.time() - self.last_stats_refresh > TerminalUI.REMOTE_STATS_REFRESH:
            self.last_stats_refresh = time.time()
            threading.Thread(target=self.get_remote_worker_info, daemon=True).start()
            threading.Thread(target=self.get_remote_horde_stats, daemon=True).start()

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
                commit_hash = f.read().strip()

            return commit_hash
        except Exception:
            return ""

    def get_input(self):
        x = self.main.getch()
        self.last_key = x
        if x == curses.KEY_RESIZE:
            self.resize()
        elif x == ord("d"):
            self.show_debug = not self.show_debug
        elif x == ord("s"):
            self.show_module = not self.show_module
        elif x == ord("a"):
            self.audio_alerts = not self.audio_alerts
        elif x == ord("q"):
            return True
        elif x == ord("m"):
            self.maintenance_mode = not self.maintenance_mode
            self.set_maintenance_mode(self.maintenance_mode)
        elif x == ord("p"):
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
        self.main = stdscr
        try:
            self.initialise()
            while True:
                if self.poll():
                    return
                time.sleep(1 / self.gpu.samples_per_second)
        except KeyboardInterrupt:
            return

    def run(self):
        curses.wrapper(self.main_loop)


if __name__ == "__main__":
    # From the project root: runtime.cmd python -m worker.ui
    # This will connect to the currently running worker via it's log file,
    # very useful for development or connecting to a worker running as a background
    # service.

    workername = ""
    apikey = ""

    # Grab worker name and apikey if available
    if os.path.isfile("bridgeData.yaml"):
        with open("bridgeData.yaml", "rt", encoding="utf-8", errors="ignore") as configfile:
            configdata = yaml.safe_load(configfile)
            workername = configdata.get("worker_name", "")
            apikey = configdata.get("api_key", "")

    term = TerminalUI(workername, apikey)
    # Standalone UI we need to inspect the log file
    term.use_log_file = True
    term.run()
