# curses.py
# A simple terminal worker UI
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

import requests
import yaml

import worker.gpu as gpu


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


class Terminal:

    REGEX = re.compile(r"(INIT|DEBUG|INFO|WARNING|ERROR).*(\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d).*\| (.*) - (.*)$")
    KUDOS_REGEX = re.compile(r".*average kudos per hour: (\d+)")
    JOBDONE_REGEX = re.compile(r".*Generation for id.*finished successfully")

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

    JUNK = [
        "Result = False",
        "Result = True",
    ]

    CLIENT_AGENT = "terminalui:1:db0"

    MIN_WIDTH = 79
    MIN_HEIGHT = 16

    def __init__(self, worker_name=None, apikey=None, url="https://stablehorde.net"):
        self.url = url
        self.main = None
        self.status = None
        self.log = None
        self.width = 0
        self.height = 0
        self.status_height = 17
        self.show_module = False
        self.show_debug = False
        self.show_dev = False
        self.last_key = None
        self.pause_log = False
        self.output = DequeOutputCollector()
        self.stdout = DequeOutputCollector()
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
        self.allow_redraw = True
        self.gpu = gpu.GPUInfo()
        self.error_count = 0
        self.warning_count = 0
        self.commit_hash = self.get_commit_hash()

    def initialise(self):
        locale.setlocale(locale.LC_ALL, "")
        self.initialise_main_window()
        self.initialise_status_window()
        self.initialise_log_window()
        self.resize()
        self.open_log()
        self.get_remote_worker_info()

    def open_log(self):
        self.input = open("logs/bridge.log", "rt", encoding="utf-8", errors="ignore")
        self.input.seek(0, os.SEEK_END)

    def load_log(self):
        while line := self.input.readline():
            ignore = False
            for skip in Terminal.JUNK:
                if skip.lower() in line.lower():
                    ignore = True
            if ignore:
                continue
            if regex := Terminal.REGEX.match(line):
                if not self.show_debug and regex.group(1) == "DEBUG":
                    continue
                if regex.group(1) == "ERROR":
                    self.error_count += 1
                elif regex.group(1) == "WARNING":
                    self.warning_count += 1
                self.output.write(f"{regex.group(1)}::::{regex.group(2)}::::{regex.group(3)}::::{regex.group(4)}")
            if regex := Terminal.KUDOS_REGEX.match(line):
                self.kudos_per_hour = int(regex.group(1))
            if regex := Terminal.JOBDONE_REGEX.match(line):
                self.jobs_done += 1
                self.jobs_per_hour = int(3600 / ((time.time() - self.start_time) / self.jobs_done))
        self.output.set_size(self.height - self.status_height)

    def window_size_ok(self, win):
        height, width = win.getmaxyx()
        if height < Terminal.MIN_HEIGHT or width < Terminal.MIN_WIDTH:
            self.allow_redraw = False
        else:
            self.allow_redraw = True
        return self.allow_redraw

    def initialise_main_window(self):
        self.main = curses.initscr()
        # Don't each key presses
        curses.noecho()
        # Respond on keydown
        curses.cbreak()
        # Determine terminal size
        self.height, self.width = self.main.getmaxyx()
        self.window_size_ok(self.main)
        self.main.keypad(True)
        curses.curs_set(0)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_BLUE, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        curses.init_pair(6, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_BLACK)

    def initialise_status_window(self):
        self.status = curses.newwin(self.status_height, self.width, 0, 0)
        # Request more helpful key names
        self.status.keypad(True)
        self.status.nodelay(True)

    def initialise_log_window(self):
        self.log = curses.newwin(self.height - self.status_height, self.width, self.status_height, 0)
        self.log.idlok(True)
        self.log.scrollok(True)
        sys.stdout = self.stdout

    def resize(self):
        # Determine terminal size
        self.window_size_ok(self.main)
        with contextlib.suppress(curses.error):
            self.main.erase()
            self.log.erase()
            self.status.erase()
            self.height, self.width = self.main.getmaxyx()
            self.status.resize(self.status_height, self.width)
            self.log.resize(self.height - self.status_height, self.width)
        self.main.refresh()
        self.status.refresh()
        self.log.refresh()

    def finalise(self):
        curses.nocbreak()
        self.main.keypad(False)
        self.status.keypad(False)
        curses.echo()
        curses.endwin()

    def draw_line(self, win, y, label):
        if not self.allow_redraw:
            return
        height, width = win.getmaxyx()
        win.addstr(
            y, 0, Terminal.ART["left-join"] + Terminal.ART["horizontal"] * (width - 2) + Terminal.ART["right-join"]
        )
        win.addstr(y, 2, label)

    def draw_box(self, win):
        if not self.allow_redraw:
            return
        # An attempt to work cross platform, box() doesn't.
        height, width = win.getmaxyx()

        # Draw the top border
        win.addstr(
            0, 0, Terminal.ART["top_left"] + Terminal.ART["horizontal"] * (width - 2) + Terminal.ART["top_right"]
        )

        # Draw the side borders
        for y in range(1, height - 1):
            win.addstr(y, 0, Terminal.ART["vertical"])
            win.addstr(y, width - 1, Terminal.ART["vertical"])

        # Draw the bottom border
        win.addstr(height - 1, 0, Terminal.ART["bottom_left"] + Terminal.ART["horizontal"] * (width - 2))
        with contextlib.suppress(curses.error):
            win.addstr(height - 1, width - 1, Terminal.ART["bottom_right"])

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
        if switch:
            colour = curses.color_pair(Terminal.COLOUR_CYAN)
        else:
            colour = curses.color_pair(Terminal.COLOUR_WHITE)
        self.status.addstr(y, x, label, colour)
        return x + len(label) + 2

    def print_status(self):
        # This is the design template: (80 columns)
        # ╔═AIDream-01══════════════════════════════════════════════════════════════════╗
        # ║ Uptime: 0:14:35       Jobs Completed: 6             Performance: 0.3 MPS    ║
        # ║ Models: 174           Kudos Per Hour: 5283        Jobs Per Hour: 524966     ║
        # ║                             Warnings: 9999               Errors: 100        ║
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
        # ║                       (m)aintenance  (s)ource  (d)ebug  (p)ause log  (q)uit ║
        # ╙─────────────────────────────────────────────────────────────────────────────╜
        self.status.erase()

        if not self.allow_redraw:
            return

        # Define three colums centres

        col_left = 10
        col_mid = self.width // 2
        col_right = self.width - 12

        # Define rows on which sections start
        row_local = 0
        row_gpu = row_local + 4
        row_total = row_gpu + 4
        row_horde = row_total + 3

        def label(y, x, label):
            self.status.addstr(y, x - len(label) - 1, label)

        self.draw_box(self.status)
        self.draw_line(self.status, row_gpu, "")
        self.draw_line(self.status, row_total, "Worker Total")
        self.draw_line(self.status, row_horde, "Entire Horde")
        self.status.addstr(row_local, 2, f"{self.worker_name}")
        self.status.addstr(row_local, self.width - 8, f"{self.commit_hash[:6]}")

        label(row_local + 1, col_left, "Uptime:")
        label(row_local + 2, col_left, "Models:")
        label(row_local + 1, col_mid, "Jobs Completed:")
        label(row_local + 2, col_mid, "Kudos Per Hour:")
        label(row_local + 3, col_mid, "Warnings:")
        label(row_local + 1, col_right, "Performance:")
        label(row_local + 2, col_right, "Jobs Per Hour:")
        label(row_local + 3, col_right, "Error:")

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

        self.status.addstr(row_local + 1, col_left, f"{self.get_uptime()}")
        self.status.addstr(row_local + 1, col_mid, f"{self.jobs_done}")
        self.status.addstr(row_local + 1, col_right, f"{self.performance}")

        self.status.addstr(row_local + 2, col_left, f"{self.total_models}")
        self.status.addstr(row_local + 2, col_mid, f"{self.kudos_per_hour}")
        self.status.addstr(row_local + 2, col_right, f"{self.jobs_per_hour}")

        # self.status.addstr(row_local+3, col_left, f"")
        self.status.addstr(row_local + 3, col_mid, f"{self.warning_count}")
        self.status.addstr(row_local + 3, col_right, f"{self.error_count}")

        gpu = self.gpu.get_info()
        if gpu:

            self.draw_line(self.status, row_gpu, gpu["product"])

            self.status.addstr(row_gpu + 1, col_left, f"{gpu['load']:4} ({gpu['avg_load']})")
            self.status.addstr(row_gpu + 1, col_mid, f"{gpu['vram_total']}")
            self.status.addstr(row_gpu + 1, col_right, f"{gpu['fan_speed']}")

            self.status.addstr(row_gpu + 2, col_left, f"{gpu['temp']:4} ({gpu['avg_temp']})")
            self.status.addstr(row_gpu + 2, col_mid, f"{gpu['vram_used']}")
            self.status.addstr(row_gpu + 2, col_right, f"{gpu['pci_gen']}")

            self.status.addstr(row_gpu + 3, col_left, f"{gpu['power']:4} ({gpu['avg_power']})")
            self.status.addstr(row_gpu + 3, col_mid, f"{gpu['vram_free']}")
            self.status.addstr(row_gpu + 3, col_right, f"{gpu['pci_width']}")

        self.status.addstr(row_total + 1, col_mid, f"{self.total_kudos}")
        self.status.addstr(row_total + 1, col_right, f"{self.total_jobs}")

        self.status.addstr(row_total + 2, col_mid, f"{self.seconds_to_timestring(self.total_uptime)}")
        self.status.addstr(row_total + 2, col_right, f"{self.total_failed_jobs}")

        self.status.addstr(row_horde + 1, col_mid, f"{self.queued_requests}")
        self.status.addstr(row_horde + 1, col_right, f"{self.seconds_to_timestring(self.queue_time)}")

        self.status.addstr(row_horde + 2, col_mid, f"{self.worker_count}")
        self.status.addstr(row_horde + 2, col_right, f"{self.thread_count}")

        inputs = [
            "(m)aintenance",
            "(s)ource",
            "(d)ebug",
            "(p)ause log",
            "(q)uit",
        ]
        x = self.width - len("  ".join(inputs)) - 2
        y = row_horde + 4
        x = self.print_switch(y, x, inputs[0], self.maintenance_mode)
        x = self.print_switch(y, x, inputs[1], self.show_module)
        x = self.print_switch(y, x, inputs[2], self.show_debug)
        x = self.print_switch(y, x, inputs[3], self.pause_log)
        x = self.print_switch(y, x, inputs[4], False)
        self.status.refresh()

    def print_log(self):
        if self.pause_log:
            return

        termrows = self.height - self.status_height
        self.load_log()
        output = list(self.output.deque)

        # How many lines of output can we fit, with wrapping and stuff
        linecount = 0
        maxrows = 0
        for i, fullline in enumerate(reversed(output)):
            line = fullline.split(Terminal.DELIM)[-1:][0]
            linecount += len(textwrap.wrap(line, self.width - 21))
            if self.show_module:
                linecount += 1
            if linecount > termrows:
                maxrows = i
                break
        output = output[-maxrows:]
        if self.show_dev:
            self.status.addstr(
                5,
                2,
                f"Output rows {len(output)}, wrapped {linecount}  Terminal rows {termrows}, "
                f"height {self.height}  Key: {self.last_key}  Status Height {self.status_height}",
            )

        y = 0
        inputrow = 0
        last_timestamp = ""
        while y < termrows:
            # If we ran out of stuff to print
            if inputrow >= len(output):
                self.log.move(y, 0)
                self.log.clrtoeol()
                y += 1
                continue
            # Print any log info we have
            self.log.move(y, 0)
            self.log.clrtoeol()
            cat, nextwhen, source, msg = output[inputrow].split(Terminal.DELIM)
            colour = Terminal.COLOUR_WHITE
            if cat == "WARNING":
                colour = Terminal.COLOUR_YELLOW
            elif cat == "ERROR":
                colour = Terminal.COLOUR_RED
            elif cat == "INIT":
                colour = Terminal.COLOUR_MAGENTA
            elif cat == "DEBUG":
                colour = Terminal.COLOUR_WHITE
            # Timestamp
            when = nextwhen if nextwhen != last_timestamp else ""
            last_timestamp = nextwhen
            length = len(last_timestamp) + 2
            self.log.addstr(y, 1, f"{when}", curses.color_pair(Terminal.COLOUR_GREEN))
            # Module
            if self.show_module:
                self.log.addstr(y, length, f"{source}", curses.color_pair(Terminal.COLOUR_GREEN))
                y += 1
                if y > termrows:
                    break
            # Message
            text = textwrap.wrap(msg, self.width - length)
            cont = False
            for line in text:
                if cont or not when:
                    self.log.move(y, 0)
                    self.log.clrtoeol()
                self.log.addstr(y, length, line, curses.color_pair(colour))
                y += 1
                if y > termrows:
                    break
                cont = True
            inputrow += 1
            self.log.clrtoeol()
        self.log.refresh()

    def get_input(self):
        x = self.status.getch()
        self.last_key = x
        if x == curses.KEY_RESIZE:
            self.resize()
        elif x == ord("d"):
            self.show_debug = not self.show_debug
        elif x == ord("s"):
            self.show_module = not self.show_module
        elif x == ord("v"):
            self.show_dev = not self.show_dev
        elif x == ord("q"):
            self.finalise()
            return True
        elif x == ord("m"):
            self.maintenance_mode = not self.maintenance_mode
            self.set_maintenance_mode(self.maintenance_mode)
        elif x == ord("p"):
            self.pause_log = not self.pause_log

    def load_worker_id(self):
        if not self.worker_name:
            return
        workers_url = f"{self.url}/api/v2/workers"
        try:
            r = requests.get(workers_url, headers={"client-agent": Terminal.CLIENT_AGENT})
        except requests.exceptions.RequestException:
            return
        if r.ok:
            worker_json = r.json()
            for item in worker_json:
                if item["name"] == self.worker_name:
                    return item["id"]

    def set_maintenance_mode(self, enabled):
        if not self.apikey or not self.worker_id:
            return
        header = {"apikey": self.apikey, "client-agent": Terminal.CLIENT_AGENT}
        payload = {"maintenance": enabled, "name": self.worker_name}
        worker_URL = f"{self.url}/api/v2/workers/{self.worker_id}"
        requests.put(worker_URL, json=payload, headers=header)

    def get_remote_worker_info(self):
        # Sneak a git commit hash in here
        self.commit_hash = self.get_commit_hash()

        if not self.worker_id:
            self.worker_id = self.load_worker_id()
        if not self.worker_id:
            return
        worker_URL = f"{self.url}/api/v2/workers/{self.worker_id}"
        try:
            r = requests.get(worker_URL, headers={"client-agent": Terminal.CLIENT_AGENT})
        except requests.exceptions.RequestException:
            return
        if not r.ok:
            return
        data = r.json()
        self.maintenance_mode = data.get("maintenance_mode", False)
        self.total_jobs = data.get("requests_fulfilled", 0)
        self.total_kudos = int(data.get("kudos_rewards", 0))
        self.total_worker_kudos = int(data.get("kudos_details", {}).get("generated", 0))
        self.performance = data.get("performance").replace("megapixelsteps per second", "MPS")
        self.threads = data.get("threads", 0)
        self.total_failed_jobs = data.get("uncompleted_jobs", 0)
        self.total_uptime = data.get("uptime", 0)
        self.total_models = len(data.get("models", []))

    def get_remote_horde_stats(self):
        url = f"{self.url}/api/v2/status/performance"
        try:
            r = requests.get(url, headers={"client-agent": Terminal.CLIENT_AGENT})
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
        if time.time() - self.last_stats_refresh > Terminal.REMOTE_STATS_REFRESH:
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

    def poll(self):
        try:
            if self.get_input():
                return True
            if self.allow_redraw:
                with contextlib.suppress(curses.error):
                    self.update_stats()
                    self.print_status()
                    self.print_log()
        except KeyboardInterrupt:
            self.finalise()
            return True
        except Exception:
            self.finalise()
            raise

    def run(self):
        try:
            self.initialise()
            while True:
                if self.poll():
                    return
                time.sleep(0.1)
        finally:
            self.finalise()


if __name__ == "__main__":
    # From the project root: runtime.cmd python -m worker.terminalui
    # This will connect to the currently running worker via it's log file,
    # very useful for development or connecting to a worker running as a background
    # service.

    workername = ""
    apikey = ""

    # Grab worker name and apikey if available
    if os.path.isfile("bridgeData.yaml"):
        with open("bridgeData.yaml", "rt", encoding="utf-8", errors="ignore") as configfile:
            config = yaml.safe_load(configfile)
            workername = config.get("worker_name", "")
            apikey = config.get("api_key", "")

    term = Terminal(workername, apikey)
    termthread = threading.Thread(target=term.run, daemon=True)
    termthread.start()
    termthread.join()
