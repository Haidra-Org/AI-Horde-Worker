# curses.py
# A simple terminal worker UI
import curses
import os
import re
import sys
import textwrap
import threading
import time
from collections import deque

import requests
import yaml
import locale

from nataili.util.logger import logger  # XXX Remove

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
        'top_left': '╓',
        'top_right': '╖',
        'bottom_left': '╙',
        'bottom_right': '╜',
        'horizontal': '─',
        'vertical': '║',
        'left-join': '╟',
        'right-join': '╢'
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

    def __init__(self, worker_name=None, apikey=None, url="https://stablehorde.net"):
        self.url = url
        self.main = None
        self.status = None
        self.log = None
        self.width = 0
        self.height = 0
        self.status_height = 12
        self.show_module = False
        self.show_debug = False
        self.show_dev = False
        self.last_key = None
        self.pause_display = False
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
        while True:
            line = self.input.readline()
            if line:
                ignore = False
                for skip in Terminal.JUNK:
                    if skip.lower() in line.lower():
                        ignore = True
                if ignore:
                    continue
                if regex := Terminal.REGEX.match(line):
                    if not self.show_debug and regex.group(1) == "DEBUG":
                        continue
                    self.output.write(f"{regex.group(1)}::::{regex.group(2)}::::{regex.group(3)}::::{regex.group(4)}")
                if regex := Terminal.KUDOS_REGEX.match(line):
                    self.kudos_per_hour = int(regex.group(1))
                if regex := Terminal.JOBDONE_REGEX.match(line):
                    self.jobs_done += 1
                    self.jobs_per_hour = int(3600 / ((time.time() - self.start_time) / self.jobs_done))
            else:
                break
        self.output.set_size(self.height - self.status_height)

    def initialise_main_window(self):
        self.main = curses.initscr()
        # Don't each key presses
        curses.noecho()
        # Respond on keydown
        curses.cbreak()
        # Determine terminal size
        self.height, self.width = self.main.getmaxyx()
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
        sys.stdout = self.stdoutput

    def resize(self):
        # Determine terminal size
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
        height, width = win.getmaxyx()
        win.addstr(y, 0, Terminal.ART['left-join'] + Terminal.ART['horizontal'] * (width - 2) + Terminal.ART['right-join'])
        win.addstr(y, 2, label)

    def draw_box(self, win):
        # An attempt to work cross platform, box() doesn't.
        height, width = win.getmaxyx()

        # Draw the top border
        win.addstr(0, 0, Terminal.ART['top_left'] + Terminal.ART['horizontal'] * (width - 2) + Terminal.ART['top_right'])

        # Draw the side borders
        for y in range(1, height - 1):
            win.addstr(y, 0, Terminal.ART['vertical'])
            win.addstr(y, width - 1, Terminal.ART['vertical'])

        # Draw the bottom border
        win.addstr(height - 1, 0, Terminal.ART['bottom_left'] + Terminal.ART['horizontal'] * (width - 2))
        try:
            win.addstr(height - 1, width - 1, Terminal.ART['bottom_right'])
        except curses.error:
            pass

    def seconds_to_timestring(self, seconds):
        hours = int(seconds // 3600)
        days = int(hours / 24)
        hours = hours % 24
        minutes = int((seconds % 3600) // 60)
        result = ""
        if days:
            result += f"{days}d "
        if hours:
            result += f"{hours}h "
        if minutes:
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
        # ║ Uptime:  0:14:35  Jobs Completed: 6           Performance:        0.3 MPS   ║
        # ║ Models:  174      Kudos Per Hour: 5283        Jobs Per Hour:      524966    ║
        # ╟─Worker Total────────────────────────────────────────────────────────────────╢
        # ║                   Worker Kudos:   9385297     Total Jobs Failed:  972       ║
        # ║                   Total Uptime:   34d 19h 14m Total Jobs Done:    701138    ║
        # ╟─Entire Horde────────────────────────────────────────────────────────────────╢
        # ║                   Jobs Queued:    99999       Queue Time:         99m       ║
        # ║                   Total Workers:  1000        Total Threads:      1000      ║
        # ║                                                                             ║
        # ║             (m)aintenance mode  (s)ource file  (d)ebug  (p)ause log  (q)uit ║
        # ╙─────────────────────────────────────────────────────────────────────────────╜        
        self.status.erase()        
        #self.status.border("|", "|", "-", "-", "+", "+", "+", "+")
        self.draw_box(self.status)
        self.draw_line(self.status, 3, "Worker Total")
        self.draw_line(self.status, 6, "Entire Horde")
        self.status.addstr(0, 2, f"{self.worker_name}")

        self.status.addstr(1, 2, "Uptime:           Jobs Completed:             Performance:       ")
        self.status.addstr(2, 2, "Models:           Kudos Per Hour:             Jobs Per Hour:     ")
        self.status.addstr(4, 2, "                  Worker Kudos:               Total Jobs Done:  ")
        self.status.addstr(5, 2, "                  Total Uptime:               Total Jobs Failed:  ")
        self.status.addstr(7, 2, "                  Jobs Queued:                Queue Time: ")
        self.status.addstr(8, 2, "                  Total Workers:              Total Threads:   ")

        self.status.addstr(1, 11, f"{self.get_uptime()}")
        self.status.addstr(1, 36, f"{self.jobs_done}")
        self.status.addstr(1, 68, f"{self.performance}")

        self.status.addstr(2, 11, f"{self.total_models}")
        self.status.addstr(2, 36, f"{self.kudos_per_hour}")
        self.status.addstr(2, 68, f"{self.jobs_per_hour}")

        self.status.addstr(4, 36, f"{self.total_kudos}")
        self.status.addstr(4, 68, f"{self.total_jobs}")

        self.status.addstr(5, 36, f"{self.seconds_to_timestring(self.total_uptime)}")
        self.status.addstr(5, 68, f"{self.total_failed_jobs}")

        self.status.addstr(7, 36, f"{self.queued_requests}")
        self.status.addstr(7, 68, f"{self.seconds_to_timestring(self.queue_time)}")

        self.status.addstr(8, 36, f"{self.worker_count}")
        self.status.addstr(8, 68, f"{self.thread_count}")

        inputs = [
            "(m)aintenance mode",
            "(s)ource file",
            "(d)ebug",
            "(p)ause log",
            "(q)uit",
        ]
        x = self.width - len("  ".join(inputs)) - 2
        y = 10
        x = self.print_switch(y, x, inputs[0], self.maintenance_mode)
        x = self.print_switch(y, x, inputs[1], self.show_module)
        x = self.print_switch(y, x, inputs[2], self.show_debug)
        x = self.print_switch(y, x, inputs[3], self.pause_display)
        x = self.print_switch(y, x, inputs[4], False)
        self.status.refresh()

    def print_log(self):
        if self.pause_display:
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
            if inputrow < len(output):
                self.log.move(y, 0)
                self.log.clrtoeol()
                try:
                    cat, nextwhen, source, msg = output[inputrow].split(Terminal.DELIM)
                except ValueError:
                    logger.error(f"Can not split string '{output[inputrow]}'")
                    raise
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
            else:
                self.log.move(y, 0)
                self.log.clrtoeol()
                y += 1
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
            self.pause_display = not self.pause_display

    def load_worker_id(self):
        if not self.worker_name:
            return
        workers_url = f"{self.url}/api/v2/workers"
        r = requests.get(workers_url, headers={"client-agent": Terminal.CLIENT_AGENT})
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
        if not self.worker_id:
            return
        worker_URL = f"{self.url}/api/v2/workers/{self.worker_id}"
        r = requests.get(worker_URL, headers={"client-agent": Terminal.CLIENT_AGENT})
        if r.ok:
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
        r = requests.get(url, headers={"client-agent": Terminal.CLIENT_AGENT})
        if r.ok:
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

    def poll(self):
        try:
            if self.get_input():
                return True
            self.update_stats()
            self.print_status()
            self.print_log()
        except KeyboardInterrupt:
            self.finalise()
            return True
        except Exception:
            self.finalise()
            raise


if __name__ == "__main__":
    # This can be used to run this terminal view along side an already running worker.
    # This is very useful for development, as you don't need to stop and start any
    # locally running worker to test this terminal UI.
    # From the project root: python -m worker.terminalui

    workername = ""
    apikey = ""

    # Grab worker name and apikey if available
    if os.path.isfile("bridgeData.yaml"):
        with open("bridgeData.yaml", "rt", encoding="utf-8", errors="ignore") as configfile:
            config = yaml.safe_load(configfile)
            workername = config.get("worker_name", "")
            apikey = config.get("api_key", "")

    term = Terminal(workername, apikey)

    while True:
        if term.poll():
            exit(0)
        time.sleep(0.1)
