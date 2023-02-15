# model-stats.py
# Calculate some basic model usage statistics from the local worker log file.
# Or optionally use the central horde stats.
# Usage: model-stats.py [-h] [--horde] [--today] [--yesterday]
import argparse
import datetime
import glob
import mmap
import re

import requests
from bridgeData import models_to_load
from tqdm import tqdm

# Location of stable horde worker bridge log
LOG_FILE = "logs/bridge*.log"

# TIME PERIODS
PERIOD_ALL = 0
PERIOD_TODAY = 1
PERIOD_YESTERDAY = 2
PERIOD_HORDE_DAY = 3
PERIOD_HORDE_MONTH = 4

# regex to identify model lines
REGEX = re.compile(r".*(\d\d\d\d-\d\d-\d\d).*Starting generation: (.*) @")


class LogStats:
    def __init__(self, period=PERIOD_ALL, logfile=LOG_FILE):
        self.used_models = {}
        self.unused_models = {}
        self.logfile = logfile
        self.period = period

    def get_date(self):
        # Dates in log format for filtering
        if self.period == PERIOD_TODAY:
            adate = datetime.datetime.now()
        elif self.period == PERIOD_YESTERDAY:
            adate = datetime.datetime.now() - datetime.timedelta(1)
        else:
            adate = None
        if adate:
            adate = adate.strftime("%Y-%m-%d")
        return adate

    def get_num_lines(self, file_path):
        fp = open(file_path, "r+")
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines

    def download_stats(self, period):
        self.unused_models = []  # not relevant

        req = requests.get("https://stablehorde.net/api/v2/stats/img/models")
        self.used_models = req.json()[period] if req.ok else {}

    def parse_log(self):
        self.used_models = {}
        # Grab any statically loaded models
        self.unused_models = models_to_load[:]
        # Models to exclude
        if "safety_checker" in self.unused_models:
            self.unused_models.remove("safety_checker")

        # If using the horde central db, skip local logs
        if self.period == PERIOD_HORDE_DAY:
            self.download_stats("day")
            return
        elif self.period == PERIOD_HORDE_MONTH:
            self.download_stats("month")
            return

        # Identify all log files and total number of log lines
        total_log_lines = 0
        for logfile in glob.glob(self.logfile):
            total_log_lines += self.get_num_lines(logfile)

        progress = tqdm(total=total_log_lines, leave=True, unit=" lines", unit_scale=True)
        for logfile in glob.glob(self.logfile):
            with open(logfile, "rt") as infile:
                for line in infile:
                    # Grab the lines we're interested in
                    regex = REGEX.match(line)
                    if regex:
                        if self.period and regex.group(1) != self.get_date():
                            continue
                        # Extract model name
                        model = regex.group(2)

                        # Remember we used this model
                        if model in self.unused_models:
                            self.unused_models.remove(model)

                        # Keep count of how many times we used a model
                        if model in self.used_models:
                            self.used_models[model] += 1
                        else:
                            self.used_models[model] = 1

                    progress.update()

    def print_stats(self):
        # Parse our log file if we haven't done that yet
        if not self.used_models:
            self.parse_log()

        # Whats our longest model name?
        max_len = max([len(x) for x in self.used_models])

        scores = sorted(((self.used_models[model], model) for model in self.used_models), reverse=True)
        total = 0
        for count, name in scores:
            total += count

        j = 1
        for count, name in scores:
            perc = round((count / total) * 100, 1)
            print(f"{j:>2}. {name:<{max_len}} {perc}% ({count})")
            j += 1

        print()
        if self.unused_models:
            print("The following models were not used at all:")
            for m in self.unused_models:
                print(f"  {m}")
        else:
            print("There were no unused models.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate local worker or horde model usage statistics")
    parser.add_argument("-t", "--today", help="Local statistics for today only", action="store_true")
    parser.add_argument("-y", "--yesterday", help="Local statistics for yesterday only", action="store_true")
    parser.add_argument("-d", "--horde", help="Show statistics for the entire horde for the day", action="store_true")
    parser.add_argument(
        "-m", "--hordemonth", help="Show statistics for the entire horde for the month", action="store_true"
    )
    args = vars(parser.parse_args())

    period = PERIOD_ALL
    if args["today"]:
        period = PERIOD_TODAY
    elif args["yesterday"]:
        period = PERIOD_YESTERDAY
    elif args["horde"]:
        period = PERIOD_HORDE_DAY
    elif args["hordemonth"]:
        period = PERIOD_HORDE_MONTH

    logs = LogStats(period)
    logs.print_stats()
