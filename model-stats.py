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
import yaml
from tqdm import tqdm

# Location of stable horde worker bridge log
LOG_FILE = "logs/bridge*.log"

# TIME PERIODS
# TODO: Use Enums
PERIOD_ALL = 0
PERIOD_TODAY = 1
PERIOD_YESTERDAY = 2
PERIOD_HORDE_DAY = 3
PERIOD_HORDE_MONTH = 4
PERIOD_KUDOS_HOUR = 5
PERIOD_TEXT_HORDE_MONTH = 6

# regex to identify model lines
REGEX = re.compile(r".*(\d\d\d\d-\d\d-\d\d).*Starting generation: (.*) @")

# regex to identify kudos lines
KUDOS_REGEX = re.compile(r".*(\d\d\d\d-\d\d-\d\d \d\d:\d\d).* and contributed for (\d+\.\d+)")


class LogStats:
    def __init__(self, period=PERIOD_ALL, logfile=LOG_FILE):
        self.used_models = {}
        self.unused_models = {}
        self.logfile = logfile
        self.period = period
        self.kudos = {}

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
        with open(file_path, "r+") as fp:
            buf = mmap.mmap(fp.fileno(), 0)
            lines = 0
            while buf.readline():
                lines += 1
            return lines

    def download_stats(self, period, model_type="img"):
        self.unused_models = []  # not relevant

        req = requests.get(f"https://stablehorde.net/api/v2/stats/{model_type}/models", verify=False)
        self.used_models = req.json()[period] if req.ok else {}

    def parse_log(self):
        self.used_models = {}
        # Grab any statically loaded models
        with open("bridgeData.yaml", "rt", encoding="utf-8", errors="ignore") as configfile:
            config = yaml.safe_load(configfile)
        self.unused_models = config["models_to_load"]
        # Models to exclude
        if "safety_checker" in self.unused_models:
            self.unused_models.remove("safety_checker")

        # If using the horde central db, skip local logs
        if self.period == PERIOD_HORDE_DAY:
            self.download_stats("day")
            return
        if self.period == PERIOD_HORDE_MONTH:
            self.download_stats("month")
            return
        if self.period == PERIOD_TEXT_HORDE_MONTH:
            self.download_stats("month", "text")
            return

        # Identify all log files and total number of log lines
        total_log_lines = sum(self.get_num_lines(logfile) for logfile in glob.glob(self.logfile))
        progress = tqdm(total=total_log_lines, leave=True, unit=" lines", unit_scale=True)
        for logfile in glob.glob(self.logfile):
            with open(logfile, "rt", encoding="UTF-8", errors="ignore") as infile:
                for line in infile:
                    # Grab the lines we're interested in for models
                    if regex := REGEX.match(line):
                        if self.period in [PERIOD_TODAY, PERIOD_YESTERDAY] and regex.group(1) != self.get_date():
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

                    # Grab kudos lines
                    # Grab the lines we're interested in
                    if regex := KUDOS_REGEX.match(line):
                        # Extract kudis and time
                        timestamp = regex.group(1)[:-2]  # truncate to hour
                        kudos = regex.group(2)
                        if timestamp in self.kudos:
                            self.kudos[timestamp] += float(kudos)
                        else:
                            self.kudos[timestamp] = float(kudos)

                    progress.update()

    def print_stats(self):
        # Parse our log file if we haven't done that yet
        if not self.used_models:
            self.parse_log()

        # If we're reporting on kudos, do that
        if self.period == PERIOD_KUDOS_HOUR:
            for k, v in self.kudos.items():
                print(k, round(v))
            return

        # Whats our longest model name?
        max_len = max(len(x) for x in self.used_models)

        scores = sorted(((self.used_models[model], model) for model in self.used_models), reverse=True)
        total = sum(count for count, name in scores)
        for j, (count, name) in enumerate(scores, start=1):
            perc = round((count / total) * 100, 1)
            print(f"{j:>2}. {name:<{max_len}} {perc}% ({count})")
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
    parser.add_argument("-k", "--kudos", help="Show statistics for the kudos per hour", action="store_true")
    parser.add_argument(
        "-m",
        "--hordemonth",
        help="Show statistics for the entire horde for the month",
        action="store_true",
    )
    parser.add_argument(
        "-x",
        "--textmonth",
        help="Show statistics for the entire horde for the month for the text models",
        action="store_true",
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
    elif args["textmonth"]:
        period = PERIOD_TEXT_HORDE_MONTH
    elif args["kudos"]:
        period = PERIOD_KUDOS_HOUR

    logs = LogStats(period)
    logs.print_stats()
