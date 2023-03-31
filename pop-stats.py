# pop-stats.py
# Calculate node pop stats from the local worker log file.
# Usage: pop-stats.py [-h] [--today] [--yesterday]
import argparse
import datetime
import glob
import mmap
import re

from tqdm import tqdm

# Location of stable horde worker bridge log
LOG_FILE = "logs/bridge*.log"

# TIME PERIODS
PERIOD_ALL = 0
PERIOD_TODAY = 1
PERIOD_YESTERDAY = 2
PERIOD_HOUR = 3

# regex to identify model lines
REGEX = re.compile(r".*(\d\d\d\d-\d\d-\d\d \d\d:\d\d).* Job pop took (\d+\.\d+).*node: (.*)\)")


class LogStats:
    def __init__(self, period=PERIOD_ALL, logfile=LOG_FILE):
        self.logfile = logfile
        self.period = period
        self.data = {}

    def get_date(self):
        # Dates in log format for filtering
        if self.period == PERIOD_TODAY:
            adate = datetime.datetime.now()
            adate = adate.strftime("%Y-%m-%d")
        elif self.period == PERIOD_YESTERDAY:
            adate = datetime.datetime.now() - datetime.timedelta(1)
            adate = adate.strftime("%Y-%m-%d")
        elif self.period == PERIOD_HOUR:
            adate = datetime.datetime.now()  # - datetime.timedelta(hours=1)
            adate = adate.strftime("%Y-%m-%d %H:")
        else:
            adate = None
        return adate

    def get_num_lines(self, file_path):
        with open(file_path, "r+") as fp:
            buf = mmap.mmap(fp.fileno(), 0)
            lines = 0
            while buf.readline():
                lines += 1
            return lines

    def parse_log(self):
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
                        if self.period and self.get_date() not in regex.group(1):
                            continue

                        # Extract kudis and time
                        kudos = regex.group(2)
                        node = regex.group(3).split(":")[0]
                        if node in self.data:
                            self.data[node] = [self.data[node][0] + float(kudos), self.data[node][1] + 1]
                        else:
                            self.data[node] = [float(kudos), 1]

                    progress.update()

    def print_stats(self):
        # Parse our log file if we haven't done that yet
        self.parse_log()

        total = 0
        for k, v in self.data.items():
            total += v[1]

        print(f"Average node pop times (out of {total} pops in total)")
        for k, v in self.data.items():
            print(f"{k.split(':')[0]:15} {round(v[0]/v[1], 2)} secs {v[1]:-8} jobs from this node")
            total += v[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate local worker job pop statistics")
    parser.add_argument("-t", "--today", help="Statistics for today only", action="store_true")
    parser.add_argument("-y", "--yesterday", help="Statistics for yesterday only", action="store_true")
    parser.add_argument("-1", "--hour", help="Statistics for last hour only", action="store_true")
    args = vars(parser.parse_args())

    period = PERIOD_ALL
    if args["today"]:
        period = PERIOD_TODAY
    elif args["yesterday"]:
        period = PERIOD_YESTERDAY
    elif args["hour"]:
        period = PERIOD_HOUR

    logs = LogStats(period)
    logs.print_stats()
