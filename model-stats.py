# model-stats.py
# Calculate some basic model usage statistics from the local worker log file.
# Usage: model-stats.py [--today]
import re
import mmap
from tqdm import tqdm
import datetime
import argparse
from bridgeData import models_to_load

# Location of stable horde worker bridge log
LOG_FILE = "logs/bridge.log"

# Today's date in log format
TODAY = datetime.datetime.now().strftime("%Y-%m-%d")
# Today's month in log format
THIS_MONTH = datetime.datetime.now().strftime("%Y-%m-")

# regex to identify model lines
REGEX = re.compile(r'.*([0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]).*\[([a-zA-Z].*)\] ')


class LogStats():

    def __init__(self, today=False, logfile=LOG_FILE):
        self.used_models = {}
        self.unused_models = {}
        self.logfile = logfile
        self.today_only = today

    def get_num_lines(self, file_path):
        fp = open(file_path, "r+")
        buf = mmap.mmap(fp.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines

    def parse_log(self):

        self.used_models = {}
        # Grab any statically loaded models
        self.unused_models = models_to_load[:]
        # Models to exclude
        if 'safety_checker' in self.unused_models:
            self.unused_models.remove('safety_checker')

        with open(LOG_FILE, "rt") as infile:
            for line in tqdm(infile, total=self.get_num_lines(self.logfile), leave=True, unit=" lines", unit_scale=True):
                # Grab the lines we're interested in
                regex = REGEX.match(line)
                if regex:
                    if self.today_only and regex.group(1) != TODAY:
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

    def print_stats(self):
        # Parse our log file if we haven't done that yet
        if not self.used_models:
            self.parse_log()

        # Whats our longest model name?
        max_len = max([len(x) for x in self.used_models])

        scores = sorted(
            ((self.used_models[model], model) for model in self.used_models),
            reverse=True)
        total = 0
        for count, name in scores:
            total += count

        j = 1
        for count, name in scores:
            perc = round((count/total)*100, 1)
            print(f"{j:>2}. {name:<{max_len}} {perc}% ({count})")
            j += 1

        print()
        if self.unused_models:
            print("The following models were not used at all:")
            for m in self.unused_models:
                print(f"  {m}")
        else:
            print("There were no unused models.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Generate local worker model usage statistics')
    parser.add_argument(
        '-t', '--today', help='Statistics for today only', action='store_true')
    args = vars(parser.parse_args())

    logs = LogStats(args['today'])
    logs.print_stats()
