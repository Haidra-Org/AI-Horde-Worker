import argparse
import glob
import os
import subprocess
import sys

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--fix",
    action="store_true",
    required=False,
    help="Fix issues which can be fixed automatically",
)
arg_parser.add_argument(
    "--debug",
    action="store_true",
    required=False,
    help="Show some extra information for debugging purposes",
)
args = arg_parser.parse_args()

# Set the working directory to where this script is located
thisFilePath = os.path.abspath(__file__)
workingDirectory = os.path.dirname(thisFilePath)

if args.debug:
    print(f"working out of {workingDirectory}")
    print(f"style.py located at {thisFilePath}")
os.chdir(workingDirectory)

src = [
    "worker",
]

ignore_src = [
    "bridgeData_template.py",
    "bridgeData.py",
]

root_folder_src = glob.glob("*.py")

src.extend(root_folder_src)

src = [item for item in src if item not in ignore_src]


black_args = [
    "black",
    "--line-length=119",
]
ruff_args = [
    "ruff",
]

if args.fix:
    print("fix requested")
    ruff_args.extend(("check", "--fix"))
else:
    print("fix not requested")

    black_args.extend(("--check", "--diff"))
    ruff_args.append("--diff")

lint_processes = [
    ruff_args,
    black_args,
]

if args.debug:
    for process in lint_processes:
        VERSION_COMMAND = process[0] + " --version"
        subprocess.run(VERSION_COMMAND, shell=True, check=True)
        print()

for process_args in lint_processes:
    process_args.extend(src)

    COMMAND = "python -s -m "
    COMMAND += " ".join(process_args)
    print(f"\nRunning {COMMAND}")
    try:
        subprocess.run(COMMAND, shell=True, check=True)
    except subprocess.CalledProcessError:
        sys.exit(1)
