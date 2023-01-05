This repository allows you to set up a Stable Horde Worker to generate or interrogate images for others

# Stable Horde Worker

This repo contains the latest implementation for the [Stable Horde](https://stablehorde.net) Worker. This will turn your graphics card(s) into a worker for the Stable Horde and you will receive in turn kudos which will give you priority for your own generations.

Alternatively you can become an interrogation worker which is much more lightweight and can even run on CPU (i.e. without a GPU)

To run the bridge, simply follow the instructions for your own OS

# Installing

If you haven't already, go to [stable horde and register an account](https://stablehorde.net/register), then store your API key somewhere secure. You will need it later in these instructions. 

This will allow your worker to gather kudos for your account.

## Windows

### Using git (recommended)

Use these instructions if you have installed [git for windows](https://gitforwindows.org/).

This option is recommended as it will make keeping your repository up to date much easier.

1. Use your start menu to open `git GUI`
1. Select "Clone Existing Repository". 
1. In the Source location put `https://github.com/db0/AI-Horde-Worker.git`
1. In the target directory, browse to any folder you want to put the horde worker folder.
1. Press `Clone`
1. In the new window that opens up, on the top menu, go to `Repository > Git Bash`. A new terminal window will open.
1. continue with the [Running](#running) instructions

### Without git

Use these instructions if you do not have git for windows and do not want to install it. These instructions make updating the worker a bit more difficult down the line.

1. Download [the zipped version](https://github.com/db0/AI-Horde-Worker/archive/refs/heads/main.zip)
1. Extract it to any folder of your choice
1. continue with the [Running](#running) instructions

## Linux

This assumes you have git installed

Open a bash terminal and run these commands (just copy-paste them all together)

```bash
git clone https://github.com/db0/AI-Horde-Worker.git
cd AI-Horde-Worker
```

This will take care of the setup for this environment and then automatically start the bridge.

For more information, see: https://github.com/db0/AI-Horde/blob/main/README_StableHorde.md#joining-the-horde


# Running

The below instructions refer to running scripts `horde-bridge` or `update-runtime`. Depending on your OS, append `.cmd` for windows, or `.sh` for linux.

You can double click the provided script files below from a file explorer or run it from a terminal like `bash`, `git bash` or `cmd` depending on your OS. 
The latter option will allow you to see errors in case of a crash, so it's recommended.

## Update runtime

If you have just installed or updated your worker code run the `update-runtime` script. This will ensure the dependencies needed for your worker to run are up to date

This script can take 10-15 minutes to complete.

## Startup

Start your worker, depending on which type your want. 

* If you want to generate Stable Diffusion images for others, run `horde-bridge`.

    **Note:** In order for your worker to work, it needs to download a stable diffusion model. To do that, you will need to register a free account at https://huggingface.co. You will need to put your username and password for it when prompted. You will also need to accept the license of the model you're about to download, so after logging in to huggingface, visit https://huggingface.co/runwayml/stable-diffusion-v1-5 and accept the license presented within.

    **Warning:** This requires a powerful GPU. You will need a GPU with at least 6G VRAM. If you do not have at least 32G or RAM, append `--disable-voodoo` to your startup command above!
* If you want to interrogate images for other, run `horde-interrogation_bridge`. This worker is very lightweight and you can even run it with just CPU (but you'll have to adjust which forms you serve)

    **Warning:** This currently the interrogation worker will download images directly from the internet, as if you're visiting a webpage. If this is a concern to you, do not run this worker type. We are working on setting up a proxy to avoid that.

Remember that worker names have to be different between Stable Diffusion worker and Interrogation worker. If you want to start a different type of worker in the same install directory, ensure a new name by using the `--name` command line argument.

## bridgeData.py

The very first time you run the bridge script, it will take you through a small interactive setup. Simply follow the instructions as you see on the terminal and type your answer to the prompts. 

During your very first run will instruct you to create a `bridgeData.py` file. If you did, it will abort the run and allow you to edit the properties of the file. If this file wasn't created automatically, you can create it now by copying `BridgeData_template.py` to `BridgeData.py`.

Open `bridgeData.py` with a text editor such as notepad or nano and edit its properties according to the comments inside. 

Fill in at least:
   * Your worker name (has to be unique horde-wide)
   * Your stable horde API key

Once done, simply run the commands from [Startup](#startup) above.

## Running with multiple GPUs

To use multiple GPUs as with NVLINK workers, each has to start their own webui instance. For linux, you just need to limit the run to a specific card:

```
CUDA_VISIBLE_DEVICES=0 ./horde-bridge.sh -n "My awesome instance #1"
CUDA_VISIBLE_DEVICES=1 ./horde-bridge.sh -n "My awesome instance #2"
```
etc

# Updating

The stable horde workers are under constant improvement. In case there is more recent code to use follow these steps to update

First step: Shut down your worker by putting it into maintenance, and then pressing ctrl+c

## git

Use this approach if you cloned the original repository using `git clone`

1. Open a or `bash`, `git bash`, `cmd`, or `powershell` terminal depending on your OS
1. Navigate to the folder you have the AI Horde Worker repository installed if you're not already there.
1. run `git pull`
1. continue with [Running](#running) instructions above

Afterwards run the `horde-bridge` script for your OS as usual.

## zip

Use this approach if you downloaded the git repository as a zip file and extracted it somewhere.


1. delete the `worker/` directory from your folder
1. Download the [repository from github as a zip file](https://github.com/db0/AI-Horde-Worker/archive/refs/heads/main.zip)
1. Extract its contents into the same the folder you have the AI Horde Worker repository installed, overwriting any existing files
1. continue with [Running](#running) instructions above


# Stopping

* First put your worker into maintenance to avoid aborting any ongoing operations. Wait until you see no more jobs running.
* In the terminal in which it's running, simply press `Ctrl+C` together.

# Model Usage
Many models in this project use the CreativeML OpenRAIL License.  [Please read the full license here.](https://huggingface.co/spaces/CompVis/stable-diffusion-license)