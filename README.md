This repository allows you to set up a AI Horde Worker to generate or alchemize images for others

# AI Horde Worker

This repo contains the latest implementation for the [AI Horde](https://aihorde.net) Worker. This will turn your graphics card(s) into a worker for the AI Horde and you will receive in turn kudos which will give you priority for your own generations.

Alternatively you can become an interrogation worker which is much more lightweight and can even run on CPU (i.e. without a GPU)

To run the bridge, simply follow the instructions for your own OS

# Installing

If you haven't already, go to [AI Horde and register an account](https://aihorde.net/register), then store your API key somewhere secure. You will need it later in these instructions.

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

Continue with the [Running](#running) instructions

# Running

The below instructions refer to running scripts `horde-bridge` or `update-runtime`. Depending on your OS, append `.cmd` for windows, or `.sh` for linux.

You can double click the provided script files below from a file explorer or run it from a terminal like `bash`, `git bash` or `cmd` depending on your OS.
The latter option will allow you to see errors in case of a crash, so it's recommended.

## Update runtime

If you have just installed or updated your worker code run the `update-runtime` script. This will ensure the dependencies needed for your worker to run are up to date

This script can take 10-15 minutes to complete.

## Configure

In order to connect to the horde with your username and a good worker name, you need to configure your horde bridge. To this end, we've developed an easy WebUI you can use

To load it, simply run `bridge-webui`. It will then show you a URL you can open with your browser. Open it and it will allow you to tweak all horde options. Once you press `Update Bridge` it will create a `bridgeData.yaml` file with all the options you set.

Fill in at least:
   * Your worker name (has to be unique horde-wide)
   * Your AI Horde API key

You can use this UI and update your bridge settings even while your worker is running. Your worker should then pick up the new settings within 60 seconds.

You can also edit this file using a text editor. We also provide a `bridgeData_template.yaml` with comments on each option which you can copy into a new `bridgeData.yaml` file. This info should soon be onboarded onto the webui as well.

## Startup

Start your worker, depending on which type your want.

* If you want to generate Stable Diffusion images for others, run `horde-bridge`.

    **Warning:** This requires a powerful GPU. You will need a GPU with at least 6G VRAM. If you do not have at least 20G or RAM, append `--disable_voodoo` to your startup command above! If you do not have at least 6Gb of swap, also `--disable_voodoo`!
* If you want to interrogate images for other, run `horde-interrogation_bridge`. This worker is very lightweight and you can even run it with just CPU (but you'll have to adjust which forms you serve)

    **Warning:** This currently the interrogation worker will download images directly from the internet, as if you're visiting a webpage. If this is a concern to you, do not run this worker type. We are working on setting up a proxy to avoid that.

Remember that worker names have to be different between Stable Diffusion worker and Interrogation worker. If you want to start a different type of worker in the same install directory, ensure a new name by using the `--name` command line argument.


## Running with multiple GPUs

To use multiple GPUs as with NVLINK workers, each has to start their own webui instance. For linux, you just need to limit the run to a specific card:

```
CUDA_VISIBLE_DEVICES=0 ./horde-bridge.sh -n "My awesome instance #1"
CUDA_VISIBLE_DEVICES=1 ./horde-bridge.sh -n "My awesome instance #2"
```
etc

# Updating

The AI Horde workers are under constant improvement. In case there is more recent code to use follow these steps to update

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
