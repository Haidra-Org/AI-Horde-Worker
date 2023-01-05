This repository allows you to set up a Stable Horde Worker to generate or interrogate images for others

# Stable Horde Worker

This repo contains the latest implementation for the [Stable Horde](https://stablehorde.net) Worker. This will turn your graphics card(s) into a worker for the Stable Horde and you will receive in turn kudos which will give you priority for your own generations.

Alternatively you can become an interrogation worker which is much more lightweight and can even run on CPU (i.e. without a GPU)

To run the bridge, simply follow the instructions for your own OS

## Initial Setup

If you haven't already, go to [stable horde and register an account](https://stablehorde.net/register), then store your API key somewhere secure. You will need it later in these instructions. 

This will allow your worker to gather kudos for your account.

### Windows

#### Using git (recommended)

Use these instructions if you have installed [git for windows](https://gitforwindows.org/).

This option is recommended as it will make keeping your repository up to date much easier.

1. Use your start menu to open `git GUI`
1. Select "Clone Existing Repository". 
1. In the Source location put `https://github.com/db0/AI-Horde-Worker.git`
1. In the target directory, browse to any folder you want to put the horde worker folder.
1. Press `Clone`
1. In the new window that opens up, on the top menu, go to `Repository > Git Bash`. A new terminal window will open.
1. continue with the common windows instructions

#### Without git

Use these instructions if you do not have git for windows and do not want to install it. These instructions make updating the worker a bit more difficult down the line.

1. Download [the zipped version](https://github.com/db0/AI-Horde-Worker/archive/refs/heads/main.zip)
1. Extract it to any folder of your choice
1. continue with the common windows instructions

#### Common windows instructions

1. Follow one of the two options below
   * (Recommended) open a `git bash` terminal, or a `cmd` terminal. Type `update-runtime.cmd` and pres enter. This approach will allow you to better see any errors
   * Run `update-runtime.cmd` by double-clicking on it from your explorer.
1. Wait until `update-runtime.cmd` is finished. This will take approximately 10-15 mins.
1. Run `horde-bridge.cmd` by double-clicking on it from your explorer of from a `git bash` or `cmd` terminal.

### Linux

This assumes you have git installed

Open a bash terminal and run these commands (just copy-paste them all together)

```bash
git clone https://github.com/db0/AI-Horde-Worker.git
cd nataili
update-runtime.sh
horde-bridge.sh
```

This will take care of the setup for this environment and then automatically start the bridge.

For more information, see: https://github.com/db0/AI-Horde/blob/main/README_StableHorde.md#joining-the-horde


## Updates

The stable horde workers are under constant improvement. In case there is more recent code to use follow these steps to update

First step: Shut down your worker by putting it into maintenance, and then pressing ctrl+c

### git

Use this approach if you cloned the original repository using `git clone`

1. Open a or `bash`, `git bash`, `cmd`, or `powershell` terminal depending on your OS
1. Navigate to the folder you have the nataili repository installed if you're not already there.
1. run `git pull`
1. continue with common update instructions

Afterwards run the `horde-bridge` script for your OS as usual.

### zip

Use this approach if you downloaded the git repository as a zip file and extracted it somewhere.


1. delete the `worker/` directory from your folder
1. Download the [repository from github as a zip file](https://github.com/db0/AI-Horde-Worker/archive/refs/heads/main.zip)
1. Extract its contents into the same the folder you have the AI Horde Worker repository installed, overwriting any existing files
1. continue with common update instructions

### common update instructions

#### Windows

1. run `update-runtime.cmd`

#### Linux

1. run `update-runtime.sh`

You can now start your worker again

## Model Usage
Many models in this project use the CreativeML OpenRAIL License.  [Please read the full license here.](https://huggingface.co/spaces/CompVis/stable-diffusion-license)