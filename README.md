This repository allows you to set up a AI Horde Worker to generate or alchemize images for others

# AI Horde Worker

## Important Note:

- As of January 2024, the official worker is now [horde-worker-reGen](https://github.com/Haidra-Org/horde-worker-reGen).
- You should use `reGen` if you are a new worker and are looking to do *image generation*.
- If you are looking to do *alchemy* (post-processing, interrogation, captioning, etc), you should continue to use `AI-Horde-Worker`.


# Legacy information: 
This repo contains the original (now outdated - see above) reference implementation for a [AI Horde](https://aihorde.net) Worker. This will turn your graphics card(s) into a worker for the AI Horde and you will receive in turn kudos which will give you priority for your own generations.

Alternatively you can become an Alchemist worker which is much more lightweight and can even run on CPU (i.e. without a GPU).

Please note that **AMD card are not currently supported**, but may be in the future. (Note: [horde-worker-reGen](https://github.com/Haidra-Org/horde-worker-reGen) has prelimanry support for AMD and if you are an AMD card and would like to help us improve support by testing it, let us know [in our discord](https://discord.gg/kwst4K7wbv))
 
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
1. In the Source location put `https://github.com/Haidra-Org/AI-Horde-Worker.git`
1. In the target directory, browse to any folder you want to put the horde worker folder.
1. Press `Clone`
1. In the new window that opens up, on the top menu, go to `Repository > Git Bash`. A new terminal window will open.
1. continue with the [Running](#running) instructions

### Without git

Use these instructions if you do not have git for windows and do not want to install it. These instructions make updating the worker a bit more difficult down the line.

1. Download [the zipped version](https://github.com/Haidra-Org/AI-Horde-Worker/archive/refs/heads/main.zip)
1. Extract it to any folder of your choice
1. continue with the [Running](#running) instructions

## Linux

This assumes you have git installed

Open a bash terminal and run these commands (just copy-paste them all together)

```bash
git clone https://github.com/Haidra-Org/AI-Horde-Worker.git
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

To load it, simply run `bridge-webui`. It will then show you a URL you can open with your browser. Open it and it will allow you to tweak all horde options. Once you press `Save Configuration` it will create a `bridgeData.yaml` file with all the options you set.

Fill in at least:
   * Your worker name (has to be unique horde-wide)
   * Your AI Horde API key

You can use this UI and update your bridge settings even while your worker is running. Your worker should then pick up the new settings within 60 seconds.

You can also edit this file using a text editor. We also provide a `bridgeData_template.yaml` with comments on each option which you can copy into a new `bridgeData.yaml` file. This info should soon be onboarded onto the webui as well.

## Startup

Start your worker, depending on which type your want.

* If you want to generate Stable Diffusion images for others, run `horde-bridge`.


    **Warning:** This requires a powerful GPU. You will need a GPU with at least 6G VRAM
    
* If you want to interrogate images for other, run `horde-alchemist_bridge`. This worker is very lightweight and you can even run it with just CPU (but you'll have to adjust which forms you serve)


    **Warning:** Currently the Alchemist worker will download images directly from the internet, as if you're visiting a webpage. If this is a concern to you, do not run this worker type. We are working on setting up a proxy to avoid that.

Remember that worker names have to be different between Stable Diffusion worker and Alchemist worker. If you want to start a different type of worker in the same install directory, ensure a new name by using the `--name` command line argument.


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


# Docker

To run the Docker container, specify the required environment variables:

- HORDE_API_KEY: The API key to use for authentication.

ghcr.io/Haidra-Org/ai-horde-worker:<insert release tag here>

Optional environment variables:

- HORDE_URL: The URL of the Horde server to connect to. Defaults to 'https://stablehorde.net'.
- HORDE_WORKER_NAME: The name of the worker. If not set, a random worker name will be generated.
- HORDE_WORKER_PREFIX: Used in random worker name generation, defaults to DockerWorker ${HORDE_WORKER_PREFIX}#0123097164
- HORDE_PRIORITY_USERNAMES: A comma-separated list of usernames that should be given priority in the queue.
- HORDE_MAX_THREADS: The maximum number of threads to use for rendering. Defaults to '1'.
- HORDE_QUEUE_SIZE: The maximum number of jobs to queue. Defaults to '0', meaning no limit.
- HORDE_REQUIRE_UPFRONT_KUDOS: Whether to require users to have enough kudos before they can submit jobs. Defaults to 'false'.
- HORDE_MAX_POWER: The maximum power level to use for rendering. Defaults to '8'.
- HORDE_NSFW: Whether to allow NSFW content. Defaults to 'true'.
- HORDE_CENSOR_NSFW: Whether to censor NSFW content. Defaults to 'false'.
- HORDE_BLACKLIST: A comma-separated list of tags to blacklist.
- HORDE_CENSORLIST: A comma-separated list of tags to censor.
- HORDE_ALLOW_IMG2IMG: Whether to allow image-to-image translation models. Defaults to 'true'.
- HORDE_ALLOW_PAINTING: Whether to allow painting models. Defaults to 'true'.
- HORDE_ALLOW_UNSAFE_IP: Whether to allow unsafe IP addresses. Defaults to 'true'.
- HORDE_ALLOW_POST_PROCESSING: Whether to allow post-processing. Defaults to 'true'.
- HORDE_ALLOW_CONTROLNET: Whether to allow ControlNet. Defaults to 'false'.
- HORDE_DYNAMIC_MODELS: Whether to use dynamic models. Defaults to 'true'.
- HORDE_NUMBER_OF_DYNAMIC_MODELS: The number of dynamic models to use. Defaults to '3'.
- HORDE_MAX_MODELS_TO_DOWNLOAD: The maximum number of models to download. Defaults to '10'.
- HORDE_STATS_OUTPUT_FREQUENCY: The frequency (in seconds) to output stats. Defaults to '30'.
- HORDE_NATAILI_CACHE_HOME: The location of the cache directory. Defaults to '/cache'.
- HORDE_LOW_VRAM_MODE: Whether to use low VRAM mode. Defaults to 'true'.
- HORDE_ENABLE_MODEL_CACHE: Whether to enable model caching. Defaults to 'false'.
- HORDE_ALWAYS_DOWNLOAD: Whether to always download models. Defaults to 'false'.
- HORDE_RAY_TEMP_DIR: The location of the Ray temporary directory. Defaults to '/cache/ray'.
- HORDE_DISABLE_VOODOO: Whether to disable Voodoo. Defaults to 'false'.
- HORDE_DISABLE_TERMINAL_UI: Whether to disable the terminal UI. Defaults to 'false'.
- HORDE_MODELS_TO_LOAD: A comma-separated list of models to load. Defaults to ['stable_diffusion_2.1', 'stable_diffusion'].
- HORDE_MODELS_TO_SKIP: A comma-separated list of models to skip. Defaults to ['stable_diffusion_inpainting'].
- HORDE_FORMS: A comma-separated list of forms to use. Defaults to ['caption', 'nsfw'].
