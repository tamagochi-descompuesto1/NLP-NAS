# NLP Text Generation NAS for the Jetson Orin Nano.
This investigation project focuses on using NAS and HWNAS to improve text generation models and adapt them to the hardware of a [Jetson Orin Nano 8 GB DevKit](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit). The main aim of this repository is to guide users to be able to reproduce the experiments made for the investigation project.

## Table of contents. 
- [Requirements.](#requirements)
- [Flashing the Jetson Nano.](#flashing-the-jetson-nano)
- [Setting up the environment.](#setting-up-the-environment)
- [Running the scripts.](#running-the-scripts)


## Requirements.
In order to correctly reproduce this investigation project it is necessary to have the following equipment:
- An 8 GB Jetson Orin Nano Development Kit.
- A microSD card with at least 64 GB of space.
- A USB mouse and keyboard.
- An ethernet cable.
- An HDMI cable.
- A computer display.
- A Display Port to HDMI adapter.
- A computer with Ubuntu 18.04 or 20.04 installed (20.04 is recommended).
- A Jumper wire female to female.
- An USB-A to USB-C 2.0 or higher cable.

## Flashing the Jetson Nano.
The first step is to flash the Jetson Nano with JetPack 5.1.2.

Firstly, it is necessary to download and install [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) in the Ubuntu device. It will also be necessary to format to FAT32 the microSD card completely using a software like SD Card Formatter. After installing and opening the application it will be necessary to boot the Jetson Orin Nano in recovery mode.

To boot the Jetson Orin Nano in recovery mode it is necessary to bridge the pins 9 (GND) and 10 (FC REC) located at one of the sides of the Jetson, below the fan and processor. Then, connect the Jetson Orin Nano to the Ubuntu device using the USB-A to USB-C cable. The USB-C end of the cable will be connected to the USB-C port of the Jetson and the the USB-A end will be connected to the Ubuntu device. Then, insert the formatted microSD card in the Jetson in its corresponding slot. Finally, the Jetson will be booted up by connecting it to the current. Refer more to [this tutorial](https://youtu.be/Ucg5Zqm9ZMk?si=7n513QVXQBR1GK42&t=417) to clear any doubts with the setup of the Jetson.

When the Jetson is booted and connected to the host Ubuntu PC, SDK Manager will automatically detect it. It is necessary to select the Jetson Nano 8GB DevKit in order for the flashing to succeed. When the device is selected the jumper forcing the recovery mode boot needs to be removed. The following step is to select the software to install. Select the JetPack 5.1.2 OS and the DeepStream software for AI implementations. After accepting the terms and conditions of NVIDIA the download and flashing will begin. Before flashing the device SDK manager will ask if the configuration of the OS will be done before the flash or in runtime, choose the runtime option and flash the Jetson. 

During flashing many errors may occur, if anyone occurs refer to the [NVIDIA forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/jetson-orin-nano/632) to receive advice. The main errors occured during this process were:
- **Use of corrupted microSD card**: Make sure the used microSD card is not corrupted and in conditions to be used.
- **Incompatible microSD card**: 128 GB cards tend to present problems during the flashing, a workaround can be done by modifying a script to make it possible to allocate bigger memory sections.
- **Software incompatibilit**: Sometimes the version of Ubuntu may present problems of incompatibility with the version of JetPack. For this project Ubuntu 18.04 was used and the JetPack version installed was 5.1.2.

You can also refer to [this tutorial](https://www.youtube.com/watch?v=Ucg5Zqm9ZMk) to better see how to flash the Jetson Orin Nano. Ultimately, remember that flashing the Jetson is the most difficult task, thus, it should be taken with patience. Additionally, it is recommended to connect the Jetson Orin Nano via TTL to USB cable to the host PC using minicom in order to better see the flashing process and the errors that might occur, refer to [this tutorial](https://www.youtube.com/watch?v=Kwpxhw41W50) to learn how to do this.

The flasing process should take around 15 to 20 minutes. When finished, disconnect the Jetson, connect it to a display via the Display Port to HDMI adapter to a PC monitor. The first boot may take a while since the Jetson will try to boot via HTTP IPV4 and HTTP IPV6, but after 5 minutes it should boot using the flashed microSD card.

Leave the SDK Manager software as it is and continue with the system configuration of the Jetson, select a username, password and all the initial configurations. After this come back to SDK Manager and finish installing all the Jetson and NVIDIA libraries, for this you will need to input your username, password and IP address (which you can find in the terminal executing the command `ifconfig`). Installation takes around 20 minutes, after this the Jetson device is ready to use.

## Setting up the environment.
If the Jetson correctly booted up, it will ask for the configurations. Choose a user, password and necessary configurations to initialize the OS. Remember the Jetson Orin Nano does not have a WiFi module, thus it is necessary to connect it to a router or modem via ethernet cable. 

After configuring the OS, the environment needs to be prepared. Firstly, it is necessary to open a Terminal and execute the following commands:
```shell
sudo apt-get update
sudo apt update
sudo apt install python3-pip
sudo pip3 install -U jetson-stats
```

This will update the apt command, install pip3 to finally install jetson-stats. This is a library that will help to visualize many of the statistics related to the Jetson hardware; from CPU usage, GPU consumption, energy, etc. It is also installed as a Python library, thus, it can be used in future scripts to dump this metrics for further analysis. To run it in the command line it is only necessary to execute the `jtop` command to launch the interface where it is possible to watch statistics related to the Jetson. 

Now, if revising the scripts and modifying them is needed, installing a code editor is recommended. We suggest installing Visual Studio Code, to download it open a terminal and execute the following commands:
```shell
wget -N -O vscode-linux-deb.arm64.deb https://update.code.visualstudio.com/latest/linux-deb-arm64/stable
sudo apt install ./vscode-linux-deb.arm64.deb
code .
```

This will download the latest .deb file of VS Code and install it in the Jetson. The `code .` command will open VS Code in the current folder, this in order to verify it installed correctly. 

Finally, it is recommended to follow good practices regarding Python library installations. This is why we suggest to use virtual environments in order to maintain the Jetson as clean as possible. For this project we used Archiconda, in order to install it open a terminal and execute the following commands: 
```shell
wget --quiet -O archiconda.sh https://github.com/Archiconda/build-tools/releases/download/0.2.3/Archiconda3-0.2.3-Linux-aarch64.sh &&     sh archiconda.sh -b -p $HOME/archiconda3 &&     rm archiconda.sh
bash Archiconda3-0.2.3-Linux-aarch64.sh 
```

This two commands will download and install the Archiconda software. Agree to the terms and services of Archiconda and choose the default options. If you get an error prompt indicating that `Archiconda3-0.2.3-Linux-aarch64.sh` does not exists then execute `bash archiconda.sh` to initialize configuration. 

Sometimes Archiconda is already installed by just executing the `wget` command. In this case just execute the commands:
 ```shell
 sudo ln -s /home/[username]/archiconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh
 echo ". /home/[username]/archiconda3/etc/profile.d/conda.sh" >> ~/.bashrc
 ```

To add conda to the path. After this you should be able to create anaconda environments without any problem.

Then, to configure it and create and environment execute:
```shell
conda init
conda create -n nlp-nas
```

This will create a virtual environment called `nlp-nas` to enter the environment run the command `conda activate nlp-nas`. You can exit the environment by executing the `conda deactivate`. This will take you to the base environment (depicted by the `(base)` indicator at the start of the line in the terminal) to exit this environment execute `conda deactivate` again. Make sure to always execute the Python scripts related to this project within a virtual environment.

With the environment active execute the following commands: 
```shell
sudo -H pip3 install --no-cache-dir jetson-stats
sudo apt-get install libopenblas-base libopenmpi-dev libomp-dev
```

This will install and configure some necessary libraries for correct script execution. After this it is necessary to configure and install a custom PyTorch version for this device. For this we need to upgrade the Python version from 3.6 to 3.8 using the folloewing commands:
```shell
sudo apt update -y
sudo apt install python3.8

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
sudo update-alternatives --config python3
```
When executing the last command an interface will prompt you to choose a default file to execute for the `python3` command. You should type 2 for Python 3.8 to be the default version to use.

Now we will define a torch version to install, this needs to be donwloaded from the official NVIDIA website and is defined in the shell by the following command:
```shell
export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
```

You are able to change the link by copying from this [NVIDIA forum post](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048). You can also change the following arguments of the URL:
- JetPack version: Current version of your JetPack installation, i.e. if you installed JetPack 5.1.1 then this argument would be `v511`.
- Torch wheel: This is the torch wheel version you can choose specific to your Python and JetPack Version. Available wheels are listed in the previously mentioned [NVIDIA forum post](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) or the ones found in this [compatibility matrix](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html#pytorch-jetson-rel).

It is recommended to use the Python, PyTorch and JetPack versions installed in this tutorial for correct replication. After defining the PyTorch installation you execute: 
```shell
python3 -m pip install --upgrade pip; python3 -m pip install numpy==’1.24.4’; python3 -m pip install --no-cache $TORCH_INSTALL
```

To test the correct PyTorch installation execute:
```shell
python3 -c 'import torch;print(torch.cuda.is_available())'
```
This should return `True` indicating that GPU is available for use. For further PyTorch installation configurations you can refer to the [official NVIDIA tutorial](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html). 

TODO CORRECTLY CONFIGURE CUDA, MAY BE DUE TO INCORRECT FLASHING.

## Running the scripts.
To run the scripts related to this project, first clone this repository in the Jetson via `git clone` or by downloading the scripts locally to the device. After cloning the repository activate the conda environment and execute the command `pip3 install -r requirements.txt` to install all the necessary libraries for the scripts to work adequately.

You can donwload and save locally all the models for this project from [Google Drive TODO GDRIVE LINK](GDRIVE). Store the checkpoints in a `models/` folder within the `NLP_NAS/` directory, make sure your project has the structure: 
```
- NLP-NAS
    - .gitignore
    - LICENSE
    - README.md
    - NLP_NAS
        - jtop_stats/
        - results/
        - stat_dumps/
        - models/
            - main model name 1/
                - variant 1 of the main model
                - variant 2 of the main model
                - ...
            - main model name 2/
                - variant 1 of the main model
                - variant 2 of the main model
                - ...
            - ...
        - jtop_test.py
        - model_loader.py
        - test_distilgpt2.py
        - models.txt
        - requirements.txt
```
 Thus, the models folder is composed of $n$ folders which correspond to each individual model selected and this folders store different variants of the model (trained with different datasets, epochs, etc.). Then you can execute any `python3 test_[model_name].py` command to test the selected model and analyze the resources each model consumes. 

Additionally, you can download untrained models located in the `models.txt` file into a previously cretated `models/` folder. You can modify this file to add as many Hugging Face models as you want but be aware that the scripts and hardware may not be adapted to them and might not work as expected. This option is also not recommended since hardware results and benchmark scores will differ from the expected results.

When a `test_[model_name].py` file is executed all variants of the model will be loaded and tested over their corresponding datasets. After completing the test a file will be created within the `/stat_dumps/[model_name]/[model_variant]` with the name `[model_variant]-test-[year]-[month]-[day]_[hour]-[minute]-[second].txt` which will store all the hardware stats related to the execution of that model at that given time. For example, if y execute the `distilgpt2` model with `python3 test_distilgpt2.py` at 05/06/2024 16:44:23, then the resulting file for the a variant that was trained for 3 epochs will be `distilgpt2-test-2024-05-06_16-44-23.txt`.

This file contains various hardware statistics. It divides in three parts: the hardware stats before the execution of the model, the hardware stats after the execution of the model and the difference (substraction) of this two metrics. They can give a good overview about how each model works and how much resources they consume.

Also, within the results folder many plots related to the model's performance will be created. Each plot will be stored in '/results/[model_name]/[model_variant]' for further analysis. These graphics describe the performance of each model during testing, including precission, recall and F1 scores.