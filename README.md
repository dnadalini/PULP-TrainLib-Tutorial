# PULP-TrainLib-Tutorial

This repository contains some basic examples to familiarize with the [PULP-TrainLib](https://github.com/pulp-platform/pulp-trainlib) framework and the topic of On-Device Learning.
This training library is written in C and optimized for RISC-V Parallel Ultra-Low Power Processors, i.e., Microcontrollers (MCU). 
We will use the open-source [PULP platform](https://github.com/pulp-platform/pulp) as the target device, leveraging the available platform simulator (GVSOC) that is included in the [PULP-SDK](https://github.com/pulp-platform/pulp-sdk).

## Getting Started

### Requirements
The tutorial has been tested on a **Ubuntu 20.04 LTS** machine (we used a Windows WSL with Ubuntu 20.04 LTS).

On a fresh machine, updating the package list may be required:
```
sudo apt-get update
```
The following packages need to be installed:
```
sudo apt-get install -y make python-is-python3 build-essential git libftdi-dev libftdi1 doxygen python3-pip libsdl2-dev curl cmake libusb-1.0-0-dev scons gtkwave libsndfile1-dev rsync autoconf automake texinfo libtool pkg-config libsdl2-ttf-dev
```
We also recommend using Anaconda or [Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) to create a conda environment for Python (3.8), e.g.:
```
conda create --name trainlib-tutorial python=3.8
conda activate trainlib-tutorial
```
Required Python packages (for GVSOC):
```
pip install --user argcomplete pyelftools
```
PULP-TrainLib uses Pytorch for generating test vectors and checking the results:
```
python -m pip install argparse
python -m pip install install torch torchvision torchaudio
python -m pip install torchsummary
```

GCC <= 9.4 is required. To check if gcc has the right version:
```
gcc --version
```
Please, refer to the official guide to update gcc if is needed.

### Installation

To install the PULP-SDK and the PULP-TrainLib library, you can run this script:
```
source install_ub20.sh
```
The script will clone the PULP-SDK and the PULP-TrainLib submodules and the RISCV compiler. 

Once the installation is completed, _do not forget to close the installation terminal and open a new one_.


### Running Application Code on PULP GVSOC
**IMPORTANT**: Every time a new terminal is open, run the source script from the top directory of the repository:
```
source setup.sh
```
Optionally, you may also need to re-activate your conda environment that was created during the installation of the requirements:
```
conda activate trainlib-tutorial
```

To check if your installation procedure was successful you can try to run a helloworld test:
```
cd pulp-sdk/tests/hello/
make clean all run
```
where:
- `clean`: remove the build folder
- `all`: compile the code
- `run`: execute the binary on the GVSOC simulator

In the case the execution is successful, the following string will appear in the terminal:
```
Hello from FC
```

In case of any issue, you can refer to the [PULP SDK](https://github.com/pulp-platform/pulp-sdk) or open an issue in this repository.


### Running PULP-TrainLib Examples
You can refer to instructions inside every _Ex_ folder to run the provided examples. 
These examples will be shown and explained during the tutorial at DATE24:  [ET02 On-Device Continual Learning Meets Ultra-Low Power Processing](https://www.date-conference.com/embedded-tutorial/et02).

Despite it is not mandatory, we suggest to install and use VSCode to navigate the tutorial repository and run the code on the terminal, e.g., [VScode on Ubuntu WSL](https://code.visualstudio.com/docs/remote/wsl-tutorial).


## PULP-TrainLib: a General Overview

PULP-TrainLib is the first On-Device Learning optimized for RISC-V Multi-Core MCUs, tailored for the Parallel Ultra-Low Power [(PULP) Platform](https://www.pulp-platform.org/). 

### On-Device Learning

On-Device Learning is a novel paradigm for enabling Deep Neural Network (DNN) Training on extreme-edge devices. 
This paradigm: 
- ensures privacy by not sharing personal data with third-party compute resources and, at the same time, reduces network traffic and potential congestion. 
- reduces the latency for model updates with respect to waiting for a server response

### The PULP Platform

The [PULP Platform](https://www.pulp-platform.org/) is a fully open-source (hardware and software) scalable platform for extreme-edge computing based on RISC-V cores. 

An example of a PULP-based System-on-Chip (SoC) is shown in this figure:

![PULP](img/PULP.png)

In this embodiment, the Cluster features 8 parallel cores (Cores 0 to 7) for the computation, and a Cluster Controller core (Core 8), which acts as the cluster controller.

PULP-TrainLib makes efficient use of the available resources of the PULP-based SoCs, which feature:

- A single Core (Fabric Controller) for the control of the system
- A Cluster of N RISC-V Cores capable of computing parallel tasks
- A hierarchical memory system, featuring a Cluster-reserved fast L1 memory and a system-level L2 memory
- A Cluster DMA to access the L2 memory from the Cluster in few cycles
- Tightly coupled accelerators, as a Mixed-Precision FPU, available for the Cluster 

### Tutorial PULP Platform Specs
The platform used in this tutorial features the following specs:
**FIXME**

### PULP-TrainLib Repository

PULP-TrainLib is available as open source [here](https://github.com/pulp-platform/pulp-trainlib). 
Yo can refer to the [README.md](../pulp-trainlib/README.md) for more details. 

In short, PULP-TrainLib is organized as follows:

```
lib/                            
        include/                
        sources/                

tests/                          
        test_<layer/function>_<possible_options>

tools/                          
        AutoTuner/              
        TrainLib_Deployer/      
```



PULP-TrainLib is written in C code, with specific calls to the [PULP PMSIS libraries](https://github.com/pulp-platform/pulp-sdk/tree/main/rtos/pmsis) for parallel execution.
To include PULP-TrainLib in your project, you have to `#include "pulp_train.h"` in your application code.

### Contributing

The project is open-source. 
In the case you want to contribute, open a pull request on the official repository [https://github.com/pulp-platform/pulp-trainlib](https://github.com/pulp-platform/pulp-trainlib), or contact the [maintainers](https://github.com/pulp-platform/pulp-trainlib/blob/main/README.md#contributors). 
We are willing to collaborate on the project!

### References

> D. Nadalini, M. Rusci, G. Tagliavini, L. Ravaglia, L. Benini, and F. Conti, "PULP-TrainLib: Enabling On-Device Training for RISC-V Multi-Core MCUs through Performance-Driven Autotuning" [SAMOS Pre-Print Version](https://www.samos-conference.com/Resources_Samos_Websites/Proceedings_Repository_SAMOS/2022/Papers/Paper_14.pdf), [Springer Published Version](https://link.springer.com/chapter/10.1007/978-3-031-15074-6_13)

> D. Nadalini, M. Rusci, L. Benini, and F. Conti, "Reduced Precision Floating-Point Optimization for Deep Neural Network On-Device Learning on MicroControllers" [ArXiv Pre-Print](https://arxiv.org/abs/2305.19167)

