# TrainLib Deployer: code generation for you On-Device Learning project

NOTE: THE WAY TO GENERATE THE NETWORK SHOULD BE CHANGED (CRISTI + ME)

This first exercise aims at showing you how you can generate and edit the code for On-Device Learning on Convolutional Neural Networks with PULP-TrainLib.
Here, we will modify the `USER SECTION` of [TrainLib_Deployer](../pulp-trainlib/tools/TrainLib_Deployer/TrainLib_Deployer.py) to generate the code for a CNN, deployed with FP32 computations.

## Generate a FP32 CNN

To generate this example, first overwrite the corresponding lines of [TrainLib_Deployer](../pulp-trainlib/tools/TrainLib_Deployer/TrainLib_Deployer.py) with the content of [CNN_FP32.txt](CNN_FP32.txt). Then, open a new terminal in this repository root folder (PULP-TrainLib-Tutorial/) and:

```
cd ../pulp-trainlib/tools/TrainLib_Deployer
python TrainLib_Deployer.py
```

This will generate the code in `Ex01-TrainLib_Deployer/CNN_FP32/`. Then:

```
source setup.sh
cd ../../../Ex01-TrainLib_Deployer/CNN_FP32/
make clean get_golden all run
```

Executing the last command will generate the golden model from PyTorch (`make get_golden`) and compile the code for the execution on the PULP simulator (`make clean all run`). On the terminal, you will find both the functional and profiling information.

## Verify & Profile your CNNs

The output of your terminal should show you an output like:

```
TYPICAL OUTPUT OF AN APPLICATION
```

First, ... represents the output of the DNN with respect to PyTorch's one.

Second, ... shows you profiling information about the execution of ODL on PULP GVSoC.



## Navigate the code 

Here, open [CNN_FP32/net.c](CNN_FP32/net.c) at line XXX, and [CNN_FP16/net.c](CNN_FP16/net.c) at line XXX. 