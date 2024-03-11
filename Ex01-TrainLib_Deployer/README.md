# End-to-End Training using PULP-TrainLib

This example shows an end-to-end training task with PULP-TrainLib.

Learning objectives: 
- Compile and Run an ODL example on the GVSOC platform
- Performance metrics
- ODL primitives and memory requirements
- Parallelization on a multi-core cluster Cluster
- Automate the ODL Code Generation with TrainLib_Deployer

## Run the example
The example code implements the training of a simple three-layer DNN, which is shown in the figure below:

![DNN](../img/DNN.png)

To run the code do not forget to source the setup script from the top directory if a new shell is open:
```
cd <top_tutorial_dir>
source setup.sh
```
and activate the conda environment that was used during the installation process. 

To run the example:
```
cd Ex01-TrainLib_Deployer/CNN_FP32/
make clean get_golden all run
```
This command generates the reference golden model from PyTorch (`make get_golden`) and compiles the code for execution on the PULP GVSoC simulator (`make clean all run`). 
As a result, training results and performance counters are printed on the terminal. 
More in detail: 

```
make clean         : deletes the build folder and old binaries
make get_golden    : generates the Golden Model's reference files (using the Pytorch module)
make all           : compiles the C code application
make run           : executes the code on the PULP GVSoC simulator
```

The typical output of the generated example will be like: 

```
Hello sir.
Configuring cluster..

Launching training procedure...
Initializing network..
Testing DNN initialization forward..
Layer 2 output:        _
-0.000023               |
0.000048                |
-0.000030               |
0.000024                \
....                     \  DNN Output before ODL
0.000009                 /  
0.000061                /
-0.000025               |
-0.000021               |
-0.000006              _|

                                            --- STATISTICS FROM CLUSTER CORE 0  ---
[0] elapsed clock cycles = 750262           <= Cycles to execute the program (LATENCY)
[0] number of instructions = 578854         <= Number of instructions
[0] TCDM contentions = 0                    <= Memory contentions in L1
[0] load stalls = 164627                    <= Stalls while loading data from L1
[0] icache miss (clk cycles count) = 1947   <= Instruction cache misses
Checking updated output..

Layer 2 output:        _
-0.000023               |
0.000048                |
-0.000030               |
0.000024                \
...                      \  DNN Output AFTER ODL
0.000009                 / 
0.000061                /
-0.000025               |
-0.000021               |
-0.000006              _|
Exiting DNN Training.
```
Note that this code is now running on a single core of the PULP Cluster (`NUM_CORES=1` in the Makefile).


## On-Device Learning Code: net.c
The [net.c](CNN_FP32/net.c) source file includes the calls to the processing functions, i.e., forward and backward passes of every layer, and the memory allocations. 
The code snippet below shows the main forward and backward functions:

```C
void forward() {
    // First layer's primitive
    pulp_conv2d_fp32_fw_cl(&l0_args);
    // Second layer's primitive
    pulp_relu_fp32_fw_cl(&l1_args);
    // Last layer's primitive
    pulp_linear_fp32_fw_cl(&l2_args);
}

void backward() {
    // Set up and compute the loss
    loss_args.output = &layer2_out;
    loss_args.target = LABEL;
    loss_args.wr_loss = &loss;
    pulp_MSELoss_backward(&loss_args);
    // Last Layer's weight and input gradient
    pulp_linear_fp32_bw_param_grads_cl(&l2_args);
    pulp_linear_fp32_bw_input_grads_cl(&l2_args);
    // Second Layer's (ReLU) input gradient
    pulp_relu_fp32_bw_cl(&l1_args);
    // First Layer's weight gradient (no need for input gradient)
    pulp_conv2d_fp32_bw_param_grads_cl(&l0_args);
}
```

<!-- 
Configuration options are located in the [Makefile](CNN_FP32/Makefile). 
This code is intended to work as a starting point for your application.
-->

The following _.h files_ stores the program parameters:
- [io-data.h](CNN_FP32/io_data.h): golden model generated using the a Pytorch program  [GM.py](CNN_FP32/utils/GM.py)
- [init-defines.h](CNN_FP32/init-defines.h): include the layer parameters


In PULP-TrainLib, tensors are defined as a `blob` structure, containing pointers to the activation and gradient data, which are statically defined as C arrays. 
For example, in the case of a Fully-Connected Layer (_Layer 2_):

```C
// Statically define data (C arrays)
PI_L1 float l2_in[Tin_C_l2 * Tin_H_l2 * Tin_W_l2];
PI_L1 float l2_in_diff[Tin_C_l2 * Tin_H_l2 * Tin_W_l2];
PI_L1 float l2_ker[Tin_C_l2 * Tout_C_l2 * Tker_H_l2 * Tker_W_l2];
PI_L1 float l2_ker_diff[Tin_C_l2 * Tout_C_l2 * Tker_H_l2 * Tker_W_l2];
PI_L1 float l2_out[Tout_C_l2 * Tout_H_l2 * Tout_W_l2];
PI_L1 float l2_out_diff[Tout_C_l2 * Tout_H_l2 * Tout_W_l2];

// Define the tensors as "blobs"
PI_L1 struct blob layer2_in, layer2_wgt, layer2_out;

// Define the arguments for the Fully-Connected
PI_L1 struct Linear_args l2_args;
```
PI_L1 is an attribute to control the static allocation into the TCDM memory. 
The configuration parameters of every layer are preserved into layer-wise structs (e.g., `PI_L1 struct Linear_args l2_args;`). 
The `void DNN_init()` function initiazes these structs with parameters and  `blob` tensor pointers of the layer. 

```C
void DNN_init() {
    // ...

    // Example of one tensor definition for Layer 2
    layer2_wgt.data = l2_ker;
    layer2_wgt.diff = l2_ker_diff;
    layer2_wgt.dim = Tin_C_l1*Tout_C_l1*Tker_H_l1*Tker_W_l1;
    layer2_wgt.C = Tin_C_l2;
    layer2_wgt.H = Tker_H_l2;
    layer2_wgt.W = Tker_W_l2;    

    // ...

    // Configuration structure for Layer 2
    l2_args.input = &layer2_in;
    l2_args.coeff = &layer2_wgt;
    l2_args.output = &layer2_out;
    l2_args.skip_in_grad = 0;
    l2_args.opt_matmul_type_fw = MATMUL_TYPE_FW_L2;
    l2_args.opt_matmul_type_wg = MATMUL_TYPE_WG_L2;
    l2_args.opt_matmul_type_ig = MATMUL_TYPE_IG_L2;
}
```

From a high-level viewpoint, the training routine is controlled by the void `net_step()` function:

```C
void net_step()
{
  DNN_init();
  forward();

  for (int epoch=0; epoch<EPOCHS; epoch++)
  {
    forward();
    compute_loss();
    backward();
    update_weights();
  }
}
```

The `void backward()` function handles the computation of the loss function's gradient, while the `void update_weights()` method updates the weights after each training iteration.  

<!-- 
This mode can be triggered by setting `SEPARATE_BACKWARD_STEPS = False`. 
In this case, the computation of the input gradient can be selected with the `skip_in_grad` parameter, which can be set to 1 to avoid the computation of the input gradient (e.g., the layer is the first of a DNN).
-->


## ODL Latency Measurements

When running the training example, the terminal returns the latency measurements in terms of `elapsed clock cycles`.
The measurements are obtained using the performance counter utils defined in [stats.h](CNN_FP32/stats.h). 
For example, the code below is instrumented for performance measurement:

<!-- 
The latency of a single online learning step can be broken down into the single components by profiling the single components with PULP's performance counters. 
This can be set up at code generation time by setting `PROFILE_SINGLE_LAYERS = True`. 
The execution profiling of any code on PULP can be performed using the functions defined in 
-->

```C
void net_step()
{
  // Initialize the performance counters
  INIT_STATS();
  PRE_START_STATS();

  DNN_init();
  forward();

  for (int epoch=0; epoch<EPOCHS; epoch++)
  {
    forward();
    compute_loss();

    // Start profiling
    START_STATS();
    backward();
    // Stop profiling
    STOP_STATS();

    update_weights();
  }
}
```
Back to tour example example, the per-layer latency breakdown is:

```
Forward Step:    
    Conv2D:     192390 
    ReLU:       5686 
    Linear:     74805 

Backward Step:   
    Linear:     174306 
    ReLU:       6550
    Conv2D:     198573 
```

The latency of the Linear's backward step (`pulp_linear_fp32_bw_param_grads_cl()` to compute the weight gradients + `pulp_linear_fp32_bw_input_grads_cl()` to compute the input gradients) is 2.5x higher with respect to the forward.
In the case of the first Conv2D layer, only `pulp_conv2d_fp32_bw_input_grads_cl()` is computed.


## 8 Core Parallelization

To parallelize the training code over the 8 available cores of the cluster, you can run:
```
make clean all run NUM_CORES=8
``` 
Thanks to the parallelization we measure a speed-up of 6.97x.
The contentions to access the L1 memory and some parallelization overhead motivates the gap vs. the ideal 8x speed-up.
More in details, we measure:
```
[0] elapsed clock cycles = 750262     <= Cycles with 1 Core
[0] elapsed clock cycles = 107356     <= Cycles with 8 Cores
PARALLEL SPEEDUP: 750262 / 107356 = 6.99 x
```

Per-layer speed-up breakdown:
```
Forward Step:   
    Conv2D:     192390 / 27668    = 6.95 x
    ReLU:       5686 / 917        = 6.20 x
    Linear:     74805 / 11784     = 6.35 x

Backward Step:  
    Linear:     174306 / 23937    = 7.28 x
    ReLU:       6550 / 1038       = 6.31 x
    Conv2D:     198573 / 28923    = 6.87 x 
```

Let's now go insight a layer step. 
The figure below graphically shows the forward and backwards operation of a Fully-Connected Layer. 
In this case, likewise the majority of the layers of a Deep Neural Network, can be reshaped as linear algebra operations between tensor data. 
In the case of a fully-Connected Layer, the Forward, Weight Gradient, and Input Gradient operators can be expressed as matrix-vector multiplications.

![FC_Steps](../img/FC_steps.png)

In particular, the input and output tensors can be represented as vectors of size `Cin` and `Cout`, respectively, while the weights like a matrix of size `Cout * Cin`. 
During each step, the resulting tensor is computed as a vector-matrix or vector-vector operation. 

To understand how PULP-TrainLib implements parallelism, let's consider the case of a Fully-Connected Layer, whose, e.g., Forward Step can be expressed as Matrix Multiplication:

![FC_Forward](../img/FC_forward.png)

The _N_ cores work in parallel to compute the output vector, evenly distributing the workload. 
More in detail, every core computes _1/N_ of the output vector.
In the PULP platform, the workload parallelization is handled by the function `pi_cl_team_fork(NUM_CORES, parallel_function, &args)`, which forks the computation of a `parallel_function` over the available cores. 
In the considered case (see [pulp_linear_fp32.c](../pulp-trainlib/lib/sources/pulp_linear_fp32.c), `pulp_linear_fp32_fw_cl()`), parallelization is cast as:

```C
// Define structure to wrap Matrix Multiplication (MM)
struct matMul_args matMul_args;
// Fill fields with operands
matMul_args.A = coeffData;              // First matrix
matMul_args.B = inputData;              // Second matrix
matMul_args.C = outData;                // Output matrix
matMul_args.N = FC_args->output->dim;   // Rows of the first matrix
matMul_args.K = FC_args->input->dim;    // Columns of the first / rows of the second
matMul_args.M = 1;                      // Rows of the second matrix

// Parallelize the MM over NUM_CORES (defined in Makefile as 8)
pi_cl_team_fork(NUM_CORES, mm, &matMul_args);
```

Internally, the Matrix Multiplication kernel (see [pulp_matmul_fp32.c](../pulp-trainlib/lib/sources/pulp_matmul_fp32.c), `mm()`) automatically calculates the portion of the input data to process (for loop iteration from `start` to `stop`):

```C
void mm(void * matMul_args) 
{
  // Set up the arguments
  struct matMul_args* args = (struct matMul_args *)matMul_args;
  float * __restrict__ A = args->A;
  float * __restrict__ B = args->B;
  float * __restrict__ C = args->C;
  const uint32_t N = args->N;
  const uint32_t M = args->M;
  const uint32_t K = args->K;

  // Detect the chunk depending on the Core that executes
  const uint32_t blockSize = (N+NUM_CORES-1) / NUM_CORES;
  const uint32_t start = pi_core_id()*blockSize;
  const uint32_t stop = start+blockSize > N ? N : start+blockSize;

  // Perform partial Matrix Multiplication
  for (uint32_t i=start; i < stop; i++) 
  {
    for (uint32_t j = 0; j < M; j++) 
    {
      float temp = 0;
      for (uint32_t k = 0; k < K; k++) 
      {
        temp += A[i*K+k] * B[j+k*M];
      } 
      C[i*M+j] = temp;
    } 
  }   
}
```

## Automate the Generation of the ODL application code

PULP-TrainLib includes the _TrainLib_Deployer_, a code generator tool for On-Device Learning on PULP platforms. 
The tool takes the DNN graph description and the platform compute and memory budgets (e.g., number of cores, working memory) and produces the application code within the specified project folder path (e.g., `./CNN_FP32/`).
The application code includes a static copy of the pulp-trainlib library, the main and the DNN training functions (net.c and net.h) and a reference model (or Golden Model) for testing the generated code. 
The DNN graph is provided by the user as a list of layers directly into the `USER SECTION` of the [TrainLib_Deployer](../pulp-trainlib/tools/TrainLib_Deployer/TrainLib_Deployer.py) script.
The current TrainLib_Deployer is still a prototype to validate the concept. We expect a refactoring of the tool in the next versions of the PULP-TrainLib framework.

As an example, the example program in this folder has been generated by calling:
```
cd pulp-trainlib/tools/TrainLib_Deployer
python TrainLib_Deployer.py
```
using the following DNN description:
```
# ---------------------
# --- USER SETTINGS ---
# ---------------------

# GENERAL PROPERTIES
project_name    = 'CNN_FP32'
project_path    = '../../../Ex01-TrainLib_Deployer/'
proj_folder     = project_path + project_name + '/'

# TRAINING PROPERTIES
epochs          = 1
batch_size      = 1                   # BATCHING NOT IMPLEMENTED!!
learning_rate   = 0.001
optimizer       = "SGD"                # Name of PyTorch's optimizer
loss_fn         = "MSELoss"            # Name of PyTorch's loss function

# ------- NETWORK GRAPH --------
# Manually define the list of the network (each layer in the list has its own properties in the relative index of each list)
layer_list      = [ 'conv2d', 'ReLU', 'linear']
# Layer properties
sumnode_connections = [0, 0, 0]             # For Skipnode and Sumnode only, for each Skipnode-Sumnode couple choose a value and assign it to both, all other layer MUST HAVE 0

in_ch_list      = [  8,  16, 16*6*6 ]       # Linear: size of input vector
out_ch_list     = [  16, 16, 32 ]           # Linear: size of output vector
hk_list         = [  3,  1,  1 ]            # Linear: = 1
wk_list         = [  3,  1,  1 ]            # Linear: = 1
# Input activations' properties
hin_list        = [  8,  6,  1 ]            # Linear: = 1
win_list        = [  8,  6,  1 ]            # Linear: = 1
# Convolutional strides
h_str_list      = [  1,  1,  1 ]            # Only for conv2d, maxpool, avgpool (NOT IMPLEMENTED FOR CONV2D)
w_str_list      = [  1,  1,  1 ]            # Only for conv2d, maxpool, avgpool (NOT IMPLEMENTED FOR CONV2D)
# Padding (bilateral, adds the specified padding to both image sides)
h_pad_list      = [  0,  0,  0 ]            # Only for conv2d, DW (NOT IMPLEMENTED)
w_pad_list      = [  0,  0,  0 ]            # Only for conv2d, DW (NOT IMPLEMENTED)
# Define the lists to call the optimized matmuls for each layer (see mm_manager_list.txt, mm_manager_list_fp16.txt or mm_manager function body)
opt_mm_fw_list  = [  0,  0,  0 ]
opt_mm_wg_list  = [  0,  0,  0 ]
opt_mm_ig_list  = [  0,  0,  0 ]
# Data type list for layer-by-layer deployment (mixed precision)
data_type_list  = ['FP32', 'FP32', 'FP32']
# Data layout list (CHW or HWC) 
data_layout_list    = ['CHW', 'CHW', 'CHW']   # TO DO
# ----- END OF NETWORK GRAPH -----




# EXECUTION PROPERTIES
NUM_CORES       = 8
L1_SIZE_BYTES   = 256*(2**10)
USE_DMA = 'NO'                          # choose whether to load all structures in L1 ('NO') or in L2 and use Single Buffer mode ('SB') or Double Buffer mode ('DB') 
# BACKWARD SETTINGS
SEPARATE_BACKWARD_STEPS = True          # If True, the tool writes separate weight and input gradient in the backward step
# PROFILING OPTIONS
PROFILE_SINGLE_LAYERS = True            # If True, profiles forward and backward layer-by-layer
# OTHER PROPERTIES
# Select if to read the network from an external source
READ_MODEL_ARCH = False                # NOT IMPLEMENTED!!

# ---------------------------
# --- END OF USER SETTING ---
# ---------------------------

```


## References

> D. Nadalini, M. Rusci, G. Tagliavini, L. Ravaglia, L. Benini, and F. Conti, "PULP-TrainLib: Enabling On-Device Training for RISC-V Multi-Core MCUs through Performance-Driven Autotuning" [SAMOS Pre-Print Version](https://www.samos-conference.com/Resources_Samos_Websites/Proceedings_Repository_SAMOS/2022/Papers/Paper_14.pdf), [Springer Published Version](https://link.springer.com/chapter/10.1007/978-3-031-15074-6_13)
