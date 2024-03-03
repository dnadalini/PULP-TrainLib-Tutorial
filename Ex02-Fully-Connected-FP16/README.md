# Optimizing On-Device Learning primitives

This tutorial presents the core optimizations that can be employed to build fast hardware-aware computational kernels on RISC-V Multicore MCUs. 

With this tutorial you will learn:
- how to consider the matrix expressions of a
- how to optimize a linear algebra operator with FP16 SIMD
- 

To understand these concepts, let's consider a Fully-Connected layer.

## Matrix Representation of ODL Layers

Most of the computational layers of CNNs can be visualized and executed as a Matrix Multiplication (MM) between suitably reshaped tensors. In case of a Fully-Connected Layer, each training step can be represented as follows:

![Fully-Connected](../img/FC_steps.png)

In this representation, the weight tensor, the input data, and the output gradient are used to compute the output, the weight gradient and the input gradient of the Fully-Connected layer. In particular, the weights of the Fully-Connected layer can be stored as a matrix of size `Cout x Cin`, while the input and output activations are of size `1 x Cin` and `1 x Cout`, respectively. 

## Optimizing a Vector-Matrix operator with FP16 SIMD 

In case the MCU is equipped with SIMD units with Reduced Precision (e.g., vectorized FP16), the data layout can be exploited to speed up the computation. In particular, both `load` and `multiply-and-accumulate (mac)` instructions can be used in their vectorized form to reduce the total number of instructions to compute a linear algebra operator, e.g., a Matrix Multiplication. This can be performed by `loading two adjacent elements from a single tensor` and by `multiplying couples of elements with a single instruction`.

As a starting point, let's consider the Input Gradient step of a Fully-Connected Layer. By considering the previously presented expressions, this step can be represented as the `vector-matrix` multiplication of the `Output Gradient (O)` and the `Weights (W)`, to compute the `Input Gradient (I)`. In the following figure, the left part presents the naive implementation of said step. When looking at the memory, tensors are represented as 1-D arrays, where adjacent elements belong to the same row, while successive column elements feature a stride which is equal to the row length of the corresponding matrix. 

When performing a vector-matrix multiplication, the naive version of the operator, using SIMD, should:
- load 2 adjacent elements of O as a vector;
- load 2 separate single column elements of W;
- pack the elements of W in a vector of 2 elements;
- multiply the vectorized O and W;
- accumulate over the results by summing the previous partial products.

The `non-adjacent position of the elements of W`, as well as the `pack` instructions represent a non-ideal execution pattern. 

![](../img/MM_MMT_new.png)

The previous operation can be optimized by introducing two simple changes:
- the `W matrix is stored as transposed`;
- the vector-matrix operation is performed `row-by-row`, instead of `row-by-column`.

The resulting operation is depicted on the right of the figure. In this case, `both the O and the W are loaded as vectors` and the `2 mac instructions are executed as a single SIMD instruction`. No pack instruction is used. The accumulation is performed as in the previous case. 

As a result, the inner iteration of the linear algebra operator is brought to 3 instructions, from 5, theoretically reducing the total latency by 40%.



## Example: optimizing a Matvec kernel

The previously introduced concept can be implemented as a linear algebra operator in C code, starting from the naive expression of the `vector-matrix` multiplication:

```C
void vm_naive (void * void_args) 
{
    struct matMul_args_fp16 * args = (struct matMul_args_fp16 *) void_args;
    fp16 * A = args->A;     // In this example, the Output Gradient
    fp16 * B = args->B;     // In this example, the Weight matrix
    fp16 * C = args->C;     // In this example, the Input Gradient
    uint32_t M = args->M; 
    uint32_t K = args->K;  

    for (int j = 0; j < M; j++) 
    {
        fp16 temp = 0;
        for (int k = 0; k < K; k++) 
        {
            temp += A[k] * B[k*M + j];
        }
        C[j] = temp;
    }
}
```

`Total estimated instructions: K*M*(2 ld + 1 mac) + M*(1 st) ~= 3*M*K + M`

The naive vectorized expression can, then, be derived by performing vectorized loads of A, while the elements of B are loaded as in the previous case:

```C
void vm_SIMD_naive (void * void_args) 
{
    struct matMul_args_fp16 * args = (struct matMul_args_fp16 *) void_args;
    fp16 * A = args->A;     // In this example, the Output Gradient
    fp16 * B = args->B;     // In this example, the Weight matrix
    fp16 * C = args->C;     // In this example, the Input Gradient
    uint32_t M = args->M; 
    uint32_t K = args->K;  


    for (int j = 0; j < M; j++) 
    {
        v2f16 temp = (v2f16) {0, 0};
        for (int k = 0; k < K; k+=2) 
        {
            // Load vectorized A (2 adjacent elements)
            v2f16 Av = *((v2f16*) &A[k]);
            // Load non-adjacent B elements and pack them
            fp16 B0 = B[k*M + j];
            fp16 B1 = B[(k+1)*M + j];
            v2f16 Bv = (v2f16) {B0, B1};
            // Multiply elementwise
            temp += Av * Bv;
        }
        C[j] = temp[0] + temp[1];
    }
}
```

In this case, to multiply A and B elements in vectors of 2, the elements of B need to be loaded and packed. This is performed in the inner loop.

`Total estimated instructions: (K/2)*M*(3 ld + 1 pack + 1 mac) + M*(1 sum + 1 st) ~= (5/2)*K*M + 2*M`

Then, the most optimized code can be obtained by considering the B matrix (the Weight Matrix) as already transposed in memory and performing a single vectorized mac:

```C
void vm_T_SIMD (void * void_args) 
{
    struct matMul_args_fp16 * args = (struct matMul_args_fp16 *) void_args;
    fp16 * A = args->A;     // In this example, the Output Gradient
    fp16 * B = args->B;     // In this example, the Weight matrix
    fp16 * C = args->C;     // In this example, the Input Gradient
    uint32_t M = args->M; 
    uint32_t K = args->K;  

    for (int j = 0; j < M; j++) 
    {
        v2f16 temp = (v2f16) {0, 0};
        for (int k = 0; k < K; k+=2) 
        {
            // Load both adjacent elements for A and B
            v2f16 Av = *((v2f16*) &A[k]);
            v2f16 Bv = *((v2f16*) &B[j*K + k]);
            temp += Av * Bv;
        }
        C[j] = temp[0] + temp[1];
    }
}
```
`Total estimated instructions: (K/2)*M*(2 ld + 1 mac) + M*(1 sum + 1 st) ~= (3/2)*M*K + 2*M`

## Optimizing a Fully-Connected Layer: Input Gradient Step

Using the previous insights, the Input Gradient Step described above can be optimized by reducing by up to 40% the clock cycles to execute. The implemented functions can be found in [pulp_vector_matrix_fp16.h](./test_linear_fp16/lib/include/pulp_vector_matrix_fp16.h) and [pulp_vector_matrix_fp16.c](./test_linear_fp16/lib/sources/pulp_vector_matrix_fp16.c). In the following tests, a Fully-Connected Layer with input feature size of 128, output feature size of 128 and weights of size 128x128 is analyzed. To see the effects of this optimization, let's run a test. First, `source ../setup.sh`. Then:

```
cd test_linear_fp16/
make clean get_golden all run MATMUL_TYPE=0 TRANSPOSE_WEIGHTS=0
```

This first command launches the Input Gradient Step of the Fully-Connected with the naive Matrix Multiplication algorithm (like `vm_naive`). Therefore, no vectorization is introduced. In this case, we obtain:

```
--- vm_naive ---
Estimated Cycles:   3*M*K + M = 49280
Measured Cycles:                67386 
```

The second command launches the Input Gradient Step of the Fully-Connected with the naive SIMD Matrix Multiplication algorithm (like `mm_SIMD_naive`). 
```
make clean get_golden all run MATMUL_TYPE=1 TRANSPOSE_WEIGHTS=0
```

In this case, we obtain:

```
--- vm_SIMD_naive ---
Estimated Cycles:   (5/2)*K*M + 2*M = 41216
Measured Cycles:                      43611
```

The third command launches the Input Gradient Step of the Fully-Connected with the naive SIMD Matrix Multiplication algorithm (like `mm_T_SIMD`). 

```
make clean get_golden all run MATMUL_TYPE=2 TRANSPOSE_WEIGHTS=1
```

In this case, we obtain:

```
--- vm_T_SIMD ---
Estimated Cycles:   (3/2)*M*K + 2*M = 24832
Measured Cycles:                      35268
```

## References

> D. Nadalini, M. Rusci, G. Tagliavini, L. Ravaglia, L. Benini, and F. Conti, "PULP-TrainLib: Enabling On-Device Training for RISC-V Multi-Core MCUs through Performance-Driven Autotuning" [SAMOS Pre-Print Version](https://www.samos-conference.com/Resources_Samos_Websites/Proceedings_Repository_SAMOS/2022/Papers/Paper_14.pdf), [Springer Published Version](https://link.springer.com/chapter/10.1007/978-3-031-15074-6_13)

> D. Nadalini, M. Rusci, L. Benini, and F. Conti, "Reduced Precision Floating-Point Optimization for Deep Neural Network On-Device Learning on MicroControllers" [ArXiv Pre-Print](https://arxiv.org/abs/2305.19167)

