# Optimizing On-Device Learning primitives

This tutorial aims at showing the core optimizations to build fast hardware-aware computational kernels on RISC-V Multicore MCUs. To understand the basics, let's consider a Fully-Connected layer.

## "Everything is a Matrix Multiplication"

Most of the computational layers of CNNs can be visualized and executed as a Matrix Multiplication (MM) between suitably reshaped tensors. In case of a Fully-Connected Layer, each training step can be represented as follows:

![Fully-Connected](../img/FC_steps.png)

In this representation, the weight tensor, the input data, and the output gradient are used to compute the output and the weight and input gradients of the Fully-Connected layer. 

## Optimizing Matrix-vector with FP16 SIMD 

`Show how the reshape + SIMD can be used to speed up the computation.`

![MM_vs_MMT](../img/MM_vs_MMT.png)

Specializzare per il caso di UN training step in cui si usa una determinata operazione matvec.

## Hands on: optimizing a Matvec kernel

```
Here we should provide an extra test for THIS tutorial to show how we can improve the performances of MM using the previous insights (simple case).

Example:
- We design a 1x1 MM with the second matrix transposed and see the performance improvement coming from a vectorized MM. In this case, no data reuse needs to be explained
```

```C
void mm_naive (void * void_args) 
{
    struct matMul_args_fp16 * args = (struct matMul_args_fp16 *) void_args;
    fp16 * A = args->A; 
    fp16 * B = args->B; 
    fp16 * C = args->C; 
    uint32_t N = args->N;  
    uint32_t M = args->M; 
    uint32_t K = args->K;  

    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < M; j++) 
        {
            float temp = 0;
            for (int k = 0; k < K; k++) 
            {
                temp += A[i*K + k] * B[k*M + j];
            }
            C[i*M + j] = temp;
        }
    }
}
```

`Total estimated instructions: N*K*M*(2 ld + 1 mac) + N*M*(1 st) ~= 3*N*M*K + N*M`

```C
void mm_T_naive (void * void_args) 
{
    struct matMul_args_fp16 * args = (struct matMul_args_fp16 *) void_args;
    fp16 * A = args->A; 
    fp16 * B = args->B; 
    fp16 * C = args->C; 
    uint32_t N = args->N;  
    uint32_t M = args->M; 
    uint32_t K = args->K;  

    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < M; j++) 
        {
            v2f16 temp = (v2f16) {0, 0};
            for (int k = 0; k < K; k+=2) 
            {
                v2f16 Av = *((v2f16*) &A[i*K + k]);
                v2f16 Bv = *((v2f16*) &B[j*K + k]);
                temp += Av * Bv;
            }
            C[i*M + j] = temp[0] + temp[1];
        }
    }
}
```
`Total estimated instructions: N*(K/2)*M*(2 ld + 1 mac) + N*M*(1 sum + 1 st) ~= (3/2)*M*N*K + 2*N*M`


## References

> D. Nadalini, M. Rusci, G. Tagliavini, L. Ravaglia, L. Benini, and F. Conti, "PULP-TrainLib: Enabling On-Device Training for RISC-V Multi-Core MCUs through Performance-Driven Autotuning" [SAMOS Pre-Print Version](https://www.samos-conference.com/Resources_Samos_Websites/Proceedings_Repository_SAMOS/2022/Papers/Paper_14.pdf), [Springer Published Version](https://link.springer.com/chapter/10.1007/978-3-031-15074-6_13)

> D. Nadalini, M. Rusci, L. Benini, and F. Conti, "Reduced Precision Floating-Point Optimization for Deep Neural Network On-Device Learning on MicroControllers" [ArXiv Pre-Print](https://arxiv.org/abs/2305.19167)

