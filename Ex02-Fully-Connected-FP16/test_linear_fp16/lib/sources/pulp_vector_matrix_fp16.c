/*
 * Copyright (C) 2021-2022 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pulp_train_utils_fp16.h"
#include "pulp_vector_matrix_fp16.h"

/**
 * Vector-Matrix Operators for the tutorial
*/

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