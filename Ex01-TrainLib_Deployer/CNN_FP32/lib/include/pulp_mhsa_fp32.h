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


/**
 * Recurrent layer training functions, grouped into FW and BW
*/

/**
 * Authors: Alberto Dequino
*/ 


/**
 * Multi-Head Self Attention layer configuration structure
 */

/**
 * @brief Structure for MHSA Training in FP32
 * @param input             Input vector for the MHSA layer.
 * @param n_heads           Number of heads the attention operation is divided.
 * @param output            Output vector.
 * @param coeff_in          Weight for input projection.
 * @param coeff_out         Weight for output projection.
 * @param qkv               Query, Key and Values extracted from the input and packed into a single matrix
 * @param attention_map     Output of the MHSA module, pre-projection
 * @param temp_buffer       Support buffer used to save transposed matrices
 * @param grad              Support buffer used when calculating gradients for each computational head during MHSA backprop
 * @param head_buffer       Attention scores for every head
 * 
 */

struct Mhsa_args {
    struct blob * input;
    int 	n_heads; 
    int opt_matmul_type_fw;
    int opt_matmul_type_wg;
    int opt_matmul_type_ig;
    struct blob * output;
    struct blob * coeff_in;
    struct blob * coeff_out;
    struct blob * qkv;
    struct blob * attention_map;
    float * temp_buffer;
    float * grad;
    struct blob * head_buffer;
    struct blob * softmax_buffer;
    float * global_max;
    float * partial_exp_sum;
    float * maxes;
    float * sums;
};




/**
 * MHSA layer training functions, grouped into FW and BW
 */

// FORWARD FUNCTIONS

/**
 * @brief Forward pass function, forked on PULP cluster.
 * @param Mhsa_args structure configuring the MHSA layer.
 */
void pulp_mhsa_fp32_fw_cl(void * Mhsa_args);


/**
 * @brief Forward pass function, forked on PULP cluster, using partial softmax.
 * @param Mhsa_args structure configuring the MHSA layer.
 */
void pulp_mhsa_fp32_fw_cl_2(void * Mhsa_args);


// BACKWARD FUNCTIONS

/**
 * @brief Backward pass function, which internally calculate both weight gradient and input gradient.
 * @param Mhsa_args structure configuring the MHSA layer.
 */
void pulp_mhsa_fp32_bw_cl(void * Mhsa_args);
