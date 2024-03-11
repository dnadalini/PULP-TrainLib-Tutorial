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
 * Authors: Davide Nadalini, Leonardo Ravaglia, Alberto Dequino
*/ 

/**
 * Activation functions configuration structure
 */  


/**
 * @brief Structure for activation functions
 * @param input blob structure for the input data of the activation layer
 * @param output blob structure for the output data of the activation layer
 */
struct act_args {
    struct blob * input;
    struct blob * output;
};

/**
 * @brief Arguments for exponential and softmax in parallel
 * @param input   pointer to input vector
 * @param dim     dimension vector
 * @param output  pointer to output vector
 * @param sum     final sum value of all exponentials
*/
struct softmax_args{
  struct blob * input;
  struct blob * output;
  int L;
  int n_heads;
  float * global_max;
  float * partial_exp_sum;
  float * maxes;
  float * sums;
};



/**
 * Activation functions, both FW and BW
 **/


/**
 * @brief Forward pass function. Configure and pass a act_args structure pointer as argument.
 * @param input Input for sigmoid.
 * @param output Output of sigmoid.
*/
void pulp_sigmoid_fp32_fw_cl( void * act_args );

/**
 * @brief Backward pass function.
 * @param input Input for sigmoid.
 * @param output Output of sigmoid.
*/
void pulp_sigmoid_fp32_bw_cl( void * act_args );

/**
 * @brief Core function to implement the forward of sigmoid (allows parallelization, parallelize with pi_cl_team_fork(NUM_CORES, sigmoid_core_fw_fp32, &args)).
 * @param act_args Input and output data (data only will be used)
*/
void sigmoid_core_fw_fp32( void * act_args );

/**
 * @brief Core function to implement the backward of sigmoid (allows parallelization, parallelize with pi_cl_team_fork(NUM_CORES, sigmoid_core_bw_fp32, &args)).
 * @param act_args Input and output data (gradients only will be used)
*/
void sigmoid_core_bw_fp32( void * act_args );



/**
 * @brief Forward pass function. Configure and pass a act_args structure pointer as argument.
 * @param input Input for relu.
 * @param output Output of relu.
*/
void pulp_relu_fp32_fw_cl( void * act_args );

/**
 * @brief Backward pass function.
 * @param input Input for relu.
 * @param output Output of relu.
*/
void pulp_relu_fp32_bw_cl( void * act_args );

/**
 * @brief Core function to implement the forward of ReLU (allows parallelization, parallelize with pi_cl_team_fork(NUM_CORES, relu_core_fw_fp32, &args)).
 * @param act_args Input and output data (data only will be used)
*/
void relu_core_fw_fp32( void * act_args );

/**
 * @brief Core function to implement the backward of ReLU (allows parallelization, parallelize with pi_cl_team_fork(NUM_CORES, relu_core_bw_fp32, &args)).
 * @param act_args Input and output data (gradients only will be used)
*/
void relu_core_bw_fp32( void * act_args );



/**
 * @brief Forward pass function.
 * @param input Input for softmax.
 * @param output Output of softmax.
*/
void pulp_softmax_fp32_fw_cl( void * act_args );

/**
 * @brief Bakcward pass function.
 * @param input Input for softmax.
 * @param output Output of softmax.
*/
void pulp_softmax_fp32_bw_cl( void * act_args );

/**
 * @brief Forward pass function, second version using partial algorithm
 * @param input Input for softmax.
 * @param output Output of softmax.
*/
void pulp_partial_softmax_fp32_fw_cl( void * act_args );

/**
 * @brief Forward pass function, second version using partial algorithm
 * @param input Input for softmax.
 * @param output Output of softmax.
*/
void pulp_partial_softmax_simple_fp32_fw_cl( void * act_args );

/**
 * @brief Forward pass function, second version using partial algorithm.
 * @param input Input for softmax.
 * @param output Output of softmax.
*/
void pulp_partial_softmax_shift_fp32_fw_cl( void * act_args );

/**
 * @brief Forward pass function, third version using partial algorithm and taylor approximation.
 * @param input Input for softmax.
 * @param output Output of softmax.
*/
void pulp_partial_softmax_approximate_fp32_fw_cl(void * act_args);

/**
 * @brief Forward pass function that parallelize the fastertanh function (below).
 * @param pointer to a tanh_args struct
*/
void tanh_prll(
    void * args
);


/**
 * @brief A tanh implementation exploiting bit manipulation and "magic numbers" to be a bit faster.
 * @param float value
*/
static inline float fasttanh (
    float p
);

/**
 * @brief A power of 2 implementation exploiting bit manipulation and "magic numbers" to be a bit faster.
 * @param float value
*/
static inline float fastpow2 (
    float p
);

/**
 * @brief An exponential implementation exploiting bit manipulation and "magic numbers" to be a bit faster.
 * @param float value
*/
static inline float fastexp (
    float p
);


