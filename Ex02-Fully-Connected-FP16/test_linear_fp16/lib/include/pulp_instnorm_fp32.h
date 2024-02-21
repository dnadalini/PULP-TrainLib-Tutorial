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
 * Authors: Giacomo Saporetti
*/ 


/**
 * Instance Norm layer configuration structure
 */

/**
 * @brief small number used to avoid division by zero 
 */ 
#define EPSILON 1e-10

/**
 * @brief Structure for Instance Norm Training in FP32
 * @param input input feauture maps for the depthwise layer
 * @param output output feature maps for the depthwise layer
 * @param coeff coefficients to compute normalization, bias are included
 * @param skip_in_grad skips the computation of the input grad (1st DNN layer)
 */
struct InstNorm_args {
	struct blob * input;
	struct blob * output; 
	struct blob * coeff;
	int skip_in_grad;
};

/**
 * @brief Dummy forward function that calls the parallelized version
 * @param (void *)  (struct InstNorm_args void_args)
 */
void pulp_instnorm_fp32_fw_cl( void * InstNorm_args );

/**
 * @brief Backward function that calls both input and param gradient functions
 * @param (void *)  (struct InstNorm_args void_args)
 */
void pulp_instnorm_fp32_bw_cl( void * InstNorm_args );

/**
 * @brief Dummy backward param gradient function that calls the parallelized version
 * @param (void *)  (struct InstNorm_args void_args)
 */
void pulp_instnorm_fp32_bw_param_grads_cl( void * InstNorm_args );

/**
 * @brief Dummy backward input gradient function that calls the parallelized version
 * @param (void *)  (struct InstNorm_args void_args)
 */
void pulp_instnorm_fp32_bw_input_grads_cl( void * InstNorm_args );

/**
 * @brief Real forward function parallelized on multicore
 * @param (void *)  (struct InstNorm_args void_args)
 */
void pulp_instnorm_parallelized_fp32_fw_cl( void * InstNorm_args );
/**
 * @brief Real bacward function for input gradients parallelized on multicore
 * @param (void *)  (struct InstNorm_args void_args)
 */
void pulp_instnorm_parallelized_fp32_bw_input_grads_cl( void * InstNorm_args );
/**
 * @brief Real bacward function for parameters gradients parallelized on multicore
 * @param (void *)  (struct InstNorm_args void_args)
 */
void pulp_instnorm_parallelized_fp32_bw_param_grads_cl( void * InstNorm_args );