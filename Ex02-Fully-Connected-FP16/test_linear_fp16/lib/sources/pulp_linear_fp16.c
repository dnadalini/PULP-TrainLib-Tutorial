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
 * Authors: Davide Nadalini, Leonardo Ravaglia
*/ 

#include "pulp_train_utils_fp16.h"
#include "pulp_matmul_fp16.h"
#include "pulp_linear_fp16.h"
#include "pulp_vector_matrix_fp16.h"

void pulp_linear_fp16_fw_cl( void * Linear_args_fp16 )
{
  struct Linear_args_fp16 * FC_args = (struct Linear_args_fp16 *) Linear_args_fp16;
  fp16 *coeffData = FC_args->coeff->data;
  fp16 *outData = FC_args->output->data;  
  fp16 *inputData = FC_args->input->data;

  int opt_matmul_type = FC_args->opt_matmul_type_fw;
  int transposed_weights = FC_args->transposed_weights;

  struct matMul_args_fp16 matMul_args;

  matMul_args.A = coeffData;
  matMul_args.B = inputData;
  matMul_args.C = outData;
  matMul_args.N = FC_args->output->dim;
  matMul_args.K = FC_args->input->dim;
  matMul_args.M = 1;
  matMul_args.trans_B = 0;

  #ifndef OPTIMIZE
  pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args);
  #else
  struct mm_manager_args_fp16 man_args;
  man_args.mm_args = &matMul_args;
  man_args.layer_type = LAYER_LINEAR;
  man_args.step_type = STEP_FW;
  man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;
  pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args);
  #endif
}


void pulp_linear_fp16_bw_cl( void * Linear_args_fp16 )
{
  struct Linear_args_fp16 * FC_args = (struct Linear_args_fp16 *) Linear_args_fp16;
  int skip_in_grad = FC_args->skip_in_grad;

  pulp_linear_fp16_bw_param_grads_cl(Linear_args_fp16);
  if (skip_in_grad == 0) 
  {
    pulp_linear_fp16_bw_input_grads_cl(Linear_args_fp16); 
  }
}


void pulp_linear_fp16_bw_param_grads_cl( void * Linear_args_fp16 )
{
  struct Linear_args_fp16 * FC_args = (struct Linear_args_fp16 *) Linear_args_fp16;
  fp16 *coeffData = FC_args->coeff->data;
  fp16 *inData = FC_args->input->data;
  fp16 *outData = FC_args->output->data;
  fp16 *coeffDiff = FC_args->coeff->diff;
  fp16 *outDiff = FC_args->output->diff;  
  fp16 *inDiff = FC_args->input->diff;

  int opt_matmul_type = FC_args->opt_matmul_type_wg;
  int transposed_weights = FC_args->transposed_weights;

  struct matMul_args_fp16 matMul_args;

  matMul_args.A = outDiff;
  matMul_args.B = inData;
  matMul_args.C = coeffDiff;
  matMul_args.N = FC_args->output->dim;
  matMul_args.K = 1;
  matMul_args.M = FC_args->input->dim;
  matMul_args.trans_B = 0;

  #ifndef OPTIMIZE
  pi_cl_team_fork(NUM_CORES, mm_fp16, &matMul_args);
  #else
  struct mm_manager_args_fp16 man_args;
  man_args.mm_args = &matMul_args;
  man_args.layer_type = LAYER_LINEAR;
  man_args.step_type = STEP_WGT_GRAD;
  man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;
  pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args);
  #endif
}


void pulp_linear_fp16_bw_input_grads_cl( void * Linear_args_fp16 )
{
  struct Linear_args_fp16 * FC_args = (struct Linear_args_fp16 *) Linear_args_fp16;
  fp16 *coeffData = FC_args->coeff->data;
  fp16 *inData = FC_args->input->data;
  fp16 *outData = FC_args->output->data;
  fp16 *coeffDiff = FC_args->coeff->diff;
  fp16 *outDiff = FC_args->output->diff;  
  fp16 *inDiff = FC_args->input->diff;

  int opt_matmul_type = FC_args->opt_matmul_type_ig;
  int transposed_weights = FC_args->transposed_weights;

  struct matMul_args_fp16 matMul_args;

  matMul_args.A = outDiff;
  matMul_args.B = coeffData;
  matMul_args.C = inDiff;
  matMul_args.N = 1;
  matMul_args.K = FC_args->output->dim;
  matMul_args.M = FC_args->input->dim;
  matMul_args.trans_B = transposed_weights;

  #ifndef OPTIMIZE
  pi_cl_team_fork(NUM_CORES, mm_M_fp16, &matMul_args);
  // CODE FOR TUTORIAL PURPOSES
  #else
    #if MATMUL_TYPE == 0
    vm_naive(&matMul_args);
    #elif MATMUL_TYPE == 1
    vm_SIMD_naive(&matMul_args);
    #elif MATMUL_TYPE == 2
      #if NUM_CORES > 1
      pi_cl_team_fork(NUM_CORES, vm_T_SIMD_parallel, &matMul_args);
      #else
      vm_T_SIMD(&matMul_args);
      #endif
    #endif
  #endif
  /** USUAL CODE
  #else
  struct mm_manager_args_fp16 man_args;
  man_args.mm_args = &matMul_args;
  man_args.layer_type = LAYER_LINEAR;
  man_args.step_type = STEP_IN_GRAD;
  man_args.matmul_type = opt_matmul_type; //MATMUL_TYPE;
  pi_cl_team_fork(NUM_CORES, mm_manager_fp16, &man_args);
  #endif
  */
}
