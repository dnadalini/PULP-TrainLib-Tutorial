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
 * @brief Naive vector-matrix multiplication
*/
void vm_naive (void * void_args);

/**
 * @brief Naive SIMD verson of vector-matrix multiplication
*/
void vm_SIMD_naive (void * void_args);

/**
 * @brief Optimized version of vector-matrix multiplication
*/
void vm_T_SIMD (void * void_args);

/**
 * @brief Parallelized version of the optimized vector-matrix multiplication
*/
void vm_T_SIMD_parallel (void * void_args);