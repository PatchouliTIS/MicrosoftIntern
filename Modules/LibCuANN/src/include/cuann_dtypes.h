/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#pragma once

/* CUANN status type */
typedef enum {
    CUANN_STATUS_SUCCESS = 0,
    CUANN_STATUS_ALLOC_FAILED = 1,
    CUANN_STATUS_NOT_INITIALIZED = 2,
    CUANN_STATUS_INVALID_VALUE = 3,
    CUANN_STATUS_INTERNAL_ERROR = 4,
    CUANN_STATUS_FILEIO_ERROR = 5,
    CUANN_STATUS_CUDA_ERROR = 6,
    CUANN_STATUS_CUBLAS_ERROR = 7,
    CUANN_STATUS_INVALID_POINTER = 8,
    CUANN_STATUS_VERSION_ERROR = 9,
    CUANN_STATUS_UNSUPPORTED_DTYPE = 10,
} cuannStatus_t;

/* CUANN similarity type */
typedef enum {
    CUANN_SIMILARITY_INNER = 0,
    CUANN_SIMILARITY_L2 = 1,
} cuannSimilarity_t;

/* CUANN PQ center type */
typedef enum {
    CUANN_PQ_CENTER_PER_SUBSPACE = 0,
    CUANN_PQ_CENTER_PER_CLUSTER = 1,
} cuannPqCenter_t;
