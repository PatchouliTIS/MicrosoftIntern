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

#include <stdint.h>
#include <cuda_runtime.h>
#include "cuann_dtypes.h"
#include "cuann_version.h"

#if defined(__cplusplus)
extern "C" {
#endif

/* Context */
struct cuannContext;
typedef struct cuannContext *cuannHandle_t;

cuannStatus_t cuannCreate(cuannHandle_t *handle);
cuannStatus_t cuannDestroy(cuannHandle_t handle);
cuannStatus_t cuannSetStream(cuannHandle_t handle, cudaStream_t stream);
cuannStatus_t cuannSetDevice(cuannHandle_t handle, int devId);

/* IvfPq */
struct cuannIvfPqDescriptor;
typedef struct cuannIvfPqDescriptor *cuannIvfPqDescriptor_t;

cuannStatus_t cuannIvfPqCreateDescriptor(cuannIvfPqDescriptor_t *desc);
cuannStatus_t cuannIvfPqDestroyDescriptor(cuannIvfPqDescriptor_t desc);

cuannStatus_t cuannIvfPqSetIndexParameters(
    cuannIvfPqDescriptor_t desc,
    const uint32_t numClusters,  /* Number of clusters */
    const uint32_t numDataset,  /* Number of dataset entries */
    const uint32_t dimDataset,  /* Dimension of each entry */
    const uint32_t dimPq,  /* Dimension of each entry after product quantization */
    const uint32_t bitPq,  /* Bit length of PQ */
    const cuannSimilarity_t similarity,
    const cuannPqCenter_t typePqCenter
    );

cuannStatus_t cuannIvfPqGetIndexParameters(
    cuannIvfPqDescriptor_t desc,
    uint32_t *numClusters,
    uint32_t *numDataset,
    uint32_t *dimDataset,
    uint32_t *dimPq,
    uint32_t *bitPq,
    cuannSimilarity_t *similarity,
    cuannPqCenter_t *typePqCenter
    );

cuannStatus_t cuannIvfPqGetIndexSize(
    cuannIvfPqDescriptor_t desc,
    size_t *size  /* bytes of dataset index */);

cuannStatus_t cuannIvfPqBuildIndex(
    cuannHandle_t handle,
    cuannIvfPqDescriptor_t desc,
    const void *dataset,  /* [numDataset, dimDataset] */
    const void *trainset,  /* [numTrainset, dimDataset] */
    cudaDataType_t dtype,
    uint32_t numTrainset,  /* Number of train-set entries */
    uint32_t numIterations,  /* Number of iterations to train kmeans */
    bool randomRotation,  /* If true, rotate vectors with randamly created rotation matrix */
    bool hierarchicalClustering,  /* If true, do kmeans training hierarchically */
    void *index  /* database index to build */);

cuannStatus_t cuannIvfPqSaveIndex(
    cuannHandle_t handle,
    cuannIvfPqDescriptor_t desc,
    const void *index,
    const char *fileName);

cuannStatus_t cuannIvfPqLoadIndex(
    cuannHandle_t handle,
    cuannIvfPqDescriptor_t desc,
    void **index,
    const char *fileName);

cuannStatus_t cuannIvfPqSetSearchParameters(
    cuannIvfPqDescriptor_t desc,
    const uint32_t numProbes,  /* Number of clusters to probe */
    const uint32_t topK,  /* Number of search results */
    cudaDataType_t internalDistanceDtype);

cuannStatus_t cuannIvfPqGetSearchParameters(
    cuannIvfPqDescriptor_t desc,
    uint32_t *numProbes,
    uint32_t *topK,
    cudaDataType_t *internalDistanceDtype);

cuannStatus_t cuannIvfPqSearch_bufferSize(
    cuannHandle_t handle,
    cuannIvfPqDescriptor_t desc,
    const void *index,
    uint32_t numQueries,
    size_t maxWorkspaceSize,
    size_t *workspaceSize);

cuannStatus_t cuannIvfPqSearch(
    cuannHandle_t handle,
    cuannIvfPqDescriptor_t desc,
    const void *index,
    const void *queries,  /* [numQueries, dimDataset] */
    cudaDataType_t dtype,
    uint32_t numQueries,
    uint64_t *neighbors,  /* [numQueries, topK] */
    float *distances,  /* [numQueries, topK] */
    void *workspace);

cuannStatus_t cuannPostprocessingRefine(
    uint32_t numDataset,
    uint32_t numQueries,
    uint32_t dimDataset,
    const void *dataset,  /* [numDataset, dimDataset] */
    const void *queries,  /* [numQueries, dimDataset] */
    cudaDataType_t dtype,
    cuannSimilarity_t similarity,
    uint32_t topK,
    const uint64_t *neighbors,  /* [numQueries, topK] */
    uint32_t refinedTopK,
    uint64_t *refinedNeighbors,  /* [numQueries, refinedTopK] */
    float *refinedDistances      /* [numQueries, refinedTopK] */
    );

cuannStatus_t cuannPostprocessingMerge(
    uint32_t numSplit,
    uint32_t numQueries,
    uint32_t topK,
    const uint32_t *eachNumDataset, /* [numSplit] */
    const uint64_t *eachNeighbors,  /* [numSplit, numQueries, topK] */
    const float *eachDistances,     /* [numSplit, numQueries, topK] */
    uint64_t *neighbors,  /* [numQueries, topK] */
    float *distances      /* [numQueries, topK] */
    );

#if defined(__cplusplus)
}
#endif /* __cplusplus */
