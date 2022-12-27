#
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from libc.stdint cimport intptr_t, uint32_t, uint64_t
from libcpp cimport bool

cdef extern from *:
    ctypedef void* cuannHandle_t 'cuannHandle_t'
    ctypedef void* cuannIvfPqDescriptor_t 'cuannIvfPqDescriptor_t'
    ctypedef int cuannStatus_t 'cuannStatus_t'
    ctypedef int cuannSimilarity_t 'cuannSimilarity_t'
    ctypedef int cuannPqCenter_t 'cuannPqCenter_t'
    ctypedef int cudaDataType_t 'cudaDataType_t'

cpdef enum:
    # cudaStatus_t
    CUANN_STATUS_SUCCESS = 0
    CUANN_STATUS_ALLOC_FAILED = 1
    CUANN_STATUS_NOT_INITIALIZED = 2
    CUANN_STATUS_INVALID_VALUE = 3
    CUANN_STATUS_INTERNAL_ERROR = 4
    CUANN_STATUS_FILEIO_ERROR = 5
    CUANN_STATUS_CUDA_ERROR = 6
    CUANN_STATUS_CUBLAS_ERROR = 7
    CUANN_STATUS_INVALID_POINTER = 8
    CUANN_STATUS_VERSION_ERROR = 9
    CUANN_STATUS_UNSUPPORTED_DTYPE = 10

    # cuannSimilarity_t
    CUANN_SIMILARITY_INNER = 0
    CUANN_SIMILARITY_L2 = 1

    # cuannPqCenter_t
    CUANN_PQ_CENTER_PER_SUBSPACE = 0
    CUANN_PQ_CENTER_PER_CLUSTER = 1

    # cudaDataType_t
    CUDA_R_32F = 0  # float
    CUDA_R_16F = 2  # half
    CUDA_R_8I = 3  # int8
    CUDA_R_8U = 8  # uint8

cdef extern from "cuann.h":
    cuannStatus_t cuannCreate(cuannHandle_t* handle)
    cuannStatus_t cuannDestroy(cuannHandle_t handle)

    cuannStatus_t cuannIvfPqCreateDescriptor(cuannIvfPqDescriptor_t* desc)
    cuannStatus_t cuannIvfPqDestroyDescriptor(cuannIvfPqDescriptor_t desc)

    cuannStatus_t cuannIvfPqSetIndexParameters(
        cuannIvfPqDescriptor_t desc,
        uint32_t numClusters,  # Number of clusters
        uint32_t numDataset,  # Number of dataset entries
        uint32_t dimDataset,  # Dimension of each entry
        uint32_t dimPq,  # Dimension of each entry after product quantization
        uint32_t bitPq,  # Bit length of PQ
        cuannSimilarity_t similarity,
        cuannPqCenter_t typePqCenter)

    cuannStatus_t cuannIvfPqGetIndexParameters(
        cuannIvfPqDescriptor_t desc,
        uint32_t *numClusters,
        uint32_t *numDataset,
        uint32_t *dimDataset,
        uint32_t *dimPq,
        uint32_t *bitPq,
        cuannSimilarity_t *similarity,
        cuannPqCenter_t *typePqCenter)

    cuannStatus_t cuannIvfPqGetIndexSize(
        cuannIvfPqDescriptor_t desc,
        size_t *size)

    cuannStatus_t cuannIvfPqBuildIndex(
        cuannHandle_t handle,
        cuannIvfPqDescriptor_t desc,
        const void *dataset,
        const void *trainset,
        cudaDataType_t dtype,
        uint32_t numTrainset,
        uint32_t numIterations,
        bool randomRotation,
        bool hierarchicalClustering,
        void *index)
    
    cuannStatus_t cuannIvfPqSaveIndex(
        cuannHandle_t handle,
        cuannIvfPqDescriptor_t desc,
        const void *index,
        const char *fileName)

    cuannStatus_t cuannIvfPqLoadIndex(
        cuannHandle_t handle,
        cuannIvfPqDescriptor_t desc,
        void **index,
        const char *fileName)

    cuannStatus_t cuannIvfPqSetSearchParameters(
        cuannIvfPqDescriptor_t desc,
        uint32_t numProbes,  # Number of clusters to probe
        uint32_t topK,  # Number of seach results
        cudaDataType_t internalDistanceDtype)

    cuannStatus_t cuannIvfPqGetSearchParameters(
        cuannIvfPqDescriptor_t desc,
        uint32_t *numProbes,
        uint32_t *topK,
        cudaDataType_t *internalDistanceDtype)

    cuannStatus_t cuannIvfPqSearch_bufferSize(
        cuannHandle_t handle,
        cuannIvfPqDescriptor_t desc,
        const void *index,
        uint32_t numQueries,
        size_t maxWorkspaceSize,
        size_t *workspaceSize)

    cuannStatus_t cuannIvfPqSearch(
        cuannHandle_t handle,
        cuannIvfPqDescriptor_t desc,
        const void *index,
        const void *queries,
        cudaDataType_t dtype,
        uint32_t numQueries,
        uint64_t *neighbors,
        float *distances,
        void *workspace)

    cuannStatus_t cuannPostprocessingRefine(
        uint32_t numDataset,
        uint32_t numQueries,
        uint32_t dimDataset,
        const void *dataset,  # [numDataset, dimDataset], host ptr
        const void *queries,  # [numQueries, dimDataset], host ptr
        cudaDataType_t dtype,
        cuannSimilarity_t similarity,
        uint32_t topK,
        const uint64_t *neighbors,  # [numQueries, topK], host ptr
        uint32_t refinedTopK,
        uint64_t *refinedNeighbors,  # [numQueries, refinedTopK], host ptr
        float *refinedDistances      # [numQueries, refinedTopK], host ptr
    )
#
# Context
#

# Create handler
cpdef create():
    cdef cuannHandle_t handle
    status = cuannCreate(&handle)
    return status, <intptr_t>handle

# Destroy handler
cpdef destroy(intptr_t handle):
    status = cuannDestroy(<cuannHandle_t>handle)
    return status,

#
# IvfPq
#

# Create descriptor
cpdef ivfPqCreateDescriptor():
    cdef cuannIvfPqDescriptor_t desc
    status = cuannIvfPqCreateDescriptor(&desc)
    return status, <intptr_t>desc

# Destroy descriptor
cpdef ivfPqDestroyDescriptor(intptr_t desc):
    status = cuannIvfPqDestroyDescriptor(<cuannIvfPqDescriptor_t>desc)
    return status,

# Set index parameters
cpdef ivfPqSetIndexParameters(intptr_t desc,
                              uint32_t numClusters,
                              uint32_t numDataset,
                              uint32_t dimDataset,
                              uint32_t dimPq,
                              uint32_t bitPq,
                              cuannSimilarity_t similarity,
                              cuannPqCenter_t typePqCenter):
    status = cuannIvfPqSetIndexParameters(
        <cuannIvfPqDescriptor_t>desc,
        numClusters,
        numDataset,
        dimDataset,
        dimPq,
        bitPq,
        similarity,
        typePqCenter)
    return status,

# Get index parameters
cpdef ivfPqGetIndexParameters(intptr_t desc):
    cdef uint32_t numClusters
    cdef uint32_t numDataset
    cdef uint32_t dimDataset
    cdef uint32_t dimPq
    cdef uint32_t bitPq
    cdef cuannSimilarity_t similarity
    cdef cuannPqCenter_t typePqCenter
    status = cuannIvfPqGetIndexParameters(
        <cuannIvfPqDescriptor_t>desc,
        &numClusters,
        &numDataset,
        &dimDataset,
        &dimPq,
        &bitPq,
        &similarity,
        &typePqCenter)
    return status, numClusters, numDataset, dimDataset, dimPq, bitPq, similarity, typePqCenter

# Get size of index
cpdef ivfPqGetIndexSize(intptr_t desc):
    cdef size_t size
    status = cuannIvfPqGetIndexSize(
        <cuannIvfPqDescriptor_t>desc,
        &size)
    return status, size

# Build index
cpdef ivfPqBuildIndex(intptr_t handle,
                      intptr_t desc,
                      intptr_t dataset,  # [numDataset, dimDataset]
                      intptr_t trainset,  # [numTrainset, dimDataset]
                      cudaDataType_t dtype,  # CUDA_R_32F, CUDA_R_8I, or CUDA_R_8U
                      uint32_t numTrainset,
                      uint32_t numIterations,
                      randomRotation,
                      hierarchicalClustering,
                      intptr_t index):
    status = cuannIvfPqBuildIndex(
        <cuannHandle_t>handle,
        <cuannIvfPqDescriptor_t>desc,
        <const void*>dataset,
        <const void*>trainset,
        dtype,
        numTrainset,
        numIterations,
        <bool>randomRotation,
        <bool>hierarchicalClustering,
        <void*>index)
    return status,

# Save index
cpdef ivfPqSaveIndex(intptr_t handle,
                     intptr_t desc,
                     intptr_t index,
                     str name):
    cdef bytes b_name = name.encode()
    cdef const char* c_name = b_name
    status = cuannIvfPqSaveIndex(
        <cuannHandle_t>handle,
        <cuannIvfPqDescriptor_t>desc,
        <const void*>index,
        c_name)
    return status,
    
# Load index
cpdef ivfPqLoadIndex(intptr_t handle,
                     intptr_t desc,
                     str name):
    cdef bytes b_name = name.encode()
    cdef const char* c_name = b_name
    cdef void* index;
    status = cuannIvfPqLoadIndex(
        <cuannHandle_t>handle,
        <cuannIvfPqDescriptor_t>desc,
        &index,
        c_name)
    return status, <intptr_t>index
    
# Set search parameters
cpdef ivfPqSetSearchParameters(intptr_t desc,
                               uint32_t numProbes,
                               uint32_t topK,
                               cudaDataType_t internalDistanceDtype):
    status = cuannIvfPqSetSearchParameters(
        <cuannIvfPqDescriptor_t>desc,
        numProbes,
        topK,
        internalDistanceDtype)
    return status,

# Get search parameters
cpdef ivfPqGetSearchParameters(intptr_t desc):
    cdef uint32_t numProbes
    cdef uint32_t topK
    cdef cudaDataType_t internalDistanceDtype
    status = cuannIvfPqGetSearchParameters(
        <cuannIvfPqDescriptor_t>desc,
        &numProbes,
        &topK,
        &internalDistanceDtype)
    return status, numProbes, topK, internalDistanceDtype

# Get workspace size
cpdef ivfPqSearch_bufferSize(intptr_t handle,
                             intptr_t desc,
                             intptr_t index,
                             uint32_t maxBatchSize,
                             size_t maxWorkspaceSize):
    cdef size_t workspaceSize
    status = cuannIvfPqSearch_bufferSize(
        <cuannHandle_t>handle,
        <cuannIvfPqDescriptor_t>desc,
        <const void*>index,
        maxBatchSize,
        maxWorkspaceSize,
        &workspaceSize)
    return status, workspaceSize

# Search
cpdef ivfPqSearch(intptr_t handle,
                  intptr_t desc,
                  intptr_t index,
                  intptr_t queries,
                  cudaDataType_t dtype,
                  uint32_t numQueries,
                  intptr_t neighbors,
                  intptr_t distances,
                  intptr_t workspace):
    status = cuannIvfPqSearch(
        <cuannHandle_t>handle,
        <cuannIvfPqDescriptor_t>desc,
        <const void*>index,
        <const void*>queries,
        dtype,
        numQueries,
        <uint64_t*>neighbors,
        <float*>distances,
        <void*>workspace)
    return status,

cpdef postprocessingRefine(uint32_t numDataset,
                           uint32_t numQueries,
                           uint32_t dimDataset,
                           intptr_t dataset,  # [numDataset, dimDataset], host ptr
                           intptr_t queries,  # [numQueries, dimDataset], host ptr
                           cudaDataType_t dtype,
                           cuannSimilarity_t similarity,
                           uint32_t topK,
                           intptr_t neighbors,  # [numQueries, topK], host ptr
                           uint32_t refinedTopK,
                           intptr_t refinedNeighbors,   # [numQueries, refinedTopK], host ptr
                           intptr_t refinedDistances):  # [numQueries, refinedTopK], host ptr
    status = cuannPostprocessingRefine(
        numDataset,
        numQueries,
        dimDataset,
        <const void*>dataset,
        <const void*>queries,
        dtype,
        similarity,
        topK,
        <const uint64_t*>neighbors,
        refinedTopK,
        <uint64_t*>refinedNeighbors,
        <float*>refinedDistances)
    return status,
