#
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import os
import time

import cupy
import numpy
from cuann import ivfpq, libcuann, utils

# *** Important: Use Unified Memory ***
mempool = cupy.cuda.MemoryPool(cupy.cuda.malloc_managed)
cupy.cuda.set_allocator(mempool.malloc)


def get_dataset():
    file_dataset = os.getenv("MY_DATASET") or "/home/harold/repo/faiss-libcuann/smile/0/0.bin"
    file_trainset = ""
    file_queries = os.getenv("MY_QUERIES") or "/home/harold/repo/faiss-libcuann/smile/0/0.bin"
    file_gt = os.getenv("MY_GT") or "../bin/gt.bin"
    dataset = utils.memmap_bin_file(file_dataset, dtype)
    trainset = None
    queries = utils.memmap_bin_file(file_queries, dtype)
    gt = utils.memmap_bin_file(file_gt, "uint32")
    if trainset is None:
        indexes = numpy.arange(dataset.shape[0])
        numpy.random.shuffle(indexes)
        num_trainset = dataset.shape[0] // 10
        indexes = numpy.sort(indexes[:num_trainset])
        trainset = dataset[indexes]

    return dataset, trainset, queries, gt

# dataset: n row of doc[100 fp32]
# trainset is a subset of dataset
#
# queries: m rows of queries[100 fp32]
#
#


if __name__ == "__main__":
    # Set the parameters
    name = "100D_13k_14M"  # just a name to identify this dataset
    num_clusters = 10000  # number of clusters
    dim_pq = 128  # PQ dims
    bit_pq = 5  # 5 or 8
    dtype = "float32"
    topk = 10

    # Prepare dataset
    (
        dataset,
        trainset,
        queries,
        gt,
    ) = get_dataset()  # get the test dataset, in numpy.array format.

    print("# dataset.shape: {}".format(dataset.shape))  # (14000000, 100)
    print("# trainset.shape: {}".format(trainset.shape))  # (1400000, 100)
    print("# queries.shape: {}".format(queries.shape))  # (13265, 100)
    print("# gt.shape: {}".format(gt.shape))  # (13265, 10)

    num_dataset, dim_dataset = dataset.shape
    num_queries = queries.shape[0]

    index_output_dir = os.getenv("MY_OUTPUT_DIR") or "."
    index_file_name = "{}-{}-{}x{}.cluster_{}.pq_{}.{}_bit".format(
        name, dtype, num_dataset, dim_dataset, num_clusters, dim_pq, bit_pq
    )
    index_file_name = os.path.join(index_output_dir, index_file_name)

    cuann_ivfpq = ivfpq.CuannIvfPq()

    # Try loading the index, if not exists, build it
    # if cuann_ivfpq.load_index(index_file_name) is False:
    cuann_ivfpq.build_index(
        index_file_name,
        dataset,
        trainset,
        num_clusters,
        dim_pq,
        bit_pq,
        randomRotation=1,
    )

    # # starting search
    print(
        "\ntopk, num_probes, recall-{}, qps, elapsed_time (sec)".format(topk)
    )  # This print is for recall test

    for num_probes in [104, 112, 120, 128, 136]:
        cuann_ivfpq.set_search_params(num_probes, topk, num_queries)
        # work-around to reduce memory usage by CuPy
        mempool.free_all_blocks()

        time_start = time.time()
        I, D = cuann_ivfpq.search(queries)  # Here get the I and D results
        elapsed_time = time.time() - time_start

        qps = num_queries / elapsed_time
        recall = utils.compute_recall(I, gt)
        print(
            "{}, {}, {}, {}, {}".format(
                topk, num_probes, recall, qps, elapsed_time
            )
        )
