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
import sys
import json
import time
import numpy
import cupy

import cuann
from cuann import libcuann
from cuann import ivfpq
from cuann import utils

# *** Important: Use Unified Memory ***
mempool = cupy.cuda.MemoryPool(cupy.cuda.malloc_managed)
cupy.cuda.set_allocator(mempool.malloc)

#
# Configs
#
f = open('config.json', 'r')
config_dict = json.load(f)
# print(config_dict)

name = config_dict['name']
file_dataset = config_dict['dataset']
file_trainset = config_dict['trainset']
file_queries = config_dict['queries']
file_gt = config_dict['gt']
dtype = config_dict['dtype']
index_params = config_dict[config_dict['name_index_params']]
search_params = config_dict[config_dict['name_search_params']]

num_clusters = index_params['num_clusters']
dim_pq = index_params['dim_pq']
bit_pq = index_params['bit_pq']

dataset = utils.memmap_bin_file(file_dataset, dtype)
trainset = utils.memmap_bin_file(file_trainset, dtype)
queries = utils.memmap_bin_file(file_queries, dtype)
gt = utils.memmap_bin_file(file_gt, 'uint32')
print('# dataset.shape: {}'.format(dataset.shape))

num_dataset, dim_dataset = dataset.shape
num_queries = queries.shape[0]
assert(queries.shape[1] == dim_dataset)
gt_topk = gt.shape[1]
assert(gt.shape[0] == num_queries)
# num_clusters = min(num_clusters, num_dataset // 1000)

create_index = True
if len(sys.argv) >= 2:
    file_index = sys.argv[1]
    if not os.path.exists(file_index):
        raise RuntimeError('File does not exits ({})'.format(file_index))
    create_index = False
else:
    file_index = 'models/{}-{}-{}x{}.cluster_{}.pq_{}.{}_bit'.format(
        name, dtype, num_dataset, dim_dataset, num_clusters, dim_pq, bit_pq)
print('# file_index: {}'.format(file_index))

#
#
#
cuann_ivfpq = ivfpq.CuannIvfPq()

#
# Load an index
#
if cuann_ivfpq.load_index(file_index) is False:
    if not create_index:
        raise RuntimeError('Failed to load index ({})'.format(file_index))

    #
    # Create an index
    #
    if trainset is None:
        indexes = numpy.arange(dataset.shape[0])
        numpy.random.shuffle(indexes)
        num_trainset = dataset.shape[0] // 10
        indexes = numpy.sort(indexes[:num_trainset])
        trainset = dataset[indexes]
    print('# trainset.shape: {}'.format(trainset.shape))
    num_trainset = trainset.shape[0]
    assert(trainset.shape[1] == dim_dataset)

    cuann_ivfpq.build_index(file_index, dataset, trainset,
                            num_clusters, dim_pq, bit_pq,
                            randomRotation=1)

#
# Search
#
print('# queries.shape: {}'.format(queries.shape))
if gt is not None:
    print('# gt.shape: {}'.format(gt.shape))
cp_queries = cupy.array(queries)
for topk in search_params['topk']: 
    for refine_rate in search_params['refine_rate']:
        if refine_rate > 1 and topk > gt_topk:
            continue
        print('\ntopk, refine_rate, num_probes, recall-{}, qps, elapsed_time (sec)'.format(topk))
        gpu_topk = max(topk, int(topk * refine_rate))
        for num_probes in search_params['num_probes']:
            cuann_ivfpq.set_search_params(num_probes, gpu_topk, num_queries)

            # work-around to reduce memory usage by CuPy
            mempool.free_all_blocks()
            
            time_start = time.time()
            I, D = cuann_ivfpq.search(cp_queries)
            if topk < gpu_topk:
                I, D = cuann_ivfpq.refine(dataset, queries, cupy.asnumpy(I), topk)
            elapsed_time = time.time() - time_start

            qps = num_queries / elapsed_time
            recall = utils.compute_recall(I, gt)
            print('{}, {}, {}, {}, {}, {}'.format(
                topk, refine_rate, num_probes, recall, qps, elapsed_time))
