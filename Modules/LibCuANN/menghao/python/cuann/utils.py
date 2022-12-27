#
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import numpy
import cupy

def memmap_bin_file(bin_file, dtype):
    if bin_file is None:
        return None
    a = numpy.memmap(bin_file, dtype='uint32', shape=(2,))
    shape = (a[0], a[1])    # (row, dimension)
    # print('# {}: shape: {}, dtype: {}'.format(bin_file, shape, dtype))
    return numpy.memmap(bin_file, dtype=dtype, offset=8, shape=shape)


def compute_recall(res, gt):
    if gt is None:
        return None
    num_queries, topk = res.shape
    if gt.shape[0] != num_queries or gt.shape[1] < topk:
        return None
    if isinstance(res, cupy.ndarray):
        res = cupy.asnumpy(res)
    count = 0
    for i in range(num_queries):
        count += len(numpy.intersect1d(res[i], gt[i, :topk], assume_unique=True))
    return count / (num_queries * topk)
