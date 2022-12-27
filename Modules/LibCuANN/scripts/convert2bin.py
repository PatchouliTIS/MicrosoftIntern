import numpy as np
from time import time, perf_counter
import struct
import os


def get_doc_dict(fname):
    doc_dict = {}
    cnt = 0
    with open(fname, "r") as f:
        line = f.readline()
        while line:
            ll = line.split("\t")
            ids = ll[0].strip()
            if ids in doc_dict:
                assert 0 == 1
            doc_dict[ids] = str(cnt)
            line = f.readline()
            cnt += 1
    return doc_dict


def get_vectors(fname):
    cache_file = "%s.npy" % fname
    if os.path.exists(cache_file):
        return np.load(cache_file)
    vectors = []
    cnt = 0
    with open(fname, "r") as f:
        line = f.readline()
        while line:
            ll = line.split("\t")
            ids = ll[1].split("|")
            # ids = line.strip().split("|")
            ids = [float(x) for x in ids]
            vectors.append(ids)
            line = f.readline()
            cnt += 1        # ID \t num1|num2|num3
            # break
    vector_tensor = np.array(vectors).astype("float32")
    np.save(cache_file, vector_tensor)
    print("------------Save cache %s-------" % cache_file)
    print(vector_tensor.shape)
    print(vector_tensor)
    return vector_tensor


def get_gt(fname):
    cache_file = "%s.npy" % fname
    if os.path.exists(cache_file):
        return np.load(cache_file)
    vectors = []
    with open(fname, "r") as f:
        line = f.readline()
        cnt = 0
        while line:
            ids = line.split(" ")
            ids = [int(x.strip()) for x in ids]
            vectors.append(ids)
            line = f.readline()
            cnt += 1
            # break
    vector_tensor = np.array(vectors)
    np.save(cache_file, vector_tensor)
    print("------------Save cache %s-------" % cache_file)
    print(vector_tensor.shape)
    print(vector_tensor)
    return vector_tensor


def vector2bin(vectors, out_file):
    num1 = vectors.shape[0]
    num2 = vectors.shape[1]

    floatlist = vectors.flatten().tolist()
    buf = struct.pack("%sf" % len(floatlist), *floatlist)

    with open(out_file, "wb") as fb:
        fb.write(struct.pack("i", num1))
        fb.write(struct.pack("i", num2))
        fb.write(buf)

    print("Done writing %dx%d into file %s" % (num1, num2, out_file))



if __name__ == "__main__":
    # doc_dataset_file = "C:/Users/menghaoli/Documents/menghao/libcuann/dataset/13k_14M_100D/doc_embedding.tsv"
    query_dataset_file = "./fixtures/query_embedding_without_id.tsv"
    # gt_dataset_file = "C:/Users/menghaoli/Documents/menghao/libcuann/dataset/13k_14M_100D/gt1.tsv"

    # doc_vecs = get_vectors(doc_dataset_file)
    query_vecs = get_vectors(query_dataset_file)
    # gt_vecs = get_gt(gt_dataset_file)

    # vector2bin(
    #     doc_vecs,
    #     "C:/Users/menghaoli/Documents/menghao/libcuann/dataset/13k_14M_100D/doc.bin",
    # )
    vector2bin(
        query_vecs,
        "./fixtures/query.bin",
    )
    # vector2bin_int(
    #     gt_vecs,
    #     "C:/Users/menghaoli/Documents/menghao/libcuann/dataset/13k_14M_100D/gt.bin",
    # )
