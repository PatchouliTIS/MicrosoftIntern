import numpy as np

from cuann import ivfpq

num_clusters = 100  # number of clusters
dim_pq = 64  # PQ dims
bit_pq = 5  # 5 or 8
dtype = "float32"
topk = 10


def main():
    raw_dataset = np.loadtxt("./fixtures/query_embedding_without_id.tsv", delimiter="|", dtype=dtype)

    mem_dataset = np.memmap("query_embedding.temp", dtype=dtype, mode="w+", shape=raw_dataset.shape)

    mem_dataset[:] = raw_dataset[:]

    print(mem_dataset.shape)

    indexes = np.arange(mem_dataset.shape[0])
    np.random.shuffle(indexes)

    trainset_num = mem_dataset.shape[0] // 10
    trainset = mem_dataset[np.sort(indexes[:trainset_num])]

    print(trainset.shape)

    cuann_ivfpq = ivfpq.CuannIvfPq()
    cuann_ivfpq.build_index(
        "index.pkl",
        mem_dataset,
        trainset,
        num_clusters,
        dim_pq,
        bit_pq,
        randomRotation=1,
    )


if __name__ == "__main__":
    main()
