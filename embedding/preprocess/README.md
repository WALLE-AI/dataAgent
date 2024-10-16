# Dataset Embedding Preprocess
* 对预训练数据和SFT数据进行去重预处理
* sklearn MinbatchKmeans
* faiss 基于相似度阈值
* single-pass +分层
* community detection
* sim哈希算法
* umap+hdbscan
* 走语义特征抽取+聚类没得错
* https://huggingface.co/blog/zh/dedup
* https://github.com/ekzhu/datasketch
* https://docs.cohere.com/docs/clustering-using-embeddings
* https://github.com/openai/openai-cookbook/blob/main/examples/Clustering.ipynb
* https://medium.com/@20pd11/unveiling-text-clustering-exploring-algorithms-and-text-embeddings-1f8776a84ddb