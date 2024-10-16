'''
https://github.com/openai/openai-cookbook/blob/main/examples/Clustering.ipynb
https://medium.com/@20pd11/unveiling-text-clustering-exploring-algorithms-and-text-embeddings-1f8776a84ddb
https://docs.cohere.com/docs/clustering-using-embeddings
'''
import json
from typing import List
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from models.embedding import EmbeddingApi
from utils.helper import single_measure_execution_time


class EmbeddingCluster():
    def __init__(self) -> None:
        self.dsc = "emdbedding clustering "
        self.n_clusters = 100
        
    def __str__(self) -> str:
        return self.dsc
    
    def read_file_to_q(self,dataset_file):
        docs_q = []
        with open(dataset_file,"r",encoding="utf-8") as file:
            for line in file:
                data  = json.loads(line)
                docs_q.append(data['input'])
        return docs_q
    @single_measure_execution_time
    def _docs_embedding(self,dataset:List[str]) -> List:
        docs_q_emb = EmbeddingApi.asyc_embed_documents(doc_list=dataset[:10000])
        return docs_q_emb
        
    @classmethod
    def kmeans_embedding_cluster(cls,dataset_file):
        docs_q = cls().read_file_to_q(dataset_file=dataset_file)
        docs_q_emb = cls()._docs_embedding(docs_q)        
        matrix = np.vstack(docs_q_emb)
        kmeans = KMeans(n_clusters=cls.n_clusters, init="k-means++", random_state=42)
        kmeans.fit(matrix)
        labels = kmeans.labels_
        data_df = pd.DataFrame()
        data_df["Cluster"] = labels
        data_df.groupby("Cluster").Score.mean().sort_values()
        return data_df
        
    
    