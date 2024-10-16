'''
https://github.com/openai/openai-cookbook/blob/main/examples/Clustering.ipynb
https://medium.com/@20pd11/unveiling-text-clustering-exploring-algorithms-and-text-embeddings-1f8776a84ddb
https://docs.cohere.com/docs/clustering-using-embeddings
'''
from ast import literal_eval
import json
from typing import List
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from models.embedding import EmbeddingModel
from models.llm import LLMApi
from prompt.theme_prompt import THEME_ANLYSIS_PROMPT
from utils.helper import single_measure_execution_time


class EmbeddingCluster():
    def __init__(self) -> None:
        self.dsc = "emdbedding clustering "
        self.n_clusters = 100
        self.embed_model = EmbeddingModel.get_embedding("flagembedding")
        
    def __str__(self) -> str:
        return self.dsc
    
    def read_file_to_q_embedding(self,dataset_file):
        docs_data= []
        with open(dataset_file,"r",encoding="utf-8") as file:
            for line in file:
                data  = json.loads(line)
                embed,tokens = self._docs_single_embedding(data["input"])
                data['q_embedding'] = embed
                data['tokens'] = tokens
                docs_data.append(data)
        return pd.DataFrame(docs_data)

    @single_measure_execution_time
    def _docs_embedding(self,dataset:List[str]) -> List:
        docs_q_emb,tokens = self.embed_model.embed_documents(dataset)
        return docs_q_emb,tokens
    @single_measure_execution_time
    def _docs_single_embedding(self,doc:str) -> List:
        docs_q_emb,tokens = self.embed_model.embed_query(doc)
        return docs_q_emb,tokens
        
    @classmethod
    def kmeans_embedding_cluster(cls,dataset_file,save_file=None):
        data_df = cls().read_file_to_q_embedding(dataset_file=dataset_file)
        data_df['q_embedding']=data_df['q_embedding'].apply(np.array)     
        matrix = np.vstack(data_df['q_embedding'].values)
        kmeans = KMeans(n_clusters=cls().n_clusters, init="k-means++", random_state=42)
        kmeans.fit(matrix)
        labels = kmeans.labels_
        data_df["Cluster"] = labels
        if save_file is not None:
            data_df.to_csv(save_file, index=False)
        return data_df
    
    @classmethod
    def theme_embedding_cluster(cls,data_df,n_clusters):
        rev_per_cluster = 5
        for i in range(n_clusters):
            print(f"Cluster {i} Theme:", end=" ")

            reviews = "\n".join(
            data_df[data_df.Cluster == i]
            .combined.str.replace("Title: ", "")
            .str.replace("\n\nContent: ", ":  ")
            .sample(rev_per_cluster, random_state=42)
            .values
            )
            messages = [
                {"role": "user", "content": THEME_ANLYSIS_PROMPT.format(reviews=reviews)}
            ]
            ##调用大模型进行聚类的主体分析
            messages = LLMApi.build_prompt(reviews)
            response = LLMApi.call_llm(messages)

            sample_cluster_rows = data_df[data_df.Cluster == i].sample(rev_per_cluster, random_state=42)
            for j in range(rev_per_cluster):
                print(sample_cluster_rows.Score.values[j], end=", ")
                print(sample_cluster_rows.Summary.values[j], end=":   ")
                print(sample_cluster_rows.Text.str[:70].values[j])
        
    
    