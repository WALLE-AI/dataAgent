from typing import List, Tuple
import loguru
import numpy as np
import torch
from parser.splitter.late_chunk import chunked_pooling
from parser.splitter.late_chunk.chunking import Chunker
from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM

def cosine_similarity(vector1, vector2):
    vector1_norm = vector1 / np.linalg.norm(vector1)
    vector2_norm = vector2 / np.linalg.norm(vector2)
    return np.dot(vector1_norm, vector2_norm)


class LateChunkingEmbedder:

    def __init__(self, 
            model: AutoModel,
            tokenizer: AutoTokenizer, 
            chunking_strategy: str = "sentences",
            n_sentences: int = 2
        ):

        self.model = model
        self.tokenizer = tokenizer

        self.chunker = Chunker(chunking_strategy = chunking_strategy)
        self.n_sentences = n_sentences
        self.chunk_size = 512

    ##TODO:需要使用GPU来进行，要不速度太慢了，如何API服务化了
    def run(self, document: str):
        chunks,annotations = self.chunker.chunk(text=document, tokenizer=self.tokenizer,chunk_size=self.chunk_size,n_sentences=self.n_sentences)
        annotations = [annotations]
        model_inputs = self.tokenizer(
            document,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=8192,
        )
        if torch.cuda.is_available():
            loguru.logger.info(f"model loads to gpu")
            model_inputs = model_inputs.to("cuda")
        ##可以采用sentence_transformers来做
        model_outputs = self.model(**model_inputs)
        self.output_embs = chunked_pooling(
            model_outputs, annotations, max_length=8192, 
        )[0]
        chunks_embs_list = []
        if len(self.output_embs) == len(chunks):
            for chunk,emb in zip(chunks,self.output_embs):
                chunks_embs_list.append((chunk,emb.tolist()))
        return chunks_embs_list

    ##这里查询没有关系，可以使用向量数据库来解决，只是解决chunk embedding的问题
    def query(self, query: str):
        if "output_embs" not in dir(self):
            raise ValueError("no embeddings calculated, use .run(document) to create chunk embeddings")
        query_embedding = self.model.encode(query)
        similarities = []
        for emb in self.output_embs:
            similarities.append(cosine_similarity(query_embedding, emb))
        
        return similarities
    

    


    
