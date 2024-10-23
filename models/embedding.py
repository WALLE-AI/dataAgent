from abc import ABC, abstractmethod
import asyncio
import json
import os
import time
from typing import List
import aiohttp
import loguru
import numpy as np
import requests

from entities.embedding import Embeddings
from entities.retrieval_methods import EmbeddingInferenceType
from utils.encoder import num_tokens_from_string


class TigEmbeddingApi(Embeddings):
    def __init__(self) -> None:
        self.des = "embedding api service"
        self.timeout = aiohttp.ClientTimeout(total=3000)
    def __str__(self) -> str:
        return self.des
    
    async def asyc_embedding(self,doc):
        url = os.getenv("EMBEDDING_SERVE_HOST") + ':9991/embed'
        headers = {'Content-Type': 'application/json'}
        data = {'inputs': [doc]}
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(url, headers=headers, data=json.dumps(data)) as response:
                    if response.status != 200:
                        loguru.logger.info(f"request error code {response.status}")
                    response_text = await response.text()
                    return json.loads(response_text)[0]
        except Exception as e:
                loguru.logger.info(f"seed request error: {e}")
    
    def _embedding(self,doc):
        url = os.getenv("EMBEDDING_SERVE_HOST") + ':9991/embed'
        headers = {'Content-Type': 'application/json'}
        data = {'inputs': [doc]}
        # data = {"query":"What is Deep Learning?", "texts": ["Deep Learning is not...", "Deep learning is..."]}
        # 发送POST请求
        response = requests.post(url, headers=headers, data=json.dumps(data))
        # 打印响应内容
        loguru.logger.info(f"Status Code:{response.status_code}")
        return json.loads(response.text)[0]
    @classmethod
    def embed_documents(cls,doc_list):
        embedding_list = [cls()._embedding(doc) for doc in doc_list]
        return embedding_list

    @classmethod
    def aembed_documents(cls,doc_list):
        embedding_list = [asyncio.run(cls().asyc_embedding(doc)) for doc in doc_list]
        return embedding_list
    
    @classmethod
    def embed_query(cls,query):
        return cls()._embedding(query)
    
    @classmethod
    def aembed_query(cls,query):
        return asyncio.run(cls().asyc_embedding(query))
    


class EmbeddingModel():
    def __init__(self):
        pass
    @classmethod
    def get_embedding(cls,inference_type):
        if inference_type == EmbeddingInferenceType.FLAG_EMBEDDING.value:
            return FlagEmbeddingInference
        elif inference_type == EmbeddingInferenceType.SENTENCE_TRANSFORMER.value:
            pass
        elif inference_type == EmbeddingInferenceType.FAST_EMBED.value:
            return FastEmbedInference
        elif inference_type == EmbeddingInferenceType.TGI_EMBEDDING_API.value:
            return TigEmbeddingApi
        else:
            loguru.logger.info(f"no support embedding inference type")

   
   
class FastEmbedInference(Embeddings):
    def __init__(self):
        self.desc = "fastembed embedding inference"
        self._model = None
        self.init_model()
    def init_model(self):
        from fastembed import TextEmbedding
        try:
            self._model = TextEmbedding(os.getenv("EMBEDDING_MODEL"),providers=["CUDAExecutionProvider"])
        except ValueError as e:
            loguru.logger.info(f"embedding init error:{e}")
    @classmethod
    def embed_documents(cls, texts: list):
        # Using the internal tokenizer to encode the texts and get the total
        # number of tokens
        batch_size=32
        encodings = cls()._model.model.tokenizer.encode_batch(texts)
        total_tokens = sum(len(e) for e in encodings)

        embeddings = [e.tolist() for e in cls()._model.embed(texts, batch_size)]

        return np.array(embeddings), total_tokens 
    
class FlagEmbeddingInference(Embeddings):
    def __init__(self):
        self.desc = "embedding inference  sentence transformer and flagembedding"
        self._model = None
        self._init_model()
        
    def _init_model(self):
        ##https://huggingface.co/BAAI/bge-en-icl 看看怎么集成
        '''
        model = FlagICLModel('BAAI/bge-en-icl', 
                     query_instruction_for_retrieval="Given a web search query, retrieve relevant passages that answer the query.",
                     examples_for_task=examples,  # set `examples_for_task=None` to use model without examples
                     use_fp16=True)
        ''' 
        from FlagEmbedding import FlagModel
        import torch
        try:
            self._model = FlagModel(os.getenv("EMBEDDING_MODEL"),query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                                            use_fp16=torch.cuda.is_available())
        except ValueError as e:
            loguru.logger.info(f"embedding init error:{e}")
    @classmethod
    def embed_query(cls,text:str) -> list[float]:
        token_count = num_tokens_from_string(text)
        return cls()._model.encode([text]).tolist()[0], token_count
    @classmethod
    def embed_documents(cls,text:List[str]) ->list[list[float]]:
        batch_size=32
        texts = [os.truncate(t, 2048) for t in text]
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        res = []
        for i in range(0, len(texts), batch_size):
            res.extend(cls()._model.encode(texts[i:i + batch_size]).tolist())
        return np.array(res), token_count
    
    