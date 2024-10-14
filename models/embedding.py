import asyncio
import json
import os
import time
import aiohttp
import loguru
import requests


class EmbeddingApi():
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
    def asyc_embed_documents(cls,doc_list):
        embedding_list = [asyncio.run(cls().asyc_embedding(doc)) for doc in doc_list]
        return embedding_list
    
    @classmethod
    def embed_query(cls,query):
        return cls()._embedding(query)
    
    @classmethod
    def asyc_embed_query(cls,query):
        return asyncio.run(cls().asyc_embedding(query))
