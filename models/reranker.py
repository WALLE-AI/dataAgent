import asyncio
import json
import os
from typing import List
import aiohttp
import loguru
import requests

from entities.document import Document


class RankerApi():
    def __init__(self) -> None:
        self.des = "reranker api service"
        self.timeout = aiohttp.ClientTimeout(total=3000)
    def __str__(self) -> str:
        return self.des
    
    def _reranker(self,query,risk_doc_list):
        url = os.getenv("EMBEDDING_SERVE_HOST") + ':9992/rerank'
        headers = {'Content-Type': 'application/json'}
        data = {"query":query, "texts": risk_doc_list,  "raw_scores": False,"return_text": True,}
        # 发送POST请求
        response = requests.post(url, headers=headers, data=json.dumps(data))
        # 打印响应内容
        loguru.logger.info(f"Status Code:{response.status_code}")
        return json.loads(response.text)
    
    async def asyc_reranker(self,query,risk_doc_list):
        url = os.getenv("EMBEDDING_SERVE_HOST") + ':9992/rerank'
        headers = {'Content-Type': 'application/json'}
        data = {"query":query, "texts": risk_doc_list,  "raw_scores": False,"return_text": True,}
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.post(url, headers=headers, data=json.dumps(data)) as response:
                    if response.status != 200:
                        loguru.logger.info(f"request error code {response.status}")
                    response_text = await response.text()
                    return json.loads(response_text)[0]
        except Exception as e:
                loguru.logger.info(f"seed request error: {e}")
    @classmethod
    def reranker_documents(cls,query,recall_doc_list:List[Document]):
        recall_doc = [recall_doc.page_content for recall_doc in recall_doc_list]
        embedding_list = cls()._reranker(query,recall_doc)
        return embedding_list
    @classmethod
    def async_reranker_documents(cls,query,recall_doc_list:List[Document])->List[Document]:
        recall_doc = [recall_doc.page_content for recall_doc in recall_doc_list]
        rerank_result = asyncio.run(cls().asyc_reranker(query,recall_doc))
        rerank_list_docs = []
        if isinstance(rerank_result,dict):
            reranker_doc = recall_doc_list[rerank_result["index"]]
            reranker_doc.metadata["index"] =rerank_result["index"]
            reranker_doc.metadata["score"] =rerank_result["score"]
            rerank_list_docs.append(reranker_doc)
        if isinstance(rerank_result,list):
            for doc in rerank_result:
                reranker_doc = recall_doc_list[doc["index"]]
                reranker_doc.metadata["index"] =doc["index"]
                reranker_doc.metadata["score"] =doc["score"]
                rerank_list_docs.append(reranker_doc)
        return rerank_list_docs