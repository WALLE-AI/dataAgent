import json
import os
from typing import List
import loguru
import requests

from entities.document import Document


class RankerApi():
    def __init__(self) -> None:
        self.des = "reranker api service"
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
    @classmethod
    def reranker_documents(cls,query,recall_doc_list:List[Document]):
        recall_doc = [recall_doc.page_content for recall_doc in recall_doc_list]
        embedding_list = cls()._reranker(query,recall_doc)
        return embedding_list