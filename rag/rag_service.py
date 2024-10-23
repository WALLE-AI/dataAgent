from typing import List
import loguru
from embedding.datasource.retrieval_service import RetrievalService
from entities.document import Document

class RAGService():
    def __init__(self,collection_name):
        self.desc = "RAG retriever based on transformer model"
        self.collection_name = collection_name
    def __str__(self):
        return self.desc
    
    def _self_rag(self):
        pass
    
    def _rig_rag(self):
        pass
    def _retrieve(self, query:str,reranker=False)->List[Document]:
        top_k = 5
        score = 0.5
        all_documents = []
        retravial = "semantic_search"
        all_documents = RetrievalService.retrieve(retrieval_method=retravial,docs_index=self.collection_name,query=query,
                              top_k=top_k,score_threshold=score,
                              reranking_model=reranker)
        loguru.logger.info(f"reponse:{all_documents}")
        return all_documents
    
    def _build_prompt(self, query:str):
        pass
    
    def rewrite_query(self, query):
        pass
    
    def expand_query(self, query):
        pass
        
    def rag(self,query):
        ##query preprocessing 什么扩展、改写等等
        ##build prompt few-shot
        ##execute llm 
        ##potsprocess response
        pass