import threading
from typing import Optional

import loguru

from embedding.datasource.vector_factory import Vector
from entities.retrieval_methods import RetrievalMethod
from models.reranker import RankerApi


default_retrieval_model = {
    "search_method": RetrievalMethod.SEMANTIC_SEARCH.value,
    "reranking_enable": False,
    "reranking_model": {"reranking_provider_name": "", "reranking_model_name": ""},
    "top_k": 2,
    "score_threshold_enabled": False,
}


class RetrievalService:
    @classmethod
    def retrieve(
        cls,
        retrieval_method: str,
        docs_index: str,
        query: str,
        top_k: int,
        score_threshold: Optional[float] = 0.0,
        reranking_model: Optional[bool] = False,
    ):
        all_documents = []
        threads = []
        exceptions = []
        # retrieval_model source with keyword 可以参考ragflow中tokenier的方案
        # if retrieval_method == "keyword_search":
        #     keyword_thread = threading.Thread(
        #         target=RetrievalService.keyword_search,
        #         kwargs={
        #             "dataset_id": dataset_id,
        #             "query": query,
        #             "top_k": top_k,
        #             "all_documents": all_documents,
        #             "exceptions": exceptions,
        #         },
        #     )
        #     threads.append(keyword_thread)
        #     keyword_thread.start()
        # retrieval_model source with semantic
        if RetrievalMethod.is_support_semantic_search(retrieval_method):
            embedding_thread = threading.Thread(
                target=RetrievalService.embedding_search,
                kwargs={
                    "docs_index": docs_index,
                    "query": query,
                    "top_k": top_k,
                    "score_threshold": score_threshold,
                    "reranking_model": reranking_model,
                    "all_documents": all_documents,
                    "exceptions":exceptions
                },
            )
            threads.append(embedding_thread)
            embedding_thread.start()

        # retrieval source with full text
        if RetrievalMethod.is_support_fulltext_search(retrieval_method):
            full_text_index_thread = threading.Thread(
                target=RetrievalService.full_text_index_search,
                kwargs={
                    "docs_index": docs_index,
                    "query": query,
                    "top_k": top_k,
                    "reranking_model": reranking_model,
                    "all_documents": all_documents,
                    "exceptions":exceptions
                },
            )
            threads.append(full_text_index_thread)
            full_text_index_thread.start()

        for thread in threads:
            thread.join()

        if exceptions:
            exception_message = ";\n".join(exceptions)
            raise Exception(exception_message)

        if retrieval_method == RetrievalMethod.HYBRID_SEARCH.value:
            pass
        return all_documents

    # @classmethod
    # def external_retrieve(cls, dataset_id: str, query: str, external_retrieval_model: Optional[dict] = None):
    #     dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()
    #     if not dataset:
    #         return []
    #     all_documents = ExternalDatasetService.fetch_external_knowledge_retrieval(
    #         dataset.tenant_id, dataset_id, query, external_retrieval_model
    #     )
    #     return all_documents

    # @classmethod
    ##可以使用jieba切好关键字或者tokenier的方式
    # def keyword_search(
    #     cls, dataset_id: str, query: str, top_k: int, all_documents: list, exceptions: list
    # ):
    #     try:
    #         dataset = db.session.query(Dataset).filter(Dataset.id == dataset_id).first()

    #         keyword = Keyword(dataset=dataset)

    #         documents = keyword.search(cls.escape_query_for_search(query), top_k=top_k)
    #         all_documents.extend(documents)
    #     except Exception as e:
    #         exceptions.append(str(e))

    @classmethod
    def embedding_search(
        cls,
        docs_index: str,
        query: str,
        top_k: int,
        score_threshold: Optional[float],
        reranking_model: bool,
        all_documents: list,
        exceptions:list
    ):
        try:
            vector = Vector(docs_index)
            documents = vector.search_by_vector(
                    cls.escape_query_for_search(query),
                    search_type="similarity_score_threshold",
                    top_k=top_k,
                    score_threshold=score_threshold,
            )

            if documents:
                if reranking_model:
                    reranker_reponse = RankerApi.reranker_documents(query,documents)
                    all_documents.append(reranker_reponse)
                else:
                    all_documents.extend(documents)
        except Exception as e:
            loguru.logger.info(f"embedding search error:{e}")
            exceptions.append(str(e))
    @classmethod
    def full_text_index_search(
        cls,
        docs_index: str,
        query: str,
        top_k: int,
        reranking_model: Optional[dict],
        all_documents: list,
        exceptions:list
    ):
        try:
            vector_processor = Vector(docs_index)
            ##chroma没有全文搜索的能力，只有部分的向量数据库具备
            documents = vector_processor.search_by_full_text(cls.escape_query_for_search(query), top_k=top_k)
            if documents:
                if reranking_model:
                    reranker_reponse = RankerApi.reranker_documents(query,documents)
                    all_documents.append(reranker_reponse)
                else:
                    all_documents.extend(documents)
        except Exception as e:
            loguru.logger.info(f"embedding full search error:{e}")
            exceptions.append(str(e))

    @staticmethod
    def escape_query_for_search(query: str) -> str:
        return query.replace('"', '\\"')
