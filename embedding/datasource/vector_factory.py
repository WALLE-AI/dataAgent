from abc import ABC, abstractmethod
import os
from typing import Any, Optional

from chromadb import Embeddings

from embedding.datasource.vector_base import BaseVector
from embedding.datasource.vector_type import VectorType
from entities.document import Document


class AbstractVectorFactory(ABC):
    @abstractmethod
    def init_vector(self, collection_name, attributes: list, embeddings: Embeddings) -> BaseVector:
        raise NotImplementedError

    @staticmethod
    def gen_index_struct_dict(vector_type: VectorType, collection_name: str) -> dict:
        index_struct_dict = {"type": vector_type, "vector_store": {"class_prefix": collection_name}}
        return index_struct_dict
    
    
class Vector:
    def __init__(self, collection_name, attributes: Optional[list] = None):
        if attributes is None:
            attributes = ["doc_id", "dataset_id", "document_id", "doc_hash"]
        self._collection_name = collection_name
        self._embeddings = self._get_embeddings()
        self._attributes = attributes
        self._vector_processor = self._init_vector()

    def _init_vector(self) -> BaseVector:
        vector_type =os.getenv("VECTOR_STORE")
        if not vector_type:
            raise ValueError("Vector store must be specified.")

        vector_factory_cls = self.get_vector_factory(vector_type)
        return vector_factory_cls().init_vector(self._collection_name, self._attributes, self._embeddings)

    @staticmethod
    def get_vector_factory(vector_type: str) -> type[AbstractVectorFactory]:
        match vector_type:
            case VectorType.CHROMA:
                from embedding.datasource.vdb.chroma.chroma_vector import ChromaVectorFactory
                return ChromaVectorFactory
            case _:
                raise ValueError(f"Vector store {vector_type} is not supported.")

    def create(self, texts: Optional[list] = None, **kwargs):
        if texts:
            embeddings = self._embeddings.aembed_documents([document.page_content for document in texts])
            self._vector_processor.create(texts=texts, embeddings=embeddings, **kwargs)

    def add_texts(self, documents: list[Document], **kwargs):
        if kwargs.get("duplicate_check", False):
            documents = self._filter_duplicate_texts(documents)

        embeddings = self._embeddings.aembed_documents([document.page_content for document in documents])
        self._vector_processor.create(texts=documents, embeddings=embeddings, **kwargs)

    def text_exists(self, id: str) -> bool:
        return self._vector_processor.text_exists(id)

    def delete_by_ids(self, ids: list[str]) -> None:
        self._vector_processor.delete_by_ids(ids)

    def delete_by_metadata_field(self, key: str, value: str) -> None:
        self._vector_processor.delete_by_metadata_field(key, value)

    def search_by_vector(self, query: str, **kwargs: Any) -> list[Document]:
        query_vector = self._embeddings.aembed_query(query)
        return self._vector_processor.search_by_vector(query_vector, **kwargs)

    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        return self._vector_processor.search_by_full_text(query, **kwargs)

    def delete(self) -> None:
        self._vector_processor.delete()
        # delete collection redis cache
        # if self._vector_processor.collection_name:
        #     collection_exist_cache_key = "vector_indexing_{}".format(self._vector_processor.collection_name)
        #     redis_client.delete(collection_exist_cache_key)

    def _get_embeddings(self) -> Embeddings:
        ##TODO:缓存embedding的数据便于重复查询,需要把inference_embedding_type ep
        inference_embedding_type = "tig_embedding_api"
        from models.embedding import EmbeddingModel
        return EmbeddingModel.get_embedding(inference_embedding_type)

    def _filter_duplicate_texts(self, texts: list[Document]) -> list[Document]:
        for text in texts.copy():
            doc_id = text.metadata["doc_id"]
            exists_duplicate_node = self.text_exists(doc_id)
            if exists_duplicate_node:
                texts.remove(text)

        return texts

    def __getattr__(self, name):
        if self._vector_processor is not None:
            method = getattr(self._vector_processor, name)
            if callable(method):
                return method

        raise AttributeError(f"'vector_processor' object has no attribute '{name}'")