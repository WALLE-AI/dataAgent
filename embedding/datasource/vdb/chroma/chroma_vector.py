import json
import os
from typing import Any, Optional

import chromadb
from chromadb import Embeddings, QueryResult, Settings
from pydantic import BaseModel

from embedding.datasource.vector_base import BaseVector
from embedding.datasource.vector_factory import AbstractVectorFactory
from embedding.datasource.vector_type import VectorType
from entities.document import Document



class ChromaConfig(BaseModel):
    host: str
    port: int
    tenant: str
    database: str
    auth_provider: Optional[str] = None
    auth_credentials: Optional[str] = None

    def to_chroma_params(self):
        settings = Settings(
            # auth
            chroma_client_auth_provider=self.auth_provider,
            chroma_client_auth_credentials=self.auth_credentials
        )

        return {
            'host': self.host,
            'port': self.port,
            'ssl': False,
            'tenant': self.tenant,
            'database': self.database,
            'settings': settings,
        }


class ChromaVector(BaseVector):

    def __init__(self, collection_name: str, config: ChromaConfig):
        super().__init__(collection_name)
        self._client_config = config
        self._client = chromadb.HttpClient(**self._client_config.to_chroma_params())

    def get_type(self) -> str:
        return VectorType.CHROMA

    def create(self, texts: list[Document], embeddings: list[list[float]], **kwargs):
        if texts:
            # create collection
            self.create_collection(self._collection_name)

            self.add_texts(texts, embeddings, **kwargs)

    def create_collection(self, collection_name: str):
        #TODO: 每一篇文章create collection index还是需要把
        self._client.get_or_create_collection(collection_name)

    def add_texts(self, documents: list[Document], embeddings: list[list[float]], **kwargs):
        uuids = self._get_uuids(documents)
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]

        collection = self._client.get_or_create_collection(self._collection_name)
        collection.upsert(ids=uuids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def delete_by_metadata_field(self, key: str, value: str):
        collection = self._client.get_or_create_collection(self._collection_name)
        collection.delete(where={key: {'$eq': value}})

    def delete(self):
        self._client.delete_collection(self._collection_name)

    def delete_by_ids(self, ids: list[str]) -> None:
        collection = self._client.get_or_create_collection(self._collection_name)
        collection.delete(ids=ids)

    def text_exists(self, id: str) -> bool:
        collection = self._client.get_or_create_collection(self._collection_name)
        response = collection.get(ids=[id])
        return len(response) > 0

    def search_by_vector(self, query_vector: list[float], **kwargs: Any) -> list[Document]:
        collection = self._client.get_or_create_collection(self._collection_name)
        results: QueryResult = collection.query(query_embeddings=query_vector, n_results=kwargs.get("top_k", 5))
        score_threshold = kwargs.get("score_threshold", .0) if kwargs.get('score_threshold', .0) else 0.0

        ids: list[str] = results['ids'][0]
        documents: list[str] = results['documents'][0]
        metadatas: dict[str, Any] = results['metadatas'][0]
        distances: list[float] = results['distances'][0]

        docs = []
        for index in range(len(ids)):
            distance = distances[index]
            metadata = metadatas[index]
            if distance >= score_threshold:
                metadata['score'] = distance
                doc = Document(
                    page_content=documents[index],
                    metadata=metadata,
                )
                docs.append(doc)
         # Sort the documents by score in descending order
        docs = sorted(docs, key=lambda x: x.metadata['score'], reverse=True)
        return docs

    def search_by_full_text(self, query: str, **kwargs: Any) -> list[Document]:
        # chroma does not support BM25 full text searching
        return []


class ChromaVectorFactory(AbstractVectorFactory):
    def init_vector(self, collection_name, attributes: list, embeddings: Embeddings) -> BaseVector:
        # if dataset.index_struct_dict:
        #     class_prefix: str = dataset.index_struct_dict["vector_store"]["class_prefix"]
        #     collection_name = class_prefix.lower()
        # else:
        #     dataset_id = dataset.id
        #     collection_name = Dataset.gen_collection_name_by_id(dataset_id).lower()
        #     index_struct_dict = {"type": VectorType.CHROMA, "vector_store": {"class_prefix": collection_name}}
        #     dataset.index_struct = json.dumps(index_struct_dict)

        return ChromaVector(
            collection_name=collection_name,
            config=ChromaConfig(
                host=os.getenv("CHROMA_HOST"),
                port=os.getenv("CHROMA_PORT"),
                tenant= chromadb.DEFAULT_TENANT,
                database=chromadb.DEFAULT_DATABASE,
                auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                auth_credentials="starchat123456",
            ),
        )