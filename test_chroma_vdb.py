import uuid
from dotenv import load_dotenv
import loguru

from embedding.datasource.retrieval_service import RetrievalService
from embedding.datasource.vector_factory import Vector
from entities.document import Document
from models.embedding import EmbeddingApi
from models.reranker import RankerApi
load_dotenv()

if __name__ == "__main__":
    loguru.logger.info("dataAgent starting...")
    ##test embedding
    # test = "hello world"
    # embedding_response = EmbeddingApi.embed_query(test) 'Vector_index_e570e853-291a-4338-9e83-d9762c22a1bb_Node'
    # loguru.logger.info(f"embedding:{embedding_response}")
    doc_id = str(uuid.uuid4())
    collection_name =  "Vector_index_"+doc_id+"_Node" 
    collection_name =  "Vector_index_96974a26_8dcd_4500_b2c3_4259a61f313a_Node_test"
    # vec = Vector(collection_name)
    # # doc_res = ["你是谁","你能够做什么","建筑大模型解决质量安全隐患问题","什么是建筑大模型，该模型具备该领域什么样特性"]
    # # documents=[]
    # # for doc in doc_res:
    # #     metadata = {"doc_id":str(uuid.uuid4())}
    # #     documents.append(Document(page_content=doc,metadata=metadata))
    # # vec.create(documents)
    query = "漏筋"
    top_k = 5
    score = 0.5
    reranking_model = True
    all_documents = []
    retravial = "semantic_search"
    # response = vec.search_by_vector(query,score_threshold=0.2)
    # reranker_reponse = RankerApi.reranker_documents(query,response)
    # loguru.logger.info(f"reponse:{reranker_reponse}")
    all_documents = RetrievalService.retrieve(retrieval_method=retravial,docs_index=collection_name,query=query,
                              top_k=top_k,score_threshold=score,
                              reranking_model=reranking_model)
    loguru.logger.info(f"reponse:{all_documents}")
    