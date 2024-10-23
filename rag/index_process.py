from pathlib import Path
from typing import List
import uuid

import loguru
from tqdm import tqdm
from embedding.datasource.vector_factory import Vector
from entities.document import Document
from parser.extract_processor import ExtractProcessor
from rag.rag_service import RAGService
from utils.helper import MeasureExecutionTime, get_directory_all_markdown_files


class IndexProcess():
    def __init__(self,file_path,index_name):
        self.desc = "index processing"
        self.vdb = Vector(index_name)
        self.file_path = file_path
        self.etl_type = "localhost"
    def __str__(self) -> str:
        return self.desc
    
    
    
    def extract_text(self):
        docs = ExtractProcessor.extract(self.file_path,self.etl_type)
        return self._preproces_docs_metadata(docs)
    
    def _preproces_docs_metadata(self,docs:List[Document]) -> List[Document]:
        file_name = Path(self.file_path).stem.split("_")[-1]
        for doc in docs:
            metadata = {
                "doc_id":str(uuid.uuid4()),
                "file_name":file_name
            }
            doc.metadata.update(metadata)
        return docs
            
    def late_chunking_embedding():
        pass
    
    def meta_chunking_embedding():
        pass
    @MeasureExecutionTime
    def embedding_process_index(self, docs:List[Document]):
        self.vdb.create(docs)
        
        
        
def test_markdown_file_embedding():
    markdown_files_dir = "data/markdown/"
    markdonw_files = get_directory_all_markdown_files(markdown_files_dir)
    llm_type="localhost"
    model_name="Qwen2.5-72B-Instruct-AWQ"
    collection_name =  "Vector_index_markdown_"+str(uuid.uuid4())+"_Node"
    loguru.logger.info(f"collection_name:{collection_name}")
    for file in tqdm(markdonw_files[:10]):
        process = IndexProcess(file,collection_name)
        docs = process.extract_text()
        loguru.logger.info(f"docs:{len(docs)}")
        process.embedding_process_index(docs)
        
        
def test_markdon_file_rag():
    test = "地基基础施工前完成什么"
    collection_name = "Vector_index_markdown_e4ec885c-4b7c-48c3-9bd2-af4f0a55c6dc_Node"
    rag = RAGService(collection_name)
    context = rag._retrieve(test,reranker=True)
    loguru.logger.info(f"context:{context}")
    
    
    
        


        
        
        