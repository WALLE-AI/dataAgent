"""Abstract interface for document loader implementations."""

from collections.abc import Iterator
import os
from typing import Optional

import loguru

from entities.document import Document
from parser.blob import Blob
from parser.extractor_base import BaseExtractor
from parser.vision.deepdoc.deepdoc_pdf_extractor import chunk
from parser.vision.gotocr2.inference import execute_gotocr2_model, init_model


class PdfVisionExtractor(BaseExtractor):
    """Load pdf files.


    Args:
        file_path: Path to the file to load.
    """

    def __init__(self, file_path: str, file_cache_key: Optional[str] = None,vision_type=None):
        """Initialize with file path."""
        self._file_path = file_path
        self._file_cache_key = file_cache_key
        self.model=None
        self.vision_type=None
        self.tokenizer=None
        if vision_type is not None:
            self.vision_type = vision_type
            if vision_type == "gotocr":
                ##初始化模型
                model_name = os.getenv("MODEL_PATH_GOT")
                self.model,self.tokenizer = init_model(model_name)

    def extract(self) -> list[Document]:
        if self.vision_type is not None:
            loguru.logger.info(f"extracting vision model execute:{self.vision_type}")
        if self.vision_type == "gotocr":
            return self.gotocr_extract()
        elif self.vision_type == "mineru":
            return self.mineru_extract()
        else:
            return self.deepdoc_extract()
    
    def mineru_extract(self)-> list[Document]:
        documents=[]
        return documents
    
    def deepdoc_extract(self)-> list[Document]:
        def dummy(prog=None, msg=""):
            pass
        ##解析后直接chunk
        doc_res = chunk(self._file_path, from_page=1, to_page=10000, callback=dummy)
        documents=[]
        for doc in doc_res:
           metadata = {"source": doc['docnm_kwd'], "page": doc["docnm_kwd"]}
           documents.append(Document(page_content=doc["content_with_weight"],metadata=metadata))
        return documents
    def gotocr_extract(self)-> list[Document]:
        documents=[]
        if self.model is not None and self.tokenizer is not None:
            doc_res,docs = execute_gotocr2_model(self.model,self.tokenizer,self._file_path)
            ##每一页的latex,测试后面要不要转markdown
            for doc in doc_res:
                documents.append(Document(page_content=doc))
            return documents
        return documents

            
