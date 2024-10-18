"""Abstract interface for document loader implementations."""

from collections.abc import Iterator
from typing import Optional

import loguru

from entities.document import Document
from parser.blob import Blob
from parser.extractor_base import BaseExtractor


class LatexExtractor(BaseExtractor):
    """Load pdf files.


    Args:
        file_path: Path to the file to load.
    """

    def __init__(self, file_path: str, file_cache_key: Optional[str] = None):
        """Initialize with file path."""
        self._file_path = file_path
        self._file_cache_key = file_cache_key

    def extract(self) -> list[Document]:
        from langchain.text_splitter import LatexTextSplitter
        ##这样splitter是否对了，能按标题进行切嘛？
        latex_splitter = LatexTextSplitter(chunk_size=1000, chunk_overlap=100)
        with open(self._file_path,'r',encoding='utf-8') as file:
            data = file.read()
            docs = latex_splitter.create_documents([data])
            documents=[]
            for doc in docs:
                documents.append(Document(page_content=doc.page_content))
            return documents
            
        