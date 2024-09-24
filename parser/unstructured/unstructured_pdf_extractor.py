import logging
import os

from entities.document import Document
from parser.extractor_base import BaseExtractor


logger = logging.getLogger(__name__)


class UnstructuredPdfExtractor(BaseExtractor):
    """Loader that uses unstructured to load word documents."""

    def __init__(
        self,
        file_path: str,
        api_url: str,
    ):
        """Initialize with file path."""
        self._file_path = file_path
        self._api_url = api_url
        
    def extract(self) -> list[Document]:
        from unstructured.partition.pdf import partition_pdf
        elements = partition_pdf(self._file_path)
        from unstructured.chunking.title import chunk_by_title

        chunks = chunk_by_title(elements, max_characters=2000, combine_text_under_n_chars=2000)
        documents = []
        for chunk in chunks:
            text = chunk.text.strip()
            documents.append(Document(page_content=text))

        return documents
        
        
    