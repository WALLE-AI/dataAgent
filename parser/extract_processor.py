from enum import Enum
from pathlib import Path
from entities.document import Document
from parser.markdown_extractor import MarkdownExtractor
from parser.pdf_extractor import PdfExtractor
from parser.unstructured.unstructured_markdown_extractor import UnstructuredMarkdownExtractor


class EtlType(Enum):
    ETL_TYPE="Unstructured"
    DEFAULT_ETL_TYPE="default"
    
def _is_file_format(file_path):
    input_file = Path(file_path)
    file_extension = input_file.suffix.lower()
    return file_extension


class ExtractProcessor:
    @classmethod
    def load_from_upload_file(file_path):
        pass
    
    @classmethod
    def extract(
        cls, file_path: str = None
    ) -> list[Document]:
        file_extension = _is_file_format(file_path)
        if EtlType.ETL_TYPE == "Unstructured":
            if file_extension in {".md", ".markdown"}:
                extractor = UnstructuredMarkdownExtractor(file_path, "unstructured_api_url")
        else:
            if file_extension == ".pdf":
                extractor = PdfExtractor(file_path)
            elif file_extension in {".md", ".markdown"}:
                extractor = MarkdownExtractor(file_path, autodetect_encoding=True)
        return extractor.extract()