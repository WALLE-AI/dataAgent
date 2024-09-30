from enum import Enum
from pathlib import Path
from entities.document import Document
from parser.latex_extractor import LatexExtractor
from parser.markdown_extractor import MarkdownExtractor
from parser.pdf_extractor import PdfExtractor
from parser.unstructured.unstructured_doc_extractor import UnstructuredWordExtractor
from parser.unstructured.unstructured_markdown_extractor import UnstructuredMarkdownExtractor
from parser.unstructured.unstructured_pdf_extractor import UnstructuredPdfExtractor
from parser.vision.pdf_extractor_vision import PdfVisionExtractor


class EtlType(Enum):
    ETL_TYPE="Unstructured"
    DEFAULT_ETL_TYPE="default"
    ETL_VISION_TYPE="Vision"
    
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
        cls, file_path: str = None,etl_type=None,
    ) -> list[Document]:
        file_extension = _is_file_format(file_path)
        if EtlType.ETL_TYPE.value == etl_type:
            #TODO：测试API的效果
            if file_extension in {".md", ".markdown"}:
                extractor = UnstructuredMarkdownExtractor(file_path, "unstructured_api_url")
            elif file_extension in {".docx",".doc"}:
                extractor = UnstructuredWordExtractor(file_path, "unstructured_api_url")
            elif file_extension == ".pdf":
                extractor = UnstructuredPdfExtractor(file_path,"unstructured_api_url")
        if EtlType.ETL_VISION_TYPE.value == etl_type:
            if file_extension == ".pdf":
                extractor = PdfVisionExtractor(file_path)
            elif file_extension == ".tex":
                extractor = LatexExtractor(file_path)
        else:
            if file_extension == ".pdf":
                extractor = PdfExtractor(file_path)
            elif file_extension in {".md", ".markdown"}:
                extractor = MarkdownExtractor(file_path, autodetect_encoding=True)
        return extractor.extract()