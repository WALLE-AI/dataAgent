##解析抽取手册数据，进行chunk后 对chunk数据进行QA抽取
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

import loguru
from tqdm import tqdm

from entities.dataset_sft_entity import DatasetsTextSFTFormat
from entities.document import Document
from models.llm import LLMApi, model_generate_qa_document
from parser.cleaner.clean_processor import CleanProcessor
from parser.extract_processor import EtlType, ExtractProcessor
from parser.markdown_extractor import MarkdownExtractor
from parser.pdf_extractor import PdfExtractor
from parser.splitter.fixed_text_splitter import FixedRecursiveCharacterTextSplitter
from parser.vision.utils.utils import get_directory_all_pdf_files, get_directory_all_tex_files
from prompt.prompt import GENERATOR_QA_PROMPT_EN, GENERATOR_QA_PROMPT_ZH, GENERATOR_QA_PROMPT_ZH_1, GENERATOR_QA_PROMPT_ZH_2
from utils.helper import generate_text_hash, get_directory_all_json_files, write_json_file_line
from dotenv import load_dotenv

load_dotenv()

def semaphore_do_work(execute_function,semaphore,thread_name,document_node,all_qa_documents,total_tokens_num,document_language):
    with semaphore:
        loguru.logger.info(f"{thread_name} is working")
        execute_function(document_node,all_qa_documents,total_tokens_num,document_language)
        loguru.logger.info(f"{thread_name} is done")


class TextSFTDatasets():
    def __init__(self, file_path):
        self.file_path = file_path
        
    def extract_text(self,etl_type):
        return ExtractProcessor.extract(self.file_path,etl_type)
    
    def chunk_text_to_qa_unstructured(self, documents: list[Document], **kwargs) -> list[Document]:
        all_qa_documents = []
        total_tokens_num = []
        max_threads = 5
        semaphore = threading.Semaphore(max_threads)
        thread_name = 0
        threads = []
        for document in tqdm(documents[10:]):
            # document clean
            document_text = CleanProcessor.clean(document.page_content, True)
            document.page_content = document_text
            document_format_thread = threading.Thread(
                        target=semaphore_do_work,
                        kwargs={
                            "execute_function": self._format_qa_document,
                            "semaphore":semaphore,
                            "thread_name":thread_name,
                            "document_node": document,
                            "all_qa_documents": all_qa_documents,
                            "total_tokens_num":total_tokens_num,
                            "document_language": "Chinese",

                        },
            )
            thread_name +=1
            threads.append(document_format_thread)
            document_format_thread.start()
        for thread in threads:
            thread.join()
        loguru.logger.info(f"total_tokens_num:{sum(total_tokens_num)}")
        return all_qa_documents

    def chunk_text_to_qa(self, documents: list[Document], **kwargs) -> list[Document]:
        splitter = self._get_splitter()
        all_documents = []
        all_qa_documents = []
        for document in documents:
            # document clean
            document_text = CleanProcessor.clean(document.page_content, True)
            document.page_content = document_text
            # parse document to nodes
            document_nodes = splitter.split_documents([document])
            split_documents = []
            for document_node in document_nodes:
                if document_node.page_content.strip():
                    doc_id = str(uuid.uuid4())
                    hash = generate_text_hash(document_node.page_content)
                    document_node.metadata["doc_id"] = doc_id
                    document_node.metadata["doc_hash"] = hash
                    # delete Splitter character
                    page_content = document_node.page_content
                    if page_content.startswith(".") or page_content.startswith("。"):
                        page_content = page_content[1:]
                    else:
                        page_content = page_content
                    document_node.page_content = page_content
                    split_documents.append(document_node)
            all_documents.extend(split_documents)
            for i in range(0, len(all_documents), 10):
                threads = []
                sub_documents = all_documents[i: i + 10]
                for doc in sub_documents:
                    document_format_thread = threading.Thread(
                        target=self._format_qa_document,
                        kwargs={
                            "document_node": doc,
                            "all_qa_documents": all_qa_documents,
                            "document_language": kwargs.get("doc_language", "Chinese"),
                        },
                    )
                    threads.append(document_format_thread)
                    document_format_thread.start()
                for thread in threads:
                    thread.join()
        return all_documents

    def _get_splitter(self):
        separator = "\n"
        character_splitter = FixedRecursiveCharacterTextSplitter.from_encoder(
            chunk_size=100,
            chunk_overlap=10,
            fixed_separator=separator,
            separators=["\n\n", "。", ". ", " ", ""],
            embedding_model_instance=None,
        )
        return character_splitter


    def _format_qa_document(self, document_node, all_qa_documents, total_tokens_num,document_language):
        format_documents = []
        if document_node.page_content is None or not document_node.page_content.strip():
            return
        try:
            # qa model document
            response,total_tokens = model_generate_qa_document(document_node.page_content, document_language)
            total_tokens_num.append(total_tokens)
            document_qa_list = self._format_split_text(response)
            loguru.logger.info(f"document_qa_list:{document_qa_list}")
            qa_documents = []
            for result in document_qa_list:
                qa_document = Document(page_content=result["question"], metadata=document_node.metadata.copy())
                doc_id = str(uuid.uuid4())
                hash = generate_text_hash(result["question"])
                qa_document.metadata["answer"] = result["answer"]
                qa_document.metadata['context']=document_node.page_content
                qa_document.metadata["doc_id"] = doc_id
                qa_document.metadata["doc_hash"] = hash
                qa_documents.append(qa_document)
            format_documents.extend(qa_documents)
        except Exception as e:
            loguru.logger.exception(e)

        all_qa_documents.extend(format_documents)

    def _format_split_text(self, text):
        regex = r"Q\d+:\s*(.*?)\s*A\d+:\s*([\s\S]*?)(?=Q\d+:|$)"
        matches = re.findall(regex, text, re.UNICODE)

        return [{"question": q, "answer": re.sub(r"\n\s*", "\n", a.strip())} for q, a in matches if q and a]

    def build_sft_format(self,all_qa_documents,handbook_name,save_sft_datasets):
        handbook_name = "《"+handbook_name+"》"
        instruction = '''使用{name}知识内容回答建筑专业性问题'''.format(name=handbook_name)
        sft_data_list = []
        for document in all_qa_documents:
            data = DatasetsTextSFTFormat(
                instruction=instruction,
                input=document.page_content,
                output=document.metadata["answer"],
                context = document.metadata["context"]
            )
            sft_data_list.append(data.to_dict())
        if sft_data_list:
            write_json_file_line(sft_data_list,save_sft_datasets)


# def execute_text_sft_dataset():
#     file_path = "data/《中华人民共和国安全生产法》（2021 年修订版）.pdf"
#     file_path_md = "data/test_readme.md"
#     file_path_md = "data/handbook_test.md"
#     file_path_tex = "data/《砌体结构工程施工质量验收规范_GB50203-2011》.tex"
#     ##有问题
#     file_path_doc = "data/《起重设备安装工程施工及验收标准》（征求意见稿）.doc"
#     file_name = Path(file_path_tex)
#     text_sft_dataset = TextSFTDatasets(file_path_tex)
#     all_docs = text_sft_dataset.extract_text()
#     loguru.logger.info(f"chunk text {len(all_docs)}")
#     all_qa_documents = text_sft_dataset.chunk_text_to_qa_unstructured(all_docs)
#     text_sft_dataset.build_sft_format(all_qa_documents,file_name.stem)

def execute_text_sft_dataset():
    tex_files = os.getenv("PDF_DIR_ROOT")
    all_tex_path = get_directory_all_pdf_files(tex_files)
    index = 0
    etl_type = "Vision"
    for text_file_path in all_tex_path:
        text_sft_dataset = TextSFTDatasets(text_file_path)
        tex_file_name = Path(text_file_path).stem
        save_sft_datasets = "data/handbook_sft/handbook_dataset_sft_"+tex_file_name+".json"
        if not os.path.exists(save_sft_datasets):
            loguru.logger.info(f"tex_file_name:{tex_file_name}")
            start_time = time.time()  # 开始时间
            all_docs = text_sft_dataset.extract_text(etl_type)
            end_time = time.time()  # 结束时间
            execution_time = end_time - start_time  # 计算执行时间
            loguru.logger.info(f"execution_time:{execution_time}")
            loguru.logger.info(f"{tex_file_name},chunk text {len(all_docs)}")
            all_qa_documents = text_sft_dataset.chunk_text_to_qa_unstructured(all_docs)
            text_sft_dataset.build_sft_format(all_qa_documents,tex_file_name,save_sft_datasets)
        else:
            index +=1
            loguru.logger.info(f"{save_sft_datasets} file is exist")
def execute_text_sft_datatsets_merge():
    json_dir = "data/handbook_sft/"
    json_files = get_directory_all_json_files(json_dir)
    all_data_list = []
    for file in json_files:
        with open(file,"r",encoding="utf-8") as file:
            for line in file:
                data  = json.loads(line)
                all_data_list.append(data)
    write_json_file_line(all_data_list,"data/handbook_sft/handbook_dataset_sft_all.json")
    loguru.logger.info(f"all data list {all_data_list}")  


