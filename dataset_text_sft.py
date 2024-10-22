##解析抽取手册数据，进行chunk后 对chunk数据进行QA抽取
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import List
import uuid

import loguru
from tqdm import tqdm

from embedding.preprocess.embedding_clustering import EmbeddingCluster
from entities.dataset_sft_entity import DatasetsSwiftTextSFTFormat, DatasetsTextSFTFormat
from entities.document import Document
from models.llm import LLMApi, model_generate_a_document, model_generate_q_document, model_generate_qa_document
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
        
def semaphore_do_work_qa(execute_function,semaphore,thread_name,**kwargs):
    with semaphore:
        loguru.logger.info(f"{thread_name} is working")
        execute_function(**kwargs)
        loguru.logger.info(f"{thread_name} is done")


class TextSFTDatasets():
    def __init__(self, file_path):
        self.file_path = file_path
        
    def extract_text(self,etl_type):
        return ExtractProcessor.extract(self.file_path,etl_type)

    def table_json_data_to_document(self) -> List[Document]:
        docs_list = []
        def extract_info(s):
            split_result = s.split("_")
            return split_result[0],split_result[1]

        with open(self.file_path,"r",encoding="utf-8") as file:
            for line in file:
                '''
                {"image": "《地下防水工程质量验收规范 GB50208-2011》_84_.png", 
                "markdown": "<table><caption>C.2渗漏水检测表C.2.2渗漏水检测工具</caption>\n<tr><td  >名称</td><td  >用途</td></tr>\n<tr><td  >0.5m~1m钢直尺</td><td  >量测混凝土湿渍、渗水范围</td></tr>\n<tr><td  >精度为0.1mm的钢尺</td><td  >量测混凝土裂缝宽度</td></tr>\n<tr><td  >放大镜</td><td  >观测混凝土裂缝</td></tr>\n<tr><td  >有刻度的塑料量筒</td><td  >量测滴水量</td></tr>\n<tr><td  >秒表</td><td  >量测渗漏水滴落速度</td></tr>\n<tr><td  >吸墨纸或报纸</td><td  >检验湿渍与渗水</td></tr>\n<tr><td  >粉笔</td><td  >在混凝土上用粉笔勾画湿渍、渗水范围</td></tr>\n<tr><td  >工作登高扶梯</td><td  >顶板渗漏水、混凝土裂缝检验</td></tr>\n<tr><td  >带有密封缘口的规定尺寸方框量测明显滴漏和连续渗流，根据工程需要可自行设</td><td></td></tr>\n</table>"}
                '''
                data = json.loads(line)
                file_name,page_num = extract_info(data["image"])
                metadata = {
                    "file_name":file_name,
                    "page_num":page_num
                }
                if re.search(r'<\s*table\s*>', data["markdown"]):
                    doc = Document(
                        page_content=data["markdown"],
                        metadata=metadata
                        )
                    docs_list.append(doc)
        return docs_list
                
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
            
    def _execute_theading_func(sef,execute_func,documents: list[Document],llm_type,model_name,reponse_doc_list:List):
        max_threads = 5
        semaphore = threading.Semaphore(max_threads)
        thread_name = 0
        threads = []
        reponse_list = []
        for doc in tqdm(documents):
            document_format_thread = threading.Thread(
               target=semaphore_do_work_qa,
                    kwargs={
                            "execute_function": execute_func,
                            "semaphore":semaphore,
                            "thread_name":thread_name,
                            "doc":doc,
                            "llm_type":llm_type,
                            "model_name":model_name,
                            "reponse_doc_list":reponse_doc_list
                        },
                    )
            thread_name +=1
            threads.append(document_format_thread)
            document_format_thread.start()
        for thread in threads:
            thread.join()
            
            
    def question_document(self,documents: list[Document],llm_type,model_name,reponse_doc_list):
        '''
        llm generator question
        '''
        self._execute_theading_func(self._format_question_document,documents,llm_type,model_name,reponse_doc_list)

            
    def answer_document(self,documents:list[Document],llm_type,model_name,reponse_doc_list):
        '''
         llm or rag generator answer
        '''
        self._execute_theading_func(self._format_answer_document,documents,llm_type,model_name,reponse_doc_list)
        
    
    def _format_answer_document(self,doc:Document,llm_type,model_name,reponse_doc_list:List):
        '''
        llm or rag generator answer
        '''
        if doc.page_content is None or not doc.page_content.strip():
            reponse_doc_list=[]
        try:
            answer,a_tokens = model_generate_a_document(doc.metadata['input'],doc.metadata['context'],llm_type=llm_type,model_name=model_name,document_language="Chinese")
            answer_dict = self._split_extract_qa_with_regex(answer)
            if answer_dict:
                for data in answer_dict:
                    doc.metadata["output"] = data['Answer']
                    doc.metadata["a_llm_client"] = llm_type
                    doc.metadata["a_model_name"] = model_name
                    doc.metadata["total_tokens"] = doc.metadata["total_tokens"]+a_tokens
                    reponse_doc_list.append(doc)
        except Exception as e:
            loguru.logger.exception(e)
                
    def save_sft_dataset_instruction_format(self,docs:List[Document],save_datasets_path:str):
        '''
        instruction:
        input:
        output:
        '''
        with open(save_datasets_path,"w",encoding="utf-8") as file:
            for doc in docs:
                doc.metadata['content'] =doc.page_content 
                file.write(json.dumps(doc.metadata, ensure_ascii=False) + "\n")
        
    def save_sft_dataset_swift_format(self):
        '''
        system:
        query:
        answer:
        '''
        pass 
    
    def _format_question_document(self,doc:Document,llm_type,model_name,reponse_doc_list:List):
        if doc.page_content is None or not doc.page_content.strip():
            reponse_doc_list=[]
        try:
            reponse,tokens = model_generate_q_document(doc.page_content,llm_type=llm_type,model_name=model_name,document_language="Chinese")
            q_reponse = self._split_extract_questions(reponse)
            if q_reponse:
                for question in q_reponse:
                    metadata = {
                       "input": question,
                       "q_llm_client":llm_type,
                       "q_model_name":model_name,
                       "total_tokens":tokens
                    }
                    q_doc = Document(
                        page_content=doc.page_content,
                        metadata=metadata                       
                    )
                    reponse_doc_list.append(q_doc)
        except Exception as e:
            loguru.logger.exception(e)
    def _split_extract_questions(self,text:str):
        questions = []
        lines = text.split('\n')
        for line in lines:
            if line.startswith('Question:'):
                questions.append(line[len('Question: '):])
        return questions
    
    def _split_extract_qa_with_regex(self,text:str):
        # 正则表达式匹配Question和Answer块
        pattern = r'Question: (.*?)\n\nAnswer: (.*?)(?=\n\nQuestion: |\Z)'
        matches = re.findall(pattern, text, re.DOTALL)
        qa_pairs = [{"Question":match[0].strip(), "Answer":match[1].strip()} for match in matches]
        return qa_pairs

# def extract_questions_re(text):
#     # 使用正则表达式匹配以"Question:"开头的句子
#     questions = re.findall(r'Question:\s*(.*?)(?=\nQuestion:|$)', text, re.DOTALL)
#     return [question.strip() for question in questions]


# def test_execute_text_sft_dataset():
#     file_path = "data/《中华人民共和国安全生产法》（2021 年修订版）.pdf"
#     file_path_md = "data/test_readme.md"
#     file_path_md = "data/pdf_markdown/GB50205-2001 钢结构工程施工质量验收规范.md"
#     file_path_tex = "data/《砌体结构工程施工质量验收规范_GB50203-2011》.tex"
#     ##有问题
#     etltype = "localhost"
#     file_path_doc = "data/《起重设备安装工程施工及验收标准》（征求意见稿）.doc"
#     file_name = Path(file_path_md)
#     text_sft_dataset = TextSFTDatasets(file_path_md)
#     all_docs = text_sft_dataset.extract_text(etltype)
#     llm_type="openrouter"
#     model_name="anthropic/claude-3.5-sonnet"
#     q_reponse_list = []
#     loguru.logger.info(f"chunk text {len(all_docs)}")
#     for doc in all_docs[20:30]:
#         loguru.logger.info(f"doc:{doc}")
#         q_reponse = []
#         if len(doc.page_content) >50:
#             reponse,tokens = model_generate_q_document(doc.page_content,llm_type=llm_type,model_name=model_name,document_language="Chinese")
#             q_reponse = extract_questions(reponse)
#             if q_reponse:
#                 for question in q_reponse:
#                     data_dict = DatasetsTextSFTFormat()
#                     answer,a_tokens = model_generate_a_document(question,doc.page_content,llm_type=llm_type,model_name=model_name,document_language="Chinese")
#                     answer_dict = extract_qa_with_regex(answer)
#                     if answer_dict:
#                         for data in answer_dict:
#                             data_dict.input = data['Question']
#                             data_dict.output = data['Answer']
#                         data_dict.llm_client = llm_type
#                         data_dict.model_name = model_name
#                         data_dict.context = doc.page_content
#                         data_dict.total_tokens = tokens +a_tokens
#                         q_reponse_list.append(data_dict.to_dict())
#     loguru.logger.info(f"response:{len(q_reponse_list)}")
#     save_file = "data/handbook_" +file_name.stem +"_.json"
#     with open(save_file,"w",encoding="utf-8") as file:
#         for line in q_reponse_list:
#             file.write(json.dumps(line, ensure_ascii=False) + "\n")
     
def table_to_generator_question():
    "抽取deepdoc中table信息"
    tables_images_save = "datasets/tables_images_save/"
    json_files = get_directory_all_json_files(tables_images_save)
    llm_type="localhost"
    model_name="Qwen2.5-72B-Instruct-AWQ"
    for file in tqdm(json_files):
        file_path = Path(file)
        save_file = "data/table_data_sft/handbook_table_sft_" +file_path.stem +"_.json"
        if not os.path.exists(save_file):
            loguru.logger.info(f"tex_file_name:{save_file}")
            text_sft_dataset = TextSFTDatasets(file)
            docs = text_sft_dataset.table_json_data_to_document()
            loguru.logger.info(f"docs: {len(docs)}")
            reponse_question_list = []
            text_sft_dataset.question_document(docs,llm_type,model_name,reponse_question_list)
            loguru.logger.info(f"reponse_question_list: {len(reponse_question_list)}")
            text_sft_dataset.save_sft_dataset_instruction_format(reponse_question_list,save_file)
        else:
            loguru.logger.info(f"{save_file} file is exist")
            
def table_to_generator_answer():
    tables_images_save = "datasets/tables_images_save/"
    json_files = get_directory_all_json_files(tables_images_save)
    pass


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
                swift_dataset=DatasetsSwiftTextSFTFormat(
                    system=data['instruction'],
                    query=data['input'],
                    response=data["output"]
                )
                all_data_list.append(swift_dataset.to_dict())
    write_json_file_line(all_data_list,"data/handbook_sft/handbook_dataset_sft_all_swift.json")
    loguru.logger.info(f"all data list size {len(all_data_list)}") 
    
    
def execute_text_sft_dataset_preproces():
    #聚类去重
    dataset_file = "data/handbook_sft/handbook_dataset_sft.json"
    save_file = "data/handbook_sft/handbook_dataset_embedding.csv"
    dataset_df = EmbeddingCluster.kmeans_embedding_cluster(dataset_file,save_file)
    loguru.logger.info(f"dataset df {len(dataset_df)}")


