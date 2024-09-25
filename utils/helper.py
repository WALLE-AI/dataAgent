
import json
from hashlib import sha256
import time

from PIL import Image

import base64
from io import BytesIO
from itertools import islice

import loguru
import requests

"""Document loader helpers."""

import concurrent.futures
from pathlib import Path
from typing import NamedTuple, Optional, cast

def pdf_file_image(pdf_file,zoomin=3):
    '''
    将pdf全部转成image
    '''
    import pdfplumber
    pdf = pdfplumber.open(pdf_file)
    images = [p.to_image(resolution=72 * zoomin).annotated for i, p in
                            enumerate(pdf.pages)]
    return images


class FileEncoding(NamedTuple):
    """A file encoding as the NamedTuple."""

    encoding: Optional[str]
    """The encoding of the file."""
    confidence: float
    """The confidence of the encoding."""
    language: Optional[str]
    """The language of the file."""


def detect_file_encodings(file_path: str, timeout: int = 5) -> list[FileEncoding]:
    """Try to detect the file encoding.

    Returns a list of `FileEncoding` tuples with the detected encodings ordered
    by confidence.

    Args:
        file_path: The path to the file to detect the encoding for.
        timeout: The timeout in seconds for the encoding detection.
    """
    import chardet

    def read_and_detect(file_path: str) -> list[dict]:
        rawdata = Path(file_path).read_bytes()
        return cast(list[dict], chardet.detect_all(rawdata))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(read_and_detect, file_path)
        try:
            encodings = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Timeout reached while detecting encoding for {file_path}")

    if all(encoding["encoding"] is None for encoding in encodings):
        raise RuntimeError(f"Could not detect encoding for {file_path}")
    return [FileEncoding(**enc) for enc in encodings if enc["encoding"] is not None]




def generate_text_hash(text: str) -> str:
    hash_text = str(text) + "None"
    return sha256(hash_text.encode()).hexdigest()

def download_image(url, filename):
    root_image = "data/sample10000_image/"
    images_dir_path = root_image + filename
    # loguru.logger.info(f"images dir path {images_dir_path}")
    if filename in load_images_from_folder(root_image):
        loguru.logger.info(f"image is exist,no donwload")
        return False
    else:
        try:
            response = requests.get(url)
            response.raise_for_status()  # 检查请求是否成功
            with open(images_dir_path, 'wb') as f:
                f.write(response.content)
                loguru.logger.info(f"image save：{filename}")
                return True
        except requests.RequestException as e:
            loguru.logger.info(f"request error：{e}")
            return False
        except IOError as e:
            loguru.logger.info(f"request io error：{e}")



def load_images_from_folder(folder_path):
    images_list = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            images_list.append(filename)
    return images_list


def image_to_base64(image_path,root_path):
    root_path =root_path
    images_path_new = root_path + image_path
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
        mime_type = image_path.split(".")[-1]
        with Image.open(images_path_new) as img:
            # 定义新的尺寸，例如缩小到原来的一半
            new_width = img.width // 2
            new_height = img.height // 2
            # 调整图片大小
            img_resized = img.resize((new_width, new_height))
            # 将图片转换为字节流
            buffered = BytesIO()
            img_resized.save(buffered, format=img.format)
            # 将字节流转换为Base64编码
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f'data:image/{mime_type};base64,{img_base64}'
    
def encode_image_base64_from_url(image_id, image_url):
    mime_type = image_id.split(".")[-1]
    try:
        # 发送GET请求获取图片内容
        response = requests.get(image_url)
        response.raise_for_status()  # 如果请求失败，这会抛出异常
        # 获取图片内容
        image_content = response.content
        # 将图片内容转换为base64编码
        base64_encoded = base64.b64encode(image_content).decode('utf-8')
        base64_encoded = f'data:image/{mime_type};base64,{base64_encoded}'
        return base64_encoded
    except requests.RequestException as e:
        print(f"download image error: {e}")
        return None
    except Exception as e:
        print(f"transformer process error: {e}")
        return None
    
def write_json_file_line(data_dict, save_file_name):
    with open(save_file_name, "w", encoding="utf-8") as file:
        for line in data_dict:
            file.write(json.dumps(line, ensure_ascii=False)+"\n")
         
         
def write_json_file(data_dict, save_file_name):
    jsonn_str_data = json.dumps(data_dict, ensure_ascii=False)
    with open(save_file_name, "w", encoding="utf-8") as file:
        loguru.logger.info(f"save json file {save_file_name}")
        file.write(jsonn_str_data)   
            
def llm_result_postprocess(llm_response_content):
    ##json的后处理
    from json_repair import repair_json
    json_string = repair_json(llm_response_content, return_objects=True)
    return json_string


class MeasureExecutionTime:
    """
    装饰器类，用于测量另一个函数的执行时间，并统计总时间
    """
    total_time = 0 

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        start_time = time.time()  # 开始时间
        result = self.func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 结束时间
        execution_time = end_time - start_time  # 计算执行时间
        self.total_time += execution_time  # 累加到总时间
        loguru.logger.info(f"function {self.func.__name__} execute time：{execution_time:.6f} 秒")
        loguru.logger.info(f"function {self.func.__name__} total execute time：{self.total_time:.6f} 秒")
        return result
    @classmethod
    def get_total_time(cls):
        """sssssssss
        返回函数调用的总执行时间
        """
        return cls.total_time


def single_measure_execution_time(func):
    """
    装饰器函数，用于测量另一个函数的执行时间,调用一次时间计算
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 开始时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 结束时间
        execution_time = end_time - start_time  # 计算执行时间
        loguru.logger.info(f"function {func.__name__} execute time：{execution_time:.6f} 秒")
        return result,execution_time
    return wrapper


def ddg_search_text(query:str, max_results=5):
    from duckduckgo_search import DDGS
    search_results = []
    reference_results = []
    with DDGS() as ddgs:
        ddgs_gen = ddgs.text(query, backend="lite")
        for r in islice(ddgs_gen, max_results):
            search_results.append(r)
    for idx, result in enumerate(search_results):
        loguru.logger.debug(f"搜索结果{idx + 1}：{result}")
        ##[result["body"], result["href"]]
        reference_results.append({
                "name": result["title"],
                "url": result["href"],
                "snippet": result["body"]
        })
    return reference_results
