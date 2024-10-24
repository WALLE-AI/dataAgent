##写一个简单的页面完成如下json格式形成：
import base64
from io import BytesIO
import json
from pathlib import Path
import re
import uuid
import loguru
from pydantic import BaseModel

from models.llm import LLMApi
from utils.encoder import jsonable_encoder
from utils.helper import llm_result_postprocess


def image_to_base64(img):
    buffered = BytesIO()
    mime_type = "png"
    img.save(buffered, format=mime_type)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f'data:image/{mime_type};base64,{img_base64}'

PROMPT_IMAGE_EXTRACR='''
你是一个智能助手，能够高质量抽取图片中的信息，请根据用户输入图片，抽取信息输出如下格式：
{
    "隐患问题":xxx,
    "风险分析":xxx
    "整改措施":xxx
    "规范要求":xxxx
}
'''

def extract_image_text(image):
    llm_type = "openrouter"
    model_name = "qwen/qwen-2-vl-72b-instruct"
    image_base64 = image_to_base64(image)
    build_prompt = LLMApi.build_image_prompt(PROMPT_IMAGE_EXTRACR,image_base64)
    llm_result_dict = LLMApi.call_llm(build_prompt,llm_type=llm_type,model_name=model_name)
    response_dict = llm_result_postprocess(llm_result_dict['content'])
    loguru.logger.info(f"response_dict:{response_dict}")
    return response_dict
    
class BenchMarkVLMData(BaseModel):
    id:str="01.png"
    file_name:str=""
    label:str="基坑施工"
    risk_problem:str=""
    risk_analysis:str=""
    corrective_method:str=""
    regulations:str=""
    def to_dict(self) -> dict:
        return jsonable_encoder(self)
    
def extract_risk_info(text):
    # 定义正则表达式模式
    pattern = re.compile(
        r'(隐患问题：)(.*?)\n(风险分析：)(.*?)\n(整改措施：)(.*?)\n(规范要求：)(.*?)(?=\n\n|$)',
        re.DOTALL
    )
    # 搜索匹配项
    match = pattern.search(text)
    if match:
        # 提取匹配的组
        hazard_issue = match.group(2).strip()
        risk_analysis = match.group(4).strip()
        correction_measures = match.group(6).strip()
        specification_requirements = match.group(8).strip()
        # 返回提取的信息
        return {
            "隐患问题": hazard_issue,
            "风险分析": risk_analysis,
            "整改措施": correction_measures,
            "规范要求": specification_requirements
        }
    else:
        return None



    
data_json_list = []
file_paths = []
def extract_pdf_info(file_paths):
    import pdfplumber
    for pdf_path in file_paths:
        file_name = Path(pdf_path).stem
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                data_dict = extract_risk_info(page_text)
                if data_dict:
                    ##抽取图片
                    for img in page.images:
                        bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                        cropped_page = page.crop(bbox)
                        crop_image = cropped_page.to_image()
                        image_id = str(uuid.uuid4()) +".png"
                        image_path = "benchmark/images/"+image_id
                        crop_image.save(image_path)
                        
                        data = BenchMarkVLMData(
                            id=image_id,
                            file_name = file_name,
                            label = "label",
                            risk_problem = data_dict['隐患问题'],
                            risk_analysis = data_dict['风险分析'],
                            corrective_method = data_dict['整改措施'],
                            regulations=data_dict['规范要求']
                        )
                        data_json_list.append(data.to_dict())
        with open("benchmark/benchmark_vlm_eval_data.json","w",encoding="utf-8") as file:
            for line in data_json_list:
                file.write(json.dumps(line, ensure_ascii=False) + "\n")
                    
def build_json_data(image,extract_image,file_name,label):
    if extract_image:
        result = extract_image_text(extract_image)
    image_path = "benchmark/images/"+str(uuid.uuid4()) +".png"
    image_id = str(uuid.uuid4()) +".png"
    image.save(image_path)
    data = BenchMarkVLMData(
        id=image_id,
        file_name = file_name,
        label = label,
        risk_problem = result['隐患问题'],
        risk_analysis = result['风险分析'],
        corrective_method = result['整改措施'],
        regulations=result['规范要求']
    )
    data_json_list.append(data.to_dict())
    with open("benchmark/benchmark_vlm_eval_data.json","w",encoding="utf-8") as file:
        for line in data_json_list:
              file.write(json.dumps(line, ensure_ascii=False) + "\n")
    return data


def upload_file(files):
    file_paths = [file.name for file in files]
    loguru.logger.info(f"Uploading file name {file_paths}")
    extract_pdf_info(file_paths)
    return file_paths
    
import gradio as gr
with gr.Blocks()as benchmark_gradio:
    gr.Markdown("# Benchmark Image data to json")
    with gr.Row(equal_height=True):
        button = gr.Button("Generate", variant="primary")
    # with gr.Row():
    #     risk_image = gr.Image(label="隐患风险图片",type="pil")
    #     extract_text_image = gr.Image(label="抽取隐患类型的图片",type="pil")
    with gr.Row(equal_height=True):
        upload_button = gr.UploadButton("Upload a File", file_types=[".pdf",".doc"], file_count="multiple")
        upload_button.upload(
            fn=upload_file, 
            inputs= upload_button,
            outputs=gr.File()
            )
        

    # button.click(
    #     fn=build_json_data,
    #     inputs=[risk_image,extract_text_image],
    #     outputs=[gr.Textbox(label="show data result", lines=3)]
    # )
    # button.click(
    #     fn=extract_pdf_info,
    #     inputs=file_paths,
    #     outputs=[gr.Textbox(label="show data result", lines=3)]
    #     )
    
    

