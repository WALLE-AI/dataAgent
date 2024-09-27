import argparse
from pathlib import Path
import threading
import loguru
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

from parser.vision.utils.conversation import SeparatorStyle,conv_templates
from parser.vision.utils.utils import disable_torch_init, get_directory_all_pdf_files
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from parser.vision.gotocr2.model import *
from parser.vision.utils.utils import KeywordsStoppingCriteria

from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO
from parser.vision.gotocr2.model.plug.blip_process import BlipImageEvalProcessor

from transformers import TextStreamer
import re
import string

from utils.helper import MeasureExecutionTime, pdf_file_image

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


 
# translation_table = str.maketrans(punctuation_dict)


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def init_model(model_path):
    disable_torch_init()
    model_name = os.path.expanduser(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = GOTQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=151643).eval()
    return model,tokenizer

@MeasureExecutionTime
def inference_model(model,tokenizer,image,type):
    # Model

    model.to(device='cuda',  dtype=torch.bfloat16)


    # TODO vary old codes, NEED del 
    image_processor = BlipImageEvalProcessor(image_size=1024)

    image_processor_high =  BlipImageEvalProcessor(image_size=1024)

    use_im_start_end = True

    image_token_len = 256

    w, h = image.size
    # print(image.size)
    
    if type == 'format':
        qs = 'OCR with format: '
    else:
        qs = 'OCR: '

    #对图片中bbox进行识别
    # if args.box:
    #     bbox = eval(args.box)
    #     if len(bbox) == 2:
    #         bbox[0] = int(bbox[0]/w*1000)
    #         bbox[1] = int(bbox[1]/h*1000)
    #     if len(bbox) == 4:
    #         bbox[0] = int(bbox[0]/w*1000)
    #         bbox[1] = int(bbox[1]/h*1000)
    #         bbox[2] = int(bbox[2]/w*1000)
    #         bbox[3] = int(bbox[3]/h*1000)
    #     if args.type == 'format':
    #         qs = str(bbox) + ' ' + 'OCR with format: '
    #     else:
    #         qs = str(bbox) + ' ' + 'OCR: '

    # if args.color:
    #     if args.type == 'format':
    #         qs = '[' + args.color + ']' + ' ' + 'OCR with format: '
    #     else:
    #         qs = '[' + args.color + ']' + ' ' + 'OCR: '

    if use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs 
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs



    conv_mode = "mpt"
    # args.conv_mode = conv_mode

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # print(prompt)


    inputs = tokenizer([prompt])


    # vary old codes, no use
    image_1 = image.copy()
    image_tensor = image_processor(image)


    image_tensor_1 = image_processor_high(image_1)


    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids,
            images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
            do_sample=False,
            num_beams = 1,
            no_repeat_ngram_size = 20,
            # streamer=streamer,
            max_new_tokens=4096,
            stopping_criteria=[stopping_criteria]
            )
        
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()    
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs
        # if args.render:
        #     print('==============rendering===============')

        #     outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            
        #     if outputs.endswith(stop_str):
        #         outputs = outputs[:-len(stop_str)]
        #     outputs = outputs.strip()

        #     if '**kern' in outputs:
        #         import verovio
        #         from cairosvg import svg2png
        #         import cv2
        #         import numpy as np
        #         tk = verovio.toolkit()
        #         tk.loadData(outputs)
        #         tk.setOptions({"pageWidth": 2100, "footer": 'none',
        #        'barLineWidth': 0.5, 'beamMaxSlope': 15,
        #        'staffLineWidth': 0.2, 'spacingStaff': 6})
        #         tk.getPageCount()
        #         svg = tk.renderToSVG()
        #         svg = svg.replace("overflow=\"inherit\"", "overflow=\"visible\"")

        #         svg_to_html(svg, "./results/demo.html")

        #     if args.type == 'format' and '**kern' not in outputs:

                
        #         if  '\\begin{tikzpicture}' not in outputs:
        #             html_path = "./render_tools/" + "/content-mmd-to-html.html"
        #             html_path_2 = "./results/demo.html"
        #             right_num = outputs.count('\\right')
        #             left_num = outputs.count('\left')

        #             if right_num != left_num:
        #                 outputs = outputs.replace('\left(', '(').replace('\\right)', ')').replace('\left[', '[').replace('\\right]', ']').replace('\left{', '{').replace('\\right}', '}').replace('\left|', '|').replace('\\right|', '|').replace('\left.', '.').replace('\\right.', '.')


        #             outputs = outputs.replace('"', '``').replace('$', '')

        #             outputs_list = outputs.split('\n')
        #             gt= ''
        #             for out in outputs_list:
        #                 gt +=  '"' + out.replace('\\', '\\\\') + r'\n' + '"' + '+' + '\n' 
                    
        #             gt = gt[:-2]

        #             with open(html_path, 'r') as web_f:
        #                 lines = web_f.read()
        #                 lines = lines.split("const text =")
        #                 new_web = lines[0] + 'const text ='  + gt  + lines[1]
        #         else:
        #             html_path = "./render_tools/" + "/tikz.html"
        #             html_path_2 = "./results/demo.html"
        #             outputs = outputs.translate(translation_table)
        #             outputs_list = outputs.split('\n')
        #             gt= ''
        #             for out in outputs_list:
        #                 if out:
        #                     if '\\begin{tikzpicture}' not in out and '\\end{tikzpicture}' not in out:
        #                         while out[-1] == ' ':
        #                             out = out[:-1]
        #                             if out is None:
        #                                 break
    
        #                         if out:
        #                             if out[-1] != ';':
        #                                 gt += out[:-1] + ';\n'
        #                             else:
        #                                 gt += out + '\n'
        #                     else:
        #                         gt += out + '\n'


        #             with open(html_path, 'r') as web_f:
        #                 lines = web_f.read()
        #                 lines = lines.split("const text =")
        #                 new_web = lines[0] + gt + lines[1]

        #         with open(html_path_2, 'w') as web_f_new:
        #             web_f_new.write(new_web)


    
def write_tex_file(content,root_path,file_name,page_num=None):
    if page_num:
        save_file_path = root_path + "/" +file_name+"_"+str(page_num)+".tex"
    else:
        save_file_path = root_path + "/" +file_name +".tex"
    with open(save_file_path,"w",encoding="utf-8") as file:
        file.write(content)


def semaphore_do_work(execute_function,semaphore,thread_name,model,tokenizer,image,type_ocr):
    with semaphore:
        loguru.logger.info(f"{thread_name} is working")
        execute_function(model,tokenizer,image,type_ocr)
        loguru.logger.info(f"{thread_name} is done")

         
def execute_gotocr2_model(pdf_file):
    model_name = os.getenv("MODEL_PATH_GOT")
    # pdf_file = "data/《砌体结构工程施工质量验收规范_GB50203-2011》.pdf"
    # image = load_image(image_file)
    pdf_file_path = Path(pdf_file)
    pdf_image = pdf_file_image(pdf_file)
    model,tokenizer = init_model(model_name)
    content_list = []
    type_ocr = "format"
    # #无法实现多线程推理
    # max_threads = 1
    # semaphore = threading.Semaphore(max_threads)
    # thread_name = 0
    # threads = []
    for image in tqdm(pdf_image):
        content = inference_model(model,tokenizer,image,type_ocr)
        content_list.append(content)
    #     document_format_thread = threading.Thread(
    #                     target=semaphore_do_work,
    #                     kwargs={
    #                         "execute_function": inference_model,
    #                         "semaphore":semaphore,
    #                         "thread_name":thread_name,
    #                         "model": model,
    #                         "tokenizer": tokenizer,
    #                         "image":image,
    #                         "type_ocr": type_ocr,
    #                     },
    #         )
    #     thread_name +=1
    #     threads.append(document_format_thread)
    #     document_format_thread.start()
    # for thread in threads:
    #     thread.join()
    content = "\n".join(content_list)
    write_tex_file(content,"data/pdf_latex",pdf_file_path.stem)
    

def execute_all_pdf_latex_preprocess():
    pdf_dir_path =os.getenv("PDF_DIR_ROOT")
    all_pdf_files = get_directory_all_pdf_files(pdf_dir_path)
    for pdf_file in all_pdf_files:
        loguru.logger.info(f"pdf file: {pdf_file}")
        execute_gotocr2_model(pdf_file)
    
