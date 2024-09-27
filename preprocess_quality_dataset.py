
import json
import re
import loguru
from matplotlib.font_manager import FontProperties
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from entities.image_entity import ImageTableProcess
from llm import LLMApi
from prompt import QUALITY_MAIN_STRUCTURE_PROMOPT_LABEL, QUALITY_MAIN_STRUCTURE_PROMOPT_LABEL_UPDATE, QUALITY_MAIN_STRUCTURE_RISK_LABEL, QUALITY_MAIN_STRUCTURE_RISK_PROMPT
from utils.helper import download_image, load_images_from_folder, write_json_file
font = FontProperties(fname=r'/home/dataset-s3-0/gaojing/llm/easy-rag/data/SimHei.ttf')


def plot_quality_data(data,save_image_name):
    plt.rcParams["font.sans-serif"] = "SimHei"  # 设置中文字体为黑体
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    fig, ax = plt.subplots(figsize=(15, 10))
    data.plot(kind='bar',ax=ax)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005),
                ha='center', va='bottom')
    plt.title('Category Distribution')
    plt.xlabel('Category')
    plt.ylabel('Count')
    # 保存图表到本地文件
    plt.show()
    plt.savefig('data/' + save_image_name+".png")
    

def read_sample_quality_total_data_10000():
    '''
    获取随机抽取整个质量库中隐患类别中前top10中每个类别的10000条
    '''
    import polars as pl
    data = pl.read_csv("data/randow_sample_label_quality_10000.csv")
    loguru.logger.info(f"total quality data size:{len(data)}")
    data = data.to_pandas()
    data_save_list = []
    for index, row_data in tqdm(data.iterrows(),total=data.shape[0]):
        data_dict = row_data.to_dict()
        image_name = data_dict['照片'].split("https://zhgd-prod-oss.oss-cn-shenzhen.aliyuncs.com/")[-1]
        ##download image dir
        if image_name in load_images_from_folder("data/sample10000_image"):
            loguru.logger.info(f"image is exist,no donwload")
            data_entity = ImageTableProcess(
                    id=str(index),
                    image_id=image_name,
                    image_correct_id="",
                    image_oss_url=data_dict['照片'],
                    conversations=[{}],
                    position=data_dict["隐患部位"] if isinstance(data_dict['隐患部位'], str) else "",
                    accident_label=data_dict['name'] if isinstance(data_dict['name'], str) else "",
                    description=data_dict['隐患内容'],
                    correct_description="",
                    type=data_dict['类型'],
                    risk="",
                    correct_basic="",
                    label="0"
                )
            data_save_list.append(data_entity.to_dict())
            continue
        ##下载图片
        is_download = download_image(data_dict['照片'], image_name)
        if is_download:
            data_entity = ImageTableProcess(
                    id=str(index),
                    image_id=image_name,
                    image_correct_id="",
                    image_oss_url=data_dict['照片'],
                    conversations=[{}],
                    position=data_dict["隐患部位"] if isinstance(data_dict['隐患部位'], str) else "",
                    accident_label=data_dict['name'] if isinstance(data_dict['name'], str) else "",
                    description=data_dict['隐患内容'],
                    correct_description="",
                    type=data_dict['类型'],
                    risk="",
                    correct_basic="",
                    label="0"
                )
            data_save_list.append(data_entity.to_dict())
    loguru.logger.info(f"data_save_list:{len(data_save_list)}")
    save_file_name = "data/images_randow_sample_label_quality_10000_" + str(len(data_save_list)) + ".json"
    write_json_file(data_save_list, save_file_name)

    
def exetract_sample_quality_total_data_10000():
    '''
    随机抽取整个质量库中隐患类别中前top10中每个类别的10000条
    '''
    import polars as pl
    data = pl.read_csv("data/image_table_quality_total_data.csv")
    loguru.logger.info(f"total quality data size:{len(data)}")
    data = data.to_pandas()
    result = data['name'].value_counts().head(10)
    result_df = result.reset_index()
    random_samples = pd.DataFrame()
    for name in result_df['name'].values:
        loguru.logger.info(f"column name {name}")
        sampled_data = data[data['name'] == name].sample(n=10000, replace=True)
        random_samples = pd.concat([random_samples, sampled_data], axis=0)
    random_samples.to_csv("data/randow_sample_label_quality_10000.csv",index=False)
    ##验证一下数据是否为随机类别
    random_result = random_samples['name'].value_counts()
    loguru.logger.info(f"random_result:{random_result}")
    
def analysize_quality_total_data(show_plot=None):
    import polars as pl
    data = pl.read_csv("data/image_table_quality_total_data.csv")
    loguru.logger.info(f"total quality data size:{len(data)}")
    data = data.to_pandas()
    result = data['name'].value_counts().head(10)
    if show_plot:
        plot_quality_data(result,"quality_total_label.png")

def analysize_quality_data(show_plot=None):
    '''
    分析一下质量隐患实例中的数据分布,统计一下各个类别的数据总量，plot画一张图片，使用superset吧
    '''
    import polars as pl
    raw_file_list = ["images_table_05.xlsx","images_table_01.xlsx","images_table_02.xlsx","images_table_03.xlsx","images_table_04.xlsx"]
    data_quality_total = 0
    total_data_list =[]
    for raw_file_path in raw_file_list:
        file_path = "/home/dataset-s3-0/gaojing/datasets/images_table/" +raw_file_path
        loguru.logger.info(f"start preprocess file:{file_path}")
        data = pl.read_excel(file_path, sheet_name="Sheet1")
        data = data.to_pandas()
        data = data[data["类型"]=="质量"]
        total_data_list.append(data)
        data_quality_total += len(data)
        ##获取top10的数据
        result = data['name'].value_counts().head(10)
        if show_plot:
            plot_quality_data(result,raw_file_path)
    total_data_df = pd.concat(total_data_list,ignore_index=True)
    total_data_df.to_csv("data/image_table_quality_total_data.csv",index=False)
    loguru.logger.info(f"total quality data size:{len(total_data_df)}")
    loguru.logger.info(f"quality data size:{data_quality_total}")


def read_xlsx_file_to_save_quality_main_structure_json(file_path):
    import polars as pl
    data = pl.read_excel(file_path, sheet_name="Sheet1")
    data = data.to_pandas()
    loguru.logger.info(f"data size :{len(data)}")
    data_save_list = []
    for index, row_data in tqdm(data.iterrows(),total=data.shape[0]):
        data_dict = row_data.to_dict()
        if data_dict['类型'] == "质量":
            type_label = data_dict['name'] if isinstance(data_dict['name'], str) else None
            if type_label=="主体结构":
                image_name = data_dict['照片'].split("https://zhgd-prod-oss.oss-cn-shenzhen.aliyuncs.com/")[-1]
                ##download image dir
                data_entity = ImageTableProcess(
                    id=str(index),
                    image_id=image_name,
                    image_correct_id="",
                    image_oss_url=data_dict['照片'],
                    conversations=[{}],
                    position=data_dict["隐患部位"] if isinstance(data_dict['隐患部位'], str) else "",
                    accident_label=data_dict['name'] if isinstance(data_dict['name'], str) else "",
                    description=data_dict['隐患内容'],
                    correct_description="",
                    type=data_dict['类型'],
                    risk="",
                    correct_basic="",
                    label="0"
                )
                data_save_list.append(data_entity.to_dict())
    loguru.logger.info(f"data_save_list:{len(data_save_list)}")
    save_file_name = "data/images_table_quality_main_structure" + str(len(data_save_list)) + ".json"
    write_json_file(data_save_list, save_file_name)
    
    
def execute_analysize_quality_data():
    # raw_file_list = ["images_table_05.xlsx","images_table_01.xlsx","images_table_02.xlsx","images_table_03.xlsx","images_table_04.xlsx"]
    # for raw_file_path in raw_file_list:
    #     file_path = "/home/dataset-s3-0/gaojing/datasets/images_table/" +raw_file_path
    #     loguru.logger.info(f"start preprocess file:{file_path}")
    exetract_sample_quality_total_data_10000()
    
def excute_preproce_sample_quality_total_data_10000():
    read_sample_quality_total_data_10000()
    
    
def execute_image_table_quality_main_structure():
    raw_file_list = ["images_table_01.xlsx","images_table_02.xlsx","images_table_03.xlsx","images_table_04.xlsx"]
    for raw_file_path in raw_file_list:
        file_path = "/home/dataset-s3-0/gaojing/datasets/images_table/" +raw_file_path
        loguru.logger.info(f"start preprocess file:{file_path}")
        read_xlsx_file_to_save_quality_main_structure_json(file_path)
        
def execute_image_table_quality_main_structure_json():
    raw_file_list_json = ["images_table_quality_main_structure236999.json","images_table_quality_main_structure237214.json",
                     "images_table_quality_main_structure237357.json","images_table_quality_main_structure237373.json"]
    for file_path in raw_file_list_json:
        file_json_path = "data/" +file_path
        loguru.logger.info(f"start preprocess file:{file_json_path}")
        read_quality_main_structure_json(file_json_path)
        
def read_quality_main_structure_json(data_json_file):
    '''
    抽取主体结构的数据 
    data:{'id': '1', 'image_id': 'afaa009c-68bc-4028-88df-41ef396b4f8f.jpg', 'image_correct_id': '', 
    'image_oss_url': 'https://zhgd-prod-oss.oss-cn-shenzhen.aliyuncs.com/afaa009c-68bc-4028-88df-41ef396b4f8f.jpg',
    'conversations': [{}], 'position': '教学楼1区4层', 'accident_label': '主体结构', 'description': '洞口加强筋',
    'correct_description': '', 'risk': '', 'type': '质量', 'correct_basic': '', 'label': '0'}
    :return:
    '''
    from json_repair import repair_json
    with open(data_json_file, "r", encoding="utf-8") as file:
        data = file.read()
        data = json.loads(data)
        loguru.logger.info(f"data size :{len(data)}")
        output_message_list = []
        user_total_tokens = 0
        for _data in tqdm(data):
            ##增强字段 
            import pdb
            pdb.set_trace()
            loguru.logger.info(f"data:{_data}")
            
def execute_analysis_main_structure_data():
    file_path = "data/images_randow_sample_label_quality_10000_81819.json"
    analyze_main_structure_data(file_path)
            
def analyze_main_structure_data(data_json_file):
    '''
    抽取主体结构的类似的数据
    :return:
    '''
    from json_repair import repair_json
    with open(data_json_file, "r", encoding="utf-8") as file:
        data = file.read()
        data = json.loads(data)
        loguru.logger.info(f"data size :{len(data)}")
        output_message_list = []
        user_total_tokens = 0
        for _data in data[:10]:
            ##判断隐患类别中是否存在
            if _data['accident_label'] == "主体结构":
                # prompt = QUALITY_MAIN_STRUCTURE_PROMOPT_LABEL.replace("{content}",_data["description"])
                prompt = QUALITY_MAIN_STRUCTURE_PROMOPT_LABEL_UPDATE.replace("{content}",_data["description"])
                prompt = LLMApi.build_prompt(prompt)
                response = LLMApi.call_llm(prompt,llm_type="openrouter",model_name="qwen/qwen-2-vl-7b-instruct")
                json_content = repair_json(response['content'], return_objects = True)
                json_content_list=["",""]
                if isinstance(json_content,dict):
                    json_content_list =[text for text in json_content.values()]
                loguru.logger.info(f"reponse:{json_content}")
                risk_part = ";".join(json_content_list)
                risk_prompt = QUALITY_MAIN_STRUCTURE_RISK_PROMPT.format(risk_des=_data["description"],risk_part=risk_part)
                risk_prompt = LLMApi.build_prompt(risk_prompt)
                risk_response = LLMApi.call_llm(risk_prompt,llm_type="openrouter",model_name="qwen/qwen-2-vl-7b-instruct")
                output_message={
                    "图片地址":_data["image_oss_url"],
                    "隐患描述":_data["description"],
                    "一级隐患类别":_data['accident_label'],
                    "二级隐患类别":json_content_list[0],
                    "三级隐患类别":json_content_list[1],
                    "total_tokens":response['total_tokens'],
                    "llm_content":risk_response['content']
                }
                output_message_list.append(output_message)
                user_total_tokens += response['total_tokens']
    loguru.logger.info(f"user_total_tokens:{user_total_tokens}")
    file_name_path = "data/" + "quality_10000_result_cluster_200.csv"
    data = pd.DataFrame(output_message_list)
    data.to_csv(file_name_path, index=False, encoding='GBK')