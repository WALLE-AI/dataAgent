import json

import loguru

from prompt import STARCHAT_QS_ANSWER_GENERATOR_RPROMOPT, STARCHAT_QS_QUESTION_GENERATOR_RPROMOPT
from tqdm import tqdm

from utils.helper import llm_result_postprocess, write_json_file_line

def image_generator_conversation_data(data_dict):
    q_prompt = STARCHAT_QS_QUESTION_GENERATOR_RPROMOPT
    a_prompt = STARCHAT_QS_ANSWER_GENERATOR_RPROMOPT
    response_dict2 = model_execute(data_dict, q_prompt.replace("{content}", data_dict['description']))
    question_dict_list = llm_result_postprocess(response_dict2['content'])
    total_tokens = response_dict2['total_tokens']
    data_dict['model'] = response_dict2['model_name']
    for question in question_dict_list:
        convesation_format_human_dict = {
            "from": "human",
            "value": "{content}\n<image>"
        }
        convesation_format_gpt_dict = {
            "from": "gpt",
            "value": ""
        }
        convesation_format_human_dict["value"] = convesation_format_human_dict["value"].format(
            content=question)
        data_dict['conversations'].clear()
        data_dict['conversations'].append(convesation_format_human_dict)
        response_dict3 = model_execute(data_dict,
                                             a_prompt.format(content1=data_dict['description'], content2=question))
        convesation_format_gpt_dict["value"] = response_dict3['content']
        data_dict['conversations'].append(convesation_format_gpt_dict)
        total_tokens +=response_dict3['total_tokens']
    return data_dict,total_tokens



def image_generator_conversation_index(data_json_file):
    with open(data_json_file, "r", encoding="utf-8") as file:
        data = file.read()
        data = json.loads(data)
        loguru.logger.info(f"data size :{len(data)}")
        all_data_use_total_tokens = 0
        data_dict_list = []
        for _data in tqdm(data[:100]):
            loguru.logger.info(f"accident_label:{_data['accident_label']},description:{_data['description']}")
            data_dict,total_tokens = image_generator_conversation_data(_data)
            data_dict_list.append(data_dict)
            all_data_use_total_tokens += total_tokens
            save_file_name = "data/starvlm_image_qa_" + str(100) + ".json"
            write_json_file_line(data_dict_list, save_file_name)
        loguru.logger.info(f"all_data_use_total_tokens:{all_data_use_total_tokens}")


def execute_image_qa_generator():
    json_file_path = "data\\images_table_format_59973.json"
    image_generator_conversation_index(json_file_path)

