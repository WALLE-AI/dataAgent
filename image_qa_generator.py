import json
import random
import threading
import timeit

import loguru

from llm import model_image_table_format_execute
from prompt import STARCHAT_QS_ANSWER_GENERATOR_RPROMOPT, STARCHAT_QS_QUESTION_GENERATOR_RPROMOPT
from tqdm import tqdm

from utils.helper import MeasureExecutionTime, llm_result_postprocess, write_json_file_line

@MeasureExecutionTime
def image_generator_conversation_data(data_dict,data_dict_list,tokens_list):
    q_prompt = STARCHAT_QS_QUESTION_GENERATOR_RPROMOPT
    a_prompt = STARCHAT_QS_ANSWER_GENERATOR_RPROMOPT
    response_dict2 = model_image_table_format_execute(data_dict, q_prompt.replace("{content}", data_dict['description']))
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
        if "question" in question:
            convesation_format_human_dict["value"] = convesation_format_human_dict["value"].format(
                content=question['question'])
        else:
            convesation_format_human_dict["value"] = convesation_format_human_dict["value"].format(
                content="")
        data_dict['conversations'].clear()
        data_dict['conversations'].append(convesation_format_human_dict)
        response_dict3 = model_image_table_format_execute(data_dict,
                                             a_prompt.format(content1=data_dict['description'], content2=question))
        convesation_format_gpt_dict["value"] = response_dict3['content']
        data_dict['conversations'].append(convesation_format_gpt_dict)
        total_tokens +=response_dict3['total_tokens']
    data_dict_list.append(data_dict)
    tokens_list.append(total_tokens)



def semaphore_do_work(data_dict,data_dict_list,tokens_list,semaphore, thread_name):
    with semaphore:
        loguru.logger.info(f"{thread_name} is working")
        image_generator_conversation_data(data_dict,data_dict_list,tokens_list)
        loguru.logger.info(f"{thread_name} is done")



def image_generator_conversation_index(data_json_file):
    with open(data_json_file, "r", encoding="utf-8") as file:
        data = file.read()
        ##打散数据
        # random.shuffle(data)
        data = json.loads(data)
        loguru.logger.info(f"data size :{len(data)}")
        all_data_use_total_tokens_list=[]
        data_dict_list = []
        threads=[]
        max_threads = 10
        semaphore = threading.Semaphore(max_threads)
        thread_name = 0
        for _data in tqdm(data[:100]):
            loguru.logger.info(f"accident_label:{_data['accident_label']},description:{_data['description']}")
            # document_format_thread = threading.Thread(
            #             target=image_generator_conversation_data,
            #             kwargs={
            #                 "data_dict": _data,
            #                 "data_dict_list": data_dict_list,
            #                 "tokens_list":all_data_use_total_tokens_list
            #             }
            #         )
            document_format_thread = threading.Thread(
                        target=semaphore_do_work,
                        kwargs={
                            "data_dict": _data,
                            "data_dict_list": data_dict_list,
                            "tokens_list":all_data_use_total_tokens_list,
                            "semaphore":semaphore,
                            "thread_name":thread_name
                        }
                    )
            ##执行线程
            thread_name+=1
            threads.append(document_format_thread)
            document_format_thread.start()
            # image_generator_conversation_data(_data,data_dict_list,all_data_use_total_tokens_list)
        for thread in threads:
            thread.join()
        save_file_name = "data/images_randow_sample_label_quality_" + str(81819) + ".json"
        write_json_file_line(data_dict_list, save_file_name)
        loguru.logger.info(f"all_data_use_total_tokens:{sum(all_data_use_total_tokens_list)}")


def execute_image_qa_generator():
    json_file_path = "data\\images_randow_sample_label_quality_10000_81819.json"
    image_generator_conversation_index(json_file_path)

