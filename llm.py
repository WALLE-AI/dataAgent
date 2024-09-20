
import loguru
import openai
import os
import threading
import httpx

from entities.image_entity import ImageVlmModelOutPut
from prompt import PROMPT_TEST
from utils.helper import ddg_search_text

MODEL_NAME_LIST = {
    "openai":{
    "gpt-3.5-turbo":"gpt-3.5-turbo",
    "gpt-3.5-turbo-16k":"gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-1106": "gpt-3.5-turbo-1106",
    "gpt-4": "gpt-4",
    "gpt-4-32k":"gpt-4-32k",
    "gpt-4-1106-preview": "gpt-4-1106-preview",
    "gpt-4-vision-preview": "gpt-4-vision-preview",
    },
    "openrouter":{
        "meta-llama/llama-3-8b-instruct:free":"meta-llama/llama-3-8b-instruct:free",
        "microsoft/phi-3-medium-4k-instruct":"microsoft/phi-3-medium-4k-instruct",
        "meta-llama/llama-3-70b-instruct":"meta-llama/llama-3-70b-instruct",
        "mistralai/mistral-7b-instruct":"mistralai/mistral-7b-instruct",
        "openai/gpt-4o":"openai/gpt-4o",
        "openai/gpt-4o-mini-2024-07-18":"openai/gpt-4o-mini-2024-07-18"
    },
    "siliconflow":{
        "Qwen/Qwen2-7B-Instruct":"Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen2-72B-Instruct":"Qwen/Qwen2-72B-Instruct"
    }
    
}


class LLMApi():
    def __init__(self) -> None:
        self.des = "llm api service"
    def __str__(self) -> str:
        return self.des
    
    def init_client_config(self,llm_type):
        if llm_type == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
            api_key  = os.environ.get("OPENROUTER_API_KEY")
        elif llm_type =='siliconflow':
            base_url = "https://api.siliconflow.cn/v1"
            api_key = os.environ.get("SILICONFLOW_API_KEY")
        elif llm_type=="openai":
            base_url = "https://api.openai.com/v1"
            api_key  = os.environ.get("OPENAI_API_KEY")
        else:
            base_url = os.getenv("VLM_SERVE_HOST") +":9005/v1"
            api_key  = "empty"
        return base_url,api_key     
    @classmethod
    def llm_client(cls,llm_type):
        base_url,api_key = cls().init_client_config(llm_type)
        thread_local = threading.local()
        try:
            return thread_local.client
        except AttributeError:
            thread_local.client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key,
                # We will set the connect timeout to be 10 seconds, and read/write
                # timeout to be 120 seconds, in case the inference server is
                # overloaded.
                timeout=httpx.Timeout(connect=10, read=120, write=120, pool=10),
            )
            return thread_local.client
    @classmethod
    def build_image_prompt(cls,query,image_base64):
        user_content = [
                {"type": "text",
                 "text": query},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_base64,
                    },
                }
        ]
        prompt = [{"role": "user", "content": user_content}]
        return prompt
    
    @classmethod
    def build_prompt(cls,query,search=False):
        context = ""
        if search:
            search_result = ddg_search_text(query)
            if search_result:
                context = ";".join([text["snippet"] for text in search_result])
                loguru.logger.info(f"search context:{context}")
        #这里可以加上ddsg api接口
        prompt = [{"role":"system","content":PROMPT_TEST.format(context=context)},
                  {"role": "user", "content": query}]
        return prompt
    
    
    @classmethod
    def call_llm(cls,prompt,stream=False,llm_type="siliconflow",model_name="Qwen/Qwen2-7B-Instruct"):
        '''
        默认选择siliconflow qwen2-72B的模型来
        '''
        llm_response = cls.get_client(llm_type=llm_type).chat.completions.create(
                model=MODEL_NAME_LIST[llm_type][model_name],
                messages=prompt,
                max_tokens=1024,
                stream=stream,
                temperature=0.2,
            )
        response_dict = ImageVlmModelOutPut(
            model_name=llm_response.model,
            content=llm_response.choices[0].message.content,
            total_tokens=llm_response.usage.total_tokens
        )
        return response_dict.to_dict()
    
    @classmethod    
    def get_client(cls,llm_type):
        return cls().llm_client(llm_type)


def model_image_table_format_execute(data_dict, prompt):
     build_prompt = LLMApi.build_image_prompt(prompt,data_dict['image_oss_url'])
     llm_result_dict = LLMApi.call_llm(build_prompt,llm_type="openrouter",model_name="openai/gpt-4o-mini-2024-07-18")
     llm_result_dict['prompt']=prompt
     return llm_result_dict

