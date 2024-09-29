
from typing import Generator
import loguru
import openai
import os
import threading
import httpx

from entities.image_entity import ImageVlmModelOutPut
from parser.tokenizers.gpt2_tokenzier import GPT2Tokenizer
from prompt import GENERATOR_QA_PROMPT_ZH, LATEXT_TO_MARKDOWN_PROMPT, PROMPT_TEST
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
        "openai/gpt-4o-mini-2024-07-18":"openai/gpt-4o-mini-2024-07-18",
        "qwen/qwen-2-vl-72b-instruct":"qwen/qwen-2-vl-72b-instruct",
        "qwen/qwen-2-vl-7b-instruct":"qwen/qwen-2-vl-7b-instruct"
    },
    "siliconflow":{
        "Qwen/Qwen2-7B-Instruct":"Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen2-72B-Instruct":"Qwen/Qwen2-72B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct":"Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct":"Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct":"Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct":"Qwen/Qwen2.5-14B-Instruct",
        
    },
    "localhost":{
        "intern_vl":"intern_vl",
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
            base_url = os.getenv("VLM_SERVE_HOST")
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
    def build_prompt(cls,query,system_prompt=None,search=False):
        if search:
            search_result = ddg_search_text(query)
            if search_result:
                context = ";".join([text["snippet"] for text in search_result])
                loguru.logger.info(f"search context:{context}")
        #这里可以加上ddsg api接口
        if system_prompt:
            prompt = [{"role":"system","content":system_prompt},
                  {"role": "user", "content": query}]
        else:
            prompt = [{"role": "user", "content": query}]
        return prompt
    
    
    @classmethod
    def messages_stream_generator(cls,response):
        message_content = ""
        for text in response:
            ##finsh_reason获取usage内容
            if text.choices[0].finish_reason == "stop":
                usage_info_dict={}
                if text.usage:
                    usage_info_dict = text.usage.to_dict()
                else:
                    ##total tokens
                    usage_info_dict['total_tokens'] = cls._get_num_tokens_by_gpt2(message_content)
                if text.choices[0].delta.content:
                    message_content += text.choices[0].delta.content
                response_dict = ImageVlmModelOutPut(
                model_name=text.model,
                content=message_content,
                total_tokens=usage_info_dict['total_tokens']
                )
                return response_dict.to_dict()
            else:
                if text.choices[0].delta.content:
                    message_content += text.choices[0].delta.content
        ##如果没有就返回为默认
        response_dict = ImageVlmModelOutPut()
        return response_dict.to_dict()
    @classmethod
    def _get_num_tokens_by_gpt2(self, text: str) -> int:
        """
        Get number of tokens for given prompt messages by gpt2
        Some provider models do not provide an interface for obtaining the number of tokens.
        Here, the gpt2 tokenizer is used to calculate the number of tokens.
        This method can be executed offline, and the gpt2 tokenizer has been cached in the project.

        :param text: plain text of prompt. You need to convert the original message to plain text
        :return: number of tokens 实际tokens计算有点误差
        """
        return GPT2Tokenizer.get_num_tokens(text)
    
    
    @classmethod
    def call_llm(cls,prompt,stream=True,llm_type="siliconflow",model_name="Qwen/Qwen2.5-7B-Instruct"):
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
        if stream:
            return cls.messages_stream_generator(llm_response)
        else:
            response_dict = ImageVlmModelOutPut(
                model_name=llm_response.model,
                content=llm_response.choices[0].message.content,
                total_tokens=llm_response.usage.total_tokens
            )
            return response_dict.to_dict()
    

    @classmethod    
    def get_client(cls,llm_type):
        return cls().llm_client(llm_type)
    
def model_generate_latex_to_markdown(query):
    ##TODO 这里prompt要更改一下
    prompt = LATEXT_TO_MARKDOWN_PROMPT.replace("{latex_content}",query)
    prompt = LLMApi.build_prompt(prompt)
    response = LLMApi.call_llm(prompt)
    answer = response["content"]
    response['total_tokens'] = LLMApi._get_num_tokens_by_gpt2(query)+response['total_tokens']
    return answer.strip(),response['total_tokens']
    
def model_generate_qa_document(query, document_language: str):
    ##TODO 这里prompt要更改一下
    sytem_prompt = GENERATOR_QA_PROMPT_ZH.format(language=document_language)
    # prompt = GENERATOR_QA_PROMPT_ZH_2.replace("{{document}}",query)
    prompt = LLMApi.build_prompt(query,sytem_prompt)
    response = LLMApi.call_llm(prompt)
    answer = response["content"]
    response['total_tokens'] = LLMApi._get_num_tokens_by_gpt2(query +" "+ sytem_prompt)+response['total_tokens']
    return answer.strip(),response['total_tokens']

def model_image_table_format_execute(data_dict, prompt,llm_type,model_name):
    build_prompt = LLMApi.build_image_prompt(prompt,data_dict['image_oss_url'])
    llm_result_dict = LLMApi.call_llm(build_prompt,llm_type=llm_type,model_name=model_name)
    llm_result_dict['prompt']=prompt
    llm_result_dict['total_tokens'] =LLMApi._get_num_tokens_by_gpt2(prompt)+llm_result_dict['total_tokens']
    return llm_result_dict

