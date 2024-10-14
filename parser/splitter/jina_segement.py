import asyncio
import os
import aiohttp
import loguru
import requests
import json
from dotenv import load_dotenv

load_dotenv()

timeout = aiohttp.ClientTimeout(total=3000)

async def aysnc_jina_segement_api_post_request(content):
    url = os.getenv("JINA_URL")
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer '+os.getenv('JINA_KEY')
    }
    data = {
        "content": content,
        "return_tokens": True,
        "return_chunks": True,
        "max_chunk_length": 1000
    }
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, data=json.dumps(data)) as response:
                if response.status != 200:
                    loguru.logger.info(f"request error code {response.status}")
                response_json = await response.json()
                return response_json
    except Exception as e:
            loguru.logger.info(f"seed request error: {e}")
def async_jian_segements(content):
    return asyncio.run(aysnc_jina_segement_api_post_request(content))

def jina_segement_api_post_request(content):
    url = os.getenv("JINA_URL")
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer '+os.getenv('JINA_KEY')
    }
    data = {
        "content": content,
        "return_tokens": True,
        "return_chunks": True,
        "max_chunk_length": 1000
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        return response.json()  # 返回JSON格式的响应内容
    else:
        return {"error": "Failed to get a successful response", "status_code": response.status_code}
    
if __name__ == "__main__":
    text ='''
    需要注意的是一般垂直领域大模型不会直接让模型生成答案，而是跟先检索相关的知识，然后基于召回的知识进行回答，也就是基于检索增强的生成（Retrieval Augmented Generation:https://www.promptingguide.ai/techniques/rag , RAG)。这种方式能减少模型的幻觉，保证答案的时效性，还能快速干预模型对特定问题的答案。

    所以SFT和RLHF阶段主要要培养模型的三个能力:

    (1) 领域内问题的判别能力，对领域外的问题需要能拒识 (2) 基于召回的知识回答问题的能力 (3) 领域内风格对齐的能力，例如什么问题要简短回答什么问题要翔实回答，以及措辞风格要与领域内的专业人士对齐。

    下面本文将从继续预训练，领域微调数据构建，减少幻觉，知识召回四个方面进行具体的介绍。
    
    '''
    # reponse = jina_segement_api_post_request(text)
    reponse = async_jian_segements(text)
    print(reponse)
    for index,text in enumerate(reponse['chunks']):
        loguru.logger.info(f"chunk {index} response:{text}")