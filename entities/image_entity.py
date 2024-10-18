from decimal import Decimal
from typing import List

from pydantic import BaseModel

from utils.encoder import jsonable_encoder


class ImageVlmModelOutPut(BaseModel):
    prompt: str = "你是一个有用的助手"
    model_name: str = "dsdsad"
    content: str = "shdhshadhsahd"
    total_tokens: int = 122343

    def to_dict(self) -> dict:
        return jsonable_encoder(self)


class ImagesConversationData(BaseModel):
    id: str = "b16f009e-030d-4028-8866-45385891d97d"
    image: str = "b16f009e-030d-4028-8866-45385891d97d.jpg"
    image_oss_url: str = "https://zhgd-prod-oss.oss-cn-shenzhen.aliyuncs.com/aeda0121-2d75-2c97-437d-01fc59f55d25.jpg"
    conversations: List[dict] = [{}]
    
    def to_dict(self) -> dict:
        return jsonable_encoder(self)


class ImageVlmModelOutPut(BaseModel):
    prompt: str = "你是一个有用的助手"
    model_name:str="default model"
    content: str = "None"
    total_tokens: int = 0
    def to_dict(self) -> dict:
        return jsonable_encoder(self)


class ImageVlmQualityLabel(BaseModel):
    image_id: str = ""
    image_oss_url: str
    description: str = ""
    prompt: str = "你是一个有用的助手"
    total_tokens: str = "122343"
    score: float = 0.0
    human_review: str = ""

    def to_dict(self) -> dict:
        return jsonable_encoder(self)


class ImageVLMTestOutPut(BaseModel):
    """
    Model class for provider quota configuration.
    """
    model_name: str = "01-ai/yi-vision"
    provider: str = "openrouter"
    image_name: str = ""
    description: str = ""
    prompt: str = "你是一个有用的助手"
    response: str = "shdhshadhsahd"
    total_tokens: int = 122343
    total_price: Decimal = 0.01
    currency: str = "USD"
    score: float = 0.0
    human_review: str = ""

    def to_dict(self) -> dict:
        return jsonable_encoder(self)


class ImageTableProcess(BaseModel):
    """
    Model class for provider quota configuration.
        "id" 图片id
        "name":"02.png", 图片名称id
        "correct_name":"correct_02.jpg", 整改图片名称id
        "position":"", 隐患位置
        "accident_label":"",隐患类别
        "description":"", 隐患描述
        "correct_description":"",整改隐患描述
        "risk":"",隐患风险分析
        type: str = "quality" 质量或者安全safe
        ##整改依据
        "correct_basic":"针对导向管未加U形筋的问题，应参照《建筑施工扣件式钢管脚手架安全技术规范》（JGJ 130 - 2011），在楼板上表面配置受力钢筋，以确保结构的稳定性和安全性。同时，结合其他相关规范和标准，如《安徽省建设项目施工现场安全风险管控标准》和《建筑施工安全生产隐患识别图集》，进一步强化施工过程中的细节管理和风险控制，确保施工质量和安全",
        "label":"0" 隐患的严重程度
        "conversations": [{"from": "human","value": "What do you see happening in this image?\n<image>"},{"from": "gpt","value": "The scene depicts a lively plaza area with several people walking and enjoying their time. A man is standing in the plaza with his legs crossed, holding a kite in his hand. The kite has multiple sections attached to it, spread out in various directions as if ready for flight.\n\nNumerous people are scattered throughout the plaza, walking and interacting with others. Some of these individuals are carrying handbags, and others have backpacks. The image captures the casual, social atmosphere of a bustling plaza on a nice day."}]
    """
    id: str = "0"
    image_id: str = "",
    image_correct_id: str = "",
    image_oss_url: str = ""
    conversations: List[dict] = [{}],
    position: str = "B2段地下室外墙",
    accident_label: str = "主体结构",
    description: str = "防水基层不平整",
    correct_description: str = "",
    risk: str = "",
    type: str = "quality"
    correct_basic: str = "",
    label: str = "0"

    def to_dict(self) -> dict:
        return jsonable_encoder(self)

