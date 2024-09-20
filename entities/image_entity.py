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
