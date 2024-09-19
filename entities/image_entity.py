
from pydantic import BaseModel

from utils.encoder import jsonable_encoder
class ImageVlmModelOutPut(BaseModel):
    prompt: str = "你是一个有用的助手"
    model_name:str="dsdsad"
    content: str = "shdhshadhsahd"
    total_tokens: int = 122343
    def to_dict(self) -> dict:
        return jsonable_encoder(self)
