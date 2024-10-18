from pydantic import BaseModel

from utils.encoder import jsonable_encoder


class DatasetsTextSFTFormat(BaseModel):
    instruction:str=""
    input:str=""
    output:str=""
    context:str=""
    total_tokens:int=0
    llm_client:str=""
    model_name:str=""

    def to_dict(self) -> dict:
        return jsonable_encoder(self)
    
class DatasetsSwiftTextSFTFormat(BaseModel):
    system:str=""
    query:str=""
    response:str=""
    def to_dict(self) -> dict:
        return jsonable_encoder(self)