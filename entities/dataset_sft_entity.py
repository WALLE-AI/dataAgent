from pydantic import BaseModel

from utils.encoder import jsonable_encoder


class DatasetsLatextToMarkdonwPage(BaseModel):
    file_name:str=""
    latex_content:str=""
    markdown_content:str=""
    page_num:int=0
    page_total_tokens:int=0
    llm_client:str=""
    model_name:str=""
    
    def to_dict(self) -> dict:
        return jsonable_encoder(self)
    


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