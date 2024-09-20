##根据文本数据合成SFT数据
import json

from entities.image_entity import ImagesConversationData
from utils.helper import write_json_file_line
from tqdm import tqdm

class ImageSFTDatasets():
    def __init__(self):
        self.desc = "image sft dataset"

    def __str__(self):
        return self.desc

    @classmethod
    def image_sft_conversation_format_dataset(cls,file_path):
        '''
        id:
        image:
        conversation
        '''
        sft_dataset_list = []
        with open(file_path, 'r', encoding="utf-8") as file:
            for line in tqdm(file):
                data = json.loads(line)
                image_id = data['image_id'].split(".")[0]
                image_dict = ImagesConversationData(
                    id=image_id,
                    image=data['image_id'],
                    image_oss_url=data['image_oss_url'],
                    conversations=data['conversations']
                )
                sft_dataset_list.append(image_dict.to_dict())
        return sft_dataset_list
    
    @classmethod
    def write_to_file(cls, data_list, file_path):
        write_json_file_line(data_list, file_path)
        
        
def execute_image_sft_conversation():
    file_path = "data/starvlm_image_qa_100.json"
    image_sft_datasets = ImageSFTDatasets.image_sft_conversation_format_dataset(file_path)
    ImageSFTDatasets.write_to_file(image_sft_datasets, "data/starvlm_image_sft_100.json")
                
