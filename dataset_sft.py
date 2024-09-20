##根据文本数据合成SFT数据
import json


class ImageSFTDatasets():
    def __init__(self, file_path):
        self.desc = "image sft dataset"
        self.file_path = file_path

    def __str__(self):
        return self.desc

    @classmethod
    def image_sft_conversation_format_dataset(cls):
        '''
        id:
        image:
        conversation
        '''
        file_path = cls().file_path
        with open(file_path, 'r', encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
