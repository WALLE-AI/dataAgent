import loguru

from dataset_text_sft import execute_text_sft_dataset
from dotenv import load_dotenv

from parser.vision.deepdoc.inference import test_deepdoc_layout_recognizer_inference_trs

load_dotenv()
if __name__ == "__main__":
    loguru.logger.info("dataAgent starting...")
    execute_text_sft_dataset()