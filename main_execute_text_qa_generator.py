import loguru

from dataset_text_sft import test_execute_text_sft_dataset
from dotenv import load_dotenv

load_dotenv()
if __name__ == "__main__":
    loguru.logger.info("dataAgent starting...")
    test_execute_text_sft_dataset()