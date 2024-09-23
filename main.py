import loguru

from dataset_text_sft import test_text_sft_dataset

if __name__ == "__main__":
    loguru.logger.info("dataAgent starting...")
    test_text_sft_dataset()