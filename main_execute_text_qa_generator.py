import loguru
from dotenv import load_dotenv

from dataset_text_sft import markdown_to_generator_question, table_to_generator_question


load_dotenv()
if __name__ == "__main__":
    loguru.logger.info("dataAgent starting...")
    markdown_to_generator_question()