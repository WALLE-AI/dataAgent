import loguru

from image_qa_generator import execute_image_qa_generator, test_execute_image_pdf_to_markdown
from dotenv import load_dotenv

load_dotenv()
if __name__ == "__main__":
    loguru.logger.info("dataAgent starting...")
    test_execute_image_pdf_to_markdown()