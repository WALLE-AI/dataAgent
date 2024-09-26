import loguru

from dotenv import load_dotenv

from parser.vision.gotocr2.inference import execute_gotocr2_model
load_dotenv()
if __name__ == "__main__":
    loguru.logger.info("dataAgent starting...")
    execute_gotocr2_model()