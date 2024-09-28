import loguru

from dotenv import load_dotenv

from parser.vision.gotocr2.inference import execute_all_pdf_latex_preprocess
load_dotenv()
if __name__ == "__main__":
    loguru.logger.info("dataAgent starting...")
    execute_all_pdf_latex_preprocess()