import loguru

from dotenv import load_dotenv

from parser.vision.gotocr2.inference import execute_all_pdf_latex_preprocess, execute_latex_to_markdown_llm, execute_markdown_file_merge
load_dotenv()
if __name__ == "__main__":
    loguru.logger.info("dataAgent starting...")
    execute_latex_to_markdown_llm()