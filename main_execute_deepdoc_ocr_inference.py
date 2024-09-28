

import loguru

from dotenv import load_dotenv

from parser.vision.deepdoc.inference import deepdoc_ocr_pdf_text_extract, test_deepdoc_layout_recognizer_inference
 

load_dotenv()
if __name__ == "__main__":
    loguru.logger.info("dataAgent starting...")
    deepdoc_ocr_pdf_text_extract()