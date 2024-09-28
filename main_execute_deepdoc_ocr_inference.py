

import loguru

from dotenv import load_dotenv

from parser.vision.deepdoc.inference import test_deepdoc_layout_recognizer_inference
 

load_dotenv()
if __name__ == "__main__":
    loguru.logger.info("dataAgent starting...")
    test_deepdoc_layout_recognizer_inference()