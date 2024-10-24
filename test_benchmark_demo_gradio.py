import loguru
from benchmark.upload_image_gradio import benchmark_gradio

from dotenv import load_dotenv
load_dotenv()
if __name__ == "__main__":
    loguru.logger.info("rag starting...")
    benchmark_gradio.launch(share=False)
    

    