import loguru

from dotenv import load_dotenv

from rag.index_process import test_markdon_file_rag, test_markdown_file_embedding

load_dotenv()
if __name__ == "__main__":
    loguru.logger.info("rag starting...")
    test_markdown_file_embedding()
    