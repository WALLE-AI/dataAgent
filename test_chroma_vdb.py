from dotenv import load_dotenv
import loguru

from models.embedding import EmbeddingApi
load_dotenv()

if __name__ == "__main__":
    loguru.logger.info("dataAgent starting...")
    ##test embedding
    test = "hello world"
    embedding_response = EmbeddingApi.embed_query(test)
    loguru.logger.info(f"embedding:{embedding_response}")
    