import loguru
from models.embedding import EmbeddingModel
from parser.splitter.late_chunk.contextual_retrieval_embedder import late_main_context

from dotenv import load_dotenv
load_dotenv()
if __name__ == "__main__":
    # late_main_context()
    text = "hello world"
    text_list =["sdadsadsa","hello world"]
    embedding_result = EmbeddingModel.get_embedding("fastembed").embed_documents(text_list)
    loguru.logger.info(f"embedding {embedding_result}")