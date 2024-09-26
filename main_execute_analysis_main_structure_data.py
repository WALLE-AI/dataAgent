import loguru

from dotenv import load_dotenv
from preprocess_quality_dataset import execute_analysis_main_structure_data
load_dotenv()
if __name__ == "__main__":
    loguru.logger.info("dataAgent starting...")
    execute_analysis_main_structure_data()