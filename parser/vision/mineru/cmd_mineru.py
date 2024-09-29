import os
from pathlib import Path
import subprocess

import loguru

from parser.vision.utils.utils import get_directory_all_pdf_files


def mineru_convert_pdf_to_md(file_path,file_output_path):    
    # 构建命令
    command = [
        "magic-pdf",
        "-p", file_path,
        "-o", file_output_path,
        "-m", "auto"
    ]
    
    # 执行命令
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        loguru.logger.info(f"mineru command execute successful")
        loguru.logger.info(f"result info {result.stdout}")
    except subprocess.CalledProcessError as e:
        loguru.logger.info(f"mineru command execute failture")
        loguru.logger.info(f"result info {e.stderr}")
        
def execute_miner_cmd():
    pdf_dir_path =os.getenv("PDF_DIR_ROOT")
    all_pdf_files = get_directory_all_pdf_files(pdf_dir_path)
    index=0
    for pdf_file in all_pdf_files:
        pdf_name = Path(pdf_file)
        loguru.logger.info(f"pdf file: {pdf_file}")
        mineru_convert_pdf_to_md(pdf_file,pdf_name.stem)
        if index == 0:
            break
    