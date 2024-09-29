

import loguru

from dotenv import load_dotenv

from parser.vision.mineru.cmd_mineru import execute_miner_cmd

 

load_dotenv()
if __name__ == "__main__":
    loguru.logger.info("dataAgent starting...")
    execute_miner_cmd()