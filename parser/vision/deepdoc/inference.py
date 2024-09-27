import os
from pathlib import Path
import loguru
import numpy as np
from parser.vision.deepdoc.ocr import OCR
from parser.vision.deepdoc.seeit import draw_box
from parser.vision.utils.utils import get_directory_all_pdf_files
from utils.helper import pdf_file_image


def deepdoc_ocr_inference():
    pdf_dir_path = ""
    ocr = OCR()
    pdf_dir_path =os.getenv("PDF_DIR_ROOT")
    all_pdf_files = get_directory_all_pdf_files(pdf_dir_path)
    for pdf_file in all_pdf_files:
        pdf_file_path = Path(pdf_file)
        pdf_image = pdf_file_image(pdf_file)
        loguru.logger.info(f"pdf file: {pdf_file}")
        for i, img in enumerate(pdf_image):
            bxs = ocr(np.array(img))
            bxs = [(line[0], line[1][0]) for line in bxs]
            bxs = [{
                "text": t,
                "bbox": [b[0][0], b[0][1], b[1][0], b[-1][1]],
                "type": "ocr",
                "score": 1} for b, t in bxs if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]]
            img = draw_box(pdf_image[i], bxs, ["ocr"], 1.)
            save_name ="data/deepdoc_test/"+pdf_file_path.stem+"_"+str(i)+".png"
            img.save(save_name, quality=95)
            with open(save_name + ".txt", "w+") as f:
                f.write("\n".join([o["text"] for o in bxs]))