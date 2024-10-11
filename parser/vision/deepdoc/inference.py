import os
from pathlib import Path
import re
import loguru
import numpy as np
from parser.vision.deepdoc.deepdoc_pdf_extractor import chunk
from parser.vision.deepdoc.deepdoc_pdf_table_extractor import table_extractor
from parser.vision.deepdoc.layout_recognizer import LayoutRecognizer
from parser.vision.deepdoc.ocr import OCR
from parser.vision.deepdoc.recognizer import Recognizer
from parser.vision.deepdoc.seeit import draw_box
from parser.vision.deepdoc.table_structure_recognizer import TableStructureRecognizer
from parser.vision.utils.utils import get_directory_all_pdf_files
from utils.helper import pdf_file_image


def test_deepdoc_ocr_inference():
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
                
def test_deepdoc_layout_recognizer_inference():
    pdf_dir_path =os.getenv("PDF_DIR_ROOT")
    all_pdf_files = get_directory_all_pdf_files(pdf_dir_path)
    labels = LayoutRecognizer.labels
    detr = Recognizer(
            labels,
            "layout",
            os.getenv("DEEP_DOC_MODEL")
    )
    threshold=0.5
    for pdf_file in all_pdf_files:
        pdf_file_path = Path(pdf_file)
        pdf_image = pdf_file_image(pdf_file)
        loguru.logger.info(f"pdf file: {pdf_file}")
        layouts = detr(pdf_image, float(threshold))
        for index,lyt in enumerate(layouts):
            img = draw_box(pdf_image[index], lyt, labels, float(threshold))
            save_name ="data/deepdoc_test/"+pdf_file_path.stem+"_"+str(index)+".png"
            img.save(save_name, quality=95)
            print("save result to: " + save_name)
            
def get_table_html(img, tb_cpns, ocr):
    from PIL import Image
    images = "datasets/tables_images_save/《混凝土结构工程施工质量验收规范 GB50204-2015》_21_.png"
    images = Image.open(images)
    boxes = ocr(np.array(images))
    boxes = Recognizer.sort_Y_firstly(
        [{"x0": b[0][0], "x1": b[1][0],
          "top": b[0][1], "text": t[0],
          "bottom": b[-1][1],
          "layout_type": "table",
          "page_number": 0} for b, t in boxes if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]],
        np.mean([b[-1][1] - b[0][1] for b, _ in boxes]) / 3
    )

    def gather(kwd, fzy=10, ption=0.6):
        nonlocal boxes
        eles = Recognizer.sort_Y_firstly(
            [r for r in tb_cpns if re.match(kwd, r["label"])], fzy)
        eles = Recognizer.layouts_cleanup(boxes, eles, 5, ption)
        return Recognizer.sort_Y_firstly(eles, 0)

    headers = gather(r".*header$")
    rows = gather(r".* (row|header)")
    spans = gather(r".*spanning")
    clmns = sorted([r for r in tb_cpns if re.match(
        r"table column$", r["label"])], key=lambda x: x["x0"])
    clmns = Recognizer.layouts_cleanup(boxes, clmns, 5, 0.5)

    for b in boxes:
        ii = Recognizer.find_overlapped_with_threashold(b, rows, thr=0.3)
        if ii is not None:
            b["R"] = ii
            b["R_top"] = rows[ii]["top"]
            b["R_bott"] = rows[ii]["bottom"]

        ii = Recognizer.find_overlapped_with_threashold(b, headers, thr=0.3)
        if ii is not None:
            b["H_top"] = headers[ii]["top"]
            b["H_bott"] = headers[ii]["bottom"]
            b["H_left"] = headers[ii]["x0"]
            b["H_right"] = headers[ii]["x1"]
            b["H"] = ii

        ii = Recognizer.find_horizontally_tightest_fit(b, clmns)
        if ii is not None:
            b["C"] = ii
            b["C_left"] = clmns[ii]["x0"]
            b["C_right"] = clmns[ii]["x1"]

        ii = Recognizer.find_overlapped_with_threashold(b, spans, thr=0.3)
        if ii is not None:
            b["H_top"] = spans[ii]["top"]
            b["H_bott"] = spans[ii]["bottom"]
            b["H_left"] = spans[ii]["x0"]
            b["H_right"] = spans[ii]["x1"]
            b["SP"] = ii

#     html = """
#     <html>
#     <head>
#     <style>
#     ._table_1nkzy_11 {
#       margin: auto;
#       width: 70%%;
#       padding: 10px;
#     }
#     ._table_1nkzy_11 p {
#       margin-bottom: 50px;
#       border: 1px solid #e1e1e1;
#     }

#     caption {
#       color: #6ac1ca;
#       font-size: 20px;
#       height: 50px;
#       line-height: 50px;
#       font-weight: 600;
#       margin-bottom: 10px;
#     }

#     ._table_1nkzy_11 table {
#       width: 100%%;
#       border-collapse: collapse;
#     }

#     th {
#       color: #fff;
#       background-color: #6ac1ca;
#     }

#     td:hover {
#       background: #c1e8e8;
#     }

#     tr:nth-child(even) {
#       background-color: #f2f2f2;
#     }

#     ._table_1nkzy_11 th,
#     ._table_1nkzy_11 td {
#       text-align: center;
#       border: 1px solid #ddd;
#       padding: 8px;
#     }
#     </style>
#     </head>
#     <body>
#     %s
#     </body>
#     </html>
# """ % TableStructureRecognizer.construct_table(boxes, html=False)
    content = TableStructureRecognizer.construct_table(boxes, html=True)
    return content
       
def test_deepdoc_layout_recognizer_inference_trs():
    # labels = TableStructureRecognizer.labels
    # detr = TableStructureRecognizer()
    # ocr = OCR()
    # threshold=0.8
    pdf_file = "data/test_ocr.pdf"
    # pdf_image = pdf_file_image(pdf_file)  
    # layouts = detr(pdf_image, threshold)
    # for i, lyt in enumerate(layouts):
    #     html = get_table_html(pdf_image[i], lyt, ocr)
    #     loguru.logger.info(f"html:{html}")
    def dummy(prog=None, msg=""):
            pass   
    tbls= table_extractor(pdf_file,callback=dummy)
    for (img, rows), poss in tbls:
        loguru.logger.info(f"tbl info:{rows}")
      
def deepdoc_ocr_pdf_text_extract():
    def dummy(prog=None, msg=""):
        pass
    pdf_dir_path = ""
    pdf_dir_path =os.getenv("PDF_DIR_ROOT")
    all_pdf_files = get_directory_all_pdf_files(pdf_dir_path)
    index = 0
    for pdf_file in all_pdf_files:
        loguru.logger.info(f"pdf file: {pdf_file}")
        chunk("data/pdf/《砌体结构工程施工质量验收规范 GB50203-2011》.pdf", from_page=1, to_page=10000, callback=dummy)
        if index==0:
            break
            
