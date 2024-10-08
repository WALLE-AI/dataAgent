#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import copy
import os
from pathlib import Path
from tika import parser
import re
from io import BytesIO

from parser.vision.deepdoc.pdf_parser import PdfParser, PlainParser
from parser.vision.deepdoc.tokenizers import rag_tokenizer
from parser.vision.utils.deepdoc_utils import bullets_category,is_english, tokenize, remove_contents_table, \
    hierarchical_merge, make_colon_as_title, naive_merge, random_choices, tokenize_table, add_positions, \
    tokenize_chunks, find_codec
    


class Pdf(PdfParser):
    def __call__(self, filename, binary=None, from_page=0,
                 to_page=100000, zoomin=3, callback=None):
        callback(msg="OCR is running...")
        self.__images__(
            filename if not binary else binary,
            zoomin,
            from_page,
            to_page,
            callback)
        callback(msg="OCR finished")

        from timeit import default_timer as timer
        start = timer()
        self._layouts_rec(zoomin)
        callback(0.67, "Layout analysis finished")
        print("layouts:", timer() - start)
        self._table_transformer_job(zoomin)
        callback(0.68, "Table analysis finished")
        self._text_merge()
        tbls = self._extract_table_figure(True, zoomin, True, True)
        self._naive_vertical_merge()
        self._filter_forpages()
        self._merge_with_same_bullet()
        callback(0.75, "Text merging finished.")

        callback(0.8, "Text extraction finished")

        return [(b["text"] + self._line_tag(b, zoomin), b.get("layoutno", ""))
                for b in self.boxes], tbls

def save_table_images(res_list):
    '''
    {'docnm_kwd': 'D:/InnovationProject/WALLE-AI/dataAgent/data/pdf/GB_50203-2011砌体结构工程施工质量验收规范.pdf',
    'title_tks': ' d : / innovationproject / walle-ai / dataag / data / pdf / gb _ 50203-2011 砌体 结构 工程施工 质量 验收 规范', 
    'title_sm_tks': ' d : / innovationproject / walle-ai / dataag / data / pdf / gb _ 50203-2011 砌体 结构 工程 施工 质量 验收 规范',
    'content_with_weight': '10 Winter Construction ',
    'content_ltks': '10 winter construct', 'content_sm_ltks': 
    '10 winter construct', 'image': <PIL.Image.Image image mode=RGB size=345x31 at 0x247717138B0>, 
    'page_num_int': [9], 'position_int': [(...)], 'top_int': [43]}
    '''
    for table_images in res_list:
        file_name = Path(table_images['docnm_kwd']).stem
        page_num = table_images['page_num_int'][0]
        tabel_images_save_path = os.getenv('TABLE_IMAGES_SAVE')
        save_image_name = tabel_images_save_path +file_name +"_"+str(page_num)+"_"+".png"
        table_images["image"].save(save_image_name)


def chunk(filename, binary=None, from_page=0, to_page=100000,
          lang="Chinese", callback=None, **kwargs):
    """
        Supported file formats are docx, pdf, txt.
        Since a book is long and not all the parts are useful, if it's a PDF,
        please setup the page ranges for every book in order eliminate negative effects and save elapsed computing time.
    """
    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
    }
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
    pdf_parser = None
    sections, tbls = [], []

    if re.search(r"\.pdf$", filename, re.IGNORECASE):
        pdf_parser = Pdf() if kwargs.get(
            "parser_config", {}).get(
            "layout_recognize", True) else PlainParser()
        sections, tbls = pdf_parser(filename if not binary else binary,
                                    from_page=from_page, to_page=to_page, callback=callback)

    else:
        raise NotImplementedError(
            "file type not supported yet(doc, docx, pdf, txt supported)")

    make_colon_as_title(sections)
    bull = bullets_category(
        [t for t in random_choices([t for t, _ in sections], k=100)])
    if bull >= 0:
        chunks = ["\n".join(ck)
                  for ck in hierarchical_merge(bull, sections, 5)]
    else:
        sections = [s.split("@") for s, _ in sections]
        sections = [(pr[0], "@" + pr[1]) if len(pr) == 2 else (pr[0], '') for pr in sections ]
        chunks = naive_merge(
            sections, kwargs.get(
                "chunk_token_num", 1024), kwargs.get(
                "delimer", "\n。；！？"))

    # is it English
    # is_english(random_choices([t for t, _ in sections], k=218))
    eng = lang.lower() == "english"

    res = tokenize_table(tbls, doc, eng)
    ##保存table数据图片
    save_table_images(res)
    res.extend(tokenize_chunks(chunks, doc, eng, pdf_parser))

    return res
