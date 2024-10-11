import re
from parser.vision.deepdoc.pdf_parser import PdfParser, PlainParser


class Pdf(PdfParser):
    def __call__(self, filename,from_page=0,
                 to_page=100000, zoomin=3,  binary=None, callback=None):
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
        callback(0.80, "Layout analysis finished")
        print("layouts:", timer() - start)
        self._table_transformer_job(zoomin)
        callback(0.80, "Table analysis finished")
        self._text_merge()
        tbls = self._extract_table_figure(True, zoomin, True, True)
        return tbls
        
  
def table_extractor(filename, binary=None, from_page=0, to_page=100000,
          lang="Chinese", callback=None, **kwargs):
    """
        Supported file formats are docx, pdf, txt.
        Since a book is long and not all the parts are useful, if it's a PDF,
        please setup the page ranges for every book in order eliminate negative effects and save elapsed computing time.
    """
    pdf_parser = None
    sections, tbls = [], []

    if re.search(r"\.pdf$", filename, re.IGNORECASE):
        pdf_parser = Pdf() if kwargs.get(
            "parser_config", {}).get(
            "layout_recognize", True) else PlainParser()
        tbls = pdf_parser(filename if not binary else binary,
                                    from_page=from_page, to_page=to_page, callback=callback)
    else:
        raise NotImplementedError(
            "file type not supported yet(doc, docx, pdf, txt supported)") 
    return tbls