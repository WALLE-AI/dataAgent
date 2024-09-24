import re


class CleanProcessor:
    @classmethod
    def clean(cls, text: str, all_remove=None) -> str:
        # default clean
        # remove invalid symbol
        text = re.sub(r"<\|", "<", text)
        text = re.sub(r"\|>", ">", text)
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\xEF\xBF\xBE]", "", text)
        # Unicode  U+FFFE
        text = re.sub("\ufffe", "", text)
        if all_remove is not None:
            text = cls()._remove_urls_emails_spaces(text)
        return text
    
    def _remove_urls_emails_spaces(self,text):
        # Remove extra spaces
        pattern = r"\n{3,}"
        text = re.sub(pattern, "\n\n", text)
        pattern = r"[\t\f\r\x20\u00a0\u1680\u180e\u2000-\u200a\u202f\u205f\u3000]{2,}"
        text = re.sub(pattern, " ", text)
        
        # Remove email
        pattern = r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
        text = re.sub(pattern, "", text)
        # Remove URL
        pattern = r"https?://[^\s]+"
        text = re.sub(pattern, "", text)
        return text
        
    def _remove_names_or_organization(self):
        '''
        移除手册中姓名和机构名称
        '''
        pass
        

    def filter_string(self, text):
        return text
