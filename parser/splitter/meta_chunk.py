##https://github.com/IAAR-Shanghai/Meta-Chunking
from typing import List, Dict
import re
import math 
from nltk.tokenize import sent_tokenize
import jieba

def build_prompt(sentence1,sentence2):
    SENTENCES_PROMPT = '''这是一个文本分块任务.你是一位文本分析专家，请根据提供的句子的逻辑结构和语义内容，从下面两种方案中选择一种分块方式：
        1. 将“{}”分割成“{}”与“{}”两部分；
        2. 将“{}”不进行分割，保持原形式；
        请回答1或2。'''.format(sentence1+sentence2,sentence1,sentence2,sentence1+sentence2)
    return SENTENCES_PROMPT 

def split_text_by_punctuation(text,language): 
    if language=='zh': 
        sentences = jieba.cut(text, cut_all=False)  
        sentences_list = list(sentences)  
        sentences = []  
        temp_sentence = ""  
        for word in sentences_list:  
            if word in ["。", "！", "？","；"]:  
                sentences.append(temp_sentence.strip()+word)  
                temp_sentence = ""  
            else:  
                temp_sentence += word  
        if temp_sentence:   
            sentences.append(temp_sentence.strip())  
        
        return sentences
    else:
        full_segments = sent_tokenize(text)
        ret = []
        for item in full_segments:
            item_l = item.strip().split(' ')
            if len(item_l) > 512:
                if len(item_l) > 1024:
                    item = ' '.join(item_l[:256]) + "..."
                else:
                    item = ' '.join(item_l[:512]) + "..."
            ret.append(item)
        return ret
    
    
def meta_chunking(original_text,base_model,language,ppl_threshold,chunk_length):
    chunk_length=int(chunk_length)
    if base_model=='PPL Chunking':
        ##可以学学llm ppl输出与计算
        # final_chunks=extract_by_html2text_db_nolist(original_text,small_model,small_tokenizer,ppl_threshold,language=language)
        pass
    else:
        full_segments = split_text_by_punctuation(original_text,language)
        tmp=''
        threshold=0
        threshold_list=[]
        final_chunks=[]
        for sentence in full_segments:
            if tmp=='':
                tmp+=sentence
            else:
                prob_subtract="调用大模型API即可"    
                threshold_list.append(prob_subtract)
                if prob_subtract>threshold:
                    tmp+=' '+sentence
                else:
                    final_chunks.append(tmp)
                    tmp=sentence
            if len(threshold_list)>=5:
                last_ten = threshold_list[-5:]  
                avg = sum(last_ten) / len(last_ten)
                threshold=avg
        if tmp!='':
            final_chunks.append(tmp)
            
    merged_paragraphs = []
    current_paragraph = ""  
    if language=='zh':
        for paragraph in final_chunks:  
            if len(current_paragraph) + len(paragraph) <= chunk_length:  
                current_paragraph +=paragraph  
            else:  
                merged_paragraphs.append(current_paragraph)  
                current_paragraph = paragraph    
    else:
        for paragraph in final_chunks:  
            if len(current_paragraph.split()) + len(paragraph.split()) <= chunk_length:  
                current_paragraph +=' '+paragraph  
            else:  
                merged_paragraphs.append(current_paragraph)   
                current_paragraph = paragraph 
    if current_paragraph:  
        merged_paragraphs.append(current_paragraph) 
    final_text='\n\n'.join(merged_paragraphs)
    return final_text