from typing import List, Tuple
import loguru
from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM

from parser.splitter.late_chunk.chunking import Chunker
from parser.splitter.late_chunk.late_chunking import LateChunkingEmbedder, cosine_similarity
from parser.splitter.late_chunk.wrappers import load_model


def setup_local_llm(llm_name):
    
    model = AutoModelForCausalLM.from_pretrained(llm_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)

    def llm(prompt):
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model.generate(inputs, max_new_tokens=512)
        text_output = tokenizer.batch_decode(outputs)[0]
        if "<|assistant|>" in text_output:
            text_output = text_output.split("<|assistant|>")[1].strip()
        return text_output
    
    return llm
class ContextualRetrievalEmbedder():
    def __init__(self, 
            model: AutoModel,
            tokenizer: AutoTokenizer, 
            llm_name: str = "microsoft/Phi-3.5-mini-instruct",
            chunking_strategy: str = "fixed"
        ):

        self.llm = setup_local_llm(llm_name)
        # self.llm = request_anthropic_api

        self.prompt = """
        <document> 
        {{WHOLE_DOCUMENT}} 
        </document> 
        Here is the chunk we want to situate within the whole document 
        <chunk> 
        {{CHUNK_CONTENT}} 
        </chunk> 
        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. 
        """.strip()

        self.model = model
        self.tokenizer = tokenizer

        self.chunker = Chunker(chunking_strategy = chunking_strategy)


    def _add_context(self, chunk: str, document: str):
        prompt = self.prompt.replace("{{WHOLE_DOCUMENT}}", document).replace("{{CHUNK_CONTENT}}", chunk)
        extra_context = self.llm(prompt)
        return extra_context + " " + chunk

    def _tokens_to_text(self, text: str, annotations: List[Tuple[int, int]]):
        tokens = self.tokenizer.encode_plus(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        token_offsets = tokens.offset_mapping
        chunks = []
        for start, end in annotations:
            chunk = text[token_offsets[start][0]:token_offsets[end-1][1]]
            chunks.append(chunk)
        return chunks
    
    def run(self, document: str):
        annotations = [self.chunker.chunk(text=document, tokenizer=self.tokenizer, n_sentences=1)]
        self.chunks = self._tokens_to_text(text=document, annotations=annotations[0])
        self.chunks = [self._add_context(chunk, document) for chunk in self.chunks]

        model_outputs = self.model.encode(self.chunks)
        self.output_embs = [model_outputs[i, :] for i in range(len(self.chunks))]
        return self.output_embs

    def query(self, query: str):
        if "output_embs" not in dir(self):
            raise ValueError("no embeddings calculated, use .run(document) to create chunk embeddings")
        query_embedding = self.model.encode(query)
        similarities = []
        for emb in self.output_embs:
            similarities.append(cosine_similarity(query_embedding, emb))
        
        return similarities

def late_main_context():
    loguru.logger.info(f"late chunking start .....")
    text = """
    The recent SEC filing provided insights into ACME Corp's performance for Q2 2023. 
    It highlighted a 3% revenue growth over the previous quarter. 
    The company, which had a revenue of $314 million in the prior quarter, showed steady progress. 
    They attributed this growth to strategic initiatives and operational efficiencies. 
    The report emphasized the company's resilience and ability to navigate market challenges, reflecting positively on their financial health and future prospects.
    """.strip().replace("\n", "")
    
    test_tex = '''
    朝鲜国家媒体报道说，朝鲜领导人金正恩星期一(10月14日)召集了一次高层国家安全会议，下令在与韩国紧张关系加剧之际，制定一项“当前的军事行动”计划。
    参加在平壤召开的这次会议的有朝鲜的最高安全事务官员，包括军队总参谋长和其他军方官员以及国家保卫相和国防相。
    朝鲜中央通讯社报道说，“他指明了当前军事行动的方向，提出了完成遏制战争行动并行使自卫权的重大任务。”
    在召开这次会议之际，拥有核武器的朝鲜指责韩国派出无人机飞越其首都，朝鲜还向边界调动部队。韩国星期一表示，如果遭到火力射击，韩国“做好充分准备”。
    朝中社报道说，在平壤的这次会议上，官员们听取了有关“敌人严重挑衅”的报告，这显然是指无人机飞越事件。
    国家媒体说，金正恩在会议上“阐明了强硬的政治和军事立场”。
    朝鲜指责韩国要对投放了充满“煽动性谣言和垃圾”的宣传单的无人机负责，并在星期日警告说，如果再次侦测到无人机，朝鲜将视其为“宣战”。
    韩国军方最开始否认无人机飞行是其所为，当地的猜测焦点是韩国活动人士团体，他们长期以来都在向北方的朝鲜发送宣传品和美元，通常是使用气球。
    
    中國強大的財政部週六表示，將增加借款幫助資金短缺的地方政府，並向國有銀行注入更多資金。這是為了應對房地產市場嚴重下滑、支撐正在崩潰的消費者信心所做的努力。
財政部部長藍佛安沒有詳細說明中央政府打算為提振疲軟的國內消費、穩定房地產市場、鞏固銀行金融承諾增加多少借款或支出。但他暗示，這項計劃可能仍在制定中。
「經過法定程序後，會及時向社會公開，」他說。
做出這一宣布之前，中國曾在上個月頻繁推出經濟刺激措施，引發中國的股市狂飆，但隨著投資者越來越擔心政府的舉措可能不足以改變經濟狀況，股市已在上週有所回落。
財政部部長藍佛安和副部長廖岷週六表示，打算向中國最大的幾家銀行注資，提高銀行承受損失、繼續為經濟增長提供所需信貸的能力。許多投資者認為，在中國房地產市場整體崩潰的時候，這些銀行已在向企業和家庭發放貸款上蒙受了巨額損失，儘管它們迄今尚未承認有大的損失。
藍佛安多次表示，財政部希望地方政府通過出售資產來籌集資金。在過去30年的瘋狂投資期間，許多市級政府修建了許多辦公樓、酒店和會議中心。
    
    IT之家 10 月 15 日消息，索尼旗下一项标记为 US-20240335740 的无障碍专利于 10 月 10 日通过申请，该专利主要涉及在游戏中添加实时手语翻译工具，从而帮助残疾玩家在联机游玩时进行实时交流。
    '''

    llm_model_name = "microsoft/Phi-3.5-mini-instruct"
    embedding_model_name = "D:/InnovationProject/models/jina-embeddings-v3"

    embedding_model, has_instructions = load_model(embedding_model_name)
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, trust_remote_code=True)

    # cr = ContextualRetrievalEmbedder(embedding_model, embedding_tokenizer, llm_model_name, chunking_strategy="sentences")
    # cr.run(text)
    # cr_cosine_similarities = cr.query("What is ACME Corp's revenue growth for Q2 2023?")

    lc = LateChunkingEmbedder(embedding_model, embedding_tokenizer)
    late_file_path = "D:/InnovationProject/WALLE-AI/dataAgent/data/pdf_latex/GB_50202-2018建筑地基工程施工质量验收标准.tex"
    docs_chunks_list = []
    with open(late_file_path,'r',encoding='utf-8') as file:
        data = file.read()
        docs_chunks = lc.run(test_tex)
        loguru.logger.info(f"docs chunks {len(docs_chunks)}")


    # # import pandas as pd
    # for i, (cr_similarity, lc_similarity) in enumerate(zip(cr_cosine_similarities, lc_cosine_similarities)):
    #     print(f"{text.split('.')[:-1][i].strip()}")
    #     print(f"Similarities: Contextual Retrieval: {cr_similarity:.4f} | Late Chunking: {lc_similarity:.4f}")
    #     print("")
    