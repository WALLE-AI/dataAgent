STARCHAT_QS_QUESTION_GENERATOR_RPROMOPT = '''
你是一个建筑施工行业资深的质量检查员，你能够高精度判别出施工工地中施工质量风险，请根据用户的施工场景图片和隐患内容描述，请生成能够描述该图片场景的1个问题，如下为用户隐患内容描述输入：
隐患内容描述：{content}

请采用如下json格式进行输出：
[
    {
        question:xxx
    }  
]
'''

STARCHAT_QS_ANSWER_GENERATOR_RPROMOPT = '''
你是一个建筑施工行业资深的质量检查员，你能够高精度识别出施工工地中施工质量验收标准，请根据用户的场景图片、隐患内容和用户输入问题进行高质量的回复，需要重点分析出隐患类别、质量分析、整改要求和法规依据
隐患内容：{content1}
用户问题：{content2}
'''


PROMPT_TEST = '''
你是一个智能助手，你能够根据用户输入的时事新闻内容和上下文信息，能够解读其中主要时政问题,解读需要简要清楚，重点指出问题,字数不超过200字,如下为用户提供搜索上下文信息,如果用户输入于该主题不相关，请自己进行回复
{context}
'''


GENERATOR_QA_PROMPT_EN = (
    "<Task> The user will send a long text. Generate a Question and Answer pairs only using the knowledge"
    " in the long text. Please think step by step."
    "Step 1: Understand and summarize the main content of this text.\n"
    "Step 2: What key information or concepts are mentioned in this text?\n"
    "Step 3: Decompose or combine multiple pieces of information and concepts.\n"
    "Step 4: Generate questions and answers based on these key information and concepts.\n"
    "<Constraints> The questions should be clear and detailed, and the answers should be detailed and complete. "
    "You must answer in {language}, in a style that is clear and detailed in {language}."
    " No language other than {language} should be used. \n"
    "<Format> Use the following format: Q1:\nA1:\nQ2:\nA2:...\n"
    "<QA Pairs>"
)
##source https://github.com/Steven-Luo/MasteringRAG/blob/main/00_PDF%E8%A7%A3%E6%9E%90%E4%B8%8EQA%E6%8A%BD%E5%8F%96_v1.1.ipynb
GENERATOR_QA_PROMPT_ZH_2 = '''
我会给你一段文本（<document></document>之间的部分），你需要阅读这段文本，分别针对这段文本生成8个问题、用户回答这个问题的上下文，和基于上下文对问题的回答。

对问题、上下文、答案的要求：

问题要与这段文本相关，不要询问类似“这个问题的答案在哪一章”这样的问题
上下文：上下文必须与原始文本的内容保持一致，不要进行缩写、扩写、改写、摘要、替换词语等
答案：回答请保持完整且简洁，无须重复问题。答案要能够独立回答问题，而不是引用现有的章节、页码等

返回结果以JSON形式组织，格式为[{"question": "...", "context": ..., "answer": "..."}, ...]。
如果当前文本主要是目录，或者是一些人名、地址、电子邮箱等没有办法生成有意义的问题时，可以返回[]。

下方是文本：
<document>
{{document}}
</document>

请生成结果：
'''

GENERATOR_INSTRUCTIONS_SEED = "根据建筑行业中华人民共和国国家标准 {handbook}"


GENERATOR_QA_PROMPT_ZH = (
    "<任务>用户将发送长文本(Latex格式):{knowledge_data}。仅使用知识生成问题和答案对"
    "长文中，请一步步思考。"
    "第一步：理解并总结本文的主要内容。\n"
    "第 2 步：本文中提到了哪些关键信息或概念？\n"
    "第 3 步：分解或组合多条信息和概念。\n"
    "第 4 步: 不应使用本规范、本标准、该规范、该标准等形式描述问题，可采用 GB_50203-2011砌体结构工程施工质量验收规范 5.2 主控项目 5.2.1条款主要讲了什么 模板形式生成问题，"
    "第 5 步: 如果长文本中有目录，或者是一些人名、地址、电子邮箱、机构名称等不应生成问题,这些问题毫无意义，请过滤"
    "第 6 步：根据这些关键信息和概念生成问题和答案,请注意，不要幻觉，所有问题的答案需要都来自于长文本中。\n"
    "<限制>问题应涵盖文本中的关键信息和主要概念，答案应该提供一个全面、信息丰富的答案，涵盖问题的所有可能角度。"
    "您必须用{language}回答，并以{language}的方式清晰详细地回答。"
    " 不应使用 {language} 以外的任何语言。\n"
    "<格式> 使用以下格式：Q1:\nA1:\nQ2:\nA2:...\n"
    "<QA 对>"   
)

##source:https://github.com/wangxb96/RAG-QA-Generator/blob/main/Code/AutoQAGPro-EN.py
GENERATOR_QA_PROMPT_EN_1 = """Based on the following given text, generate a set of high-quality question-answer pairs. Please follow these guidelines:

1. Question part:
- Create as many different phrasings of questions (e.g., K questions) as you can for the same topic.
- Questions should cover key information and main concepts in the text.
- Use various questioning methods, such as direct inquiries, requests for confirmation, seeking explanations, etc.

2. Answer part:
- Provide a comprehensive, informative answer that covers all possible angles of the questions.
- The answer should be directly based on the given text, ensuring accuracy.
- Include relevant details such as dates, names, positions, and other specific information.

3. Format:
- Use "Q:" to mark the beginning of the question set, all questions should be in one paragraph.
- Use "A:" to mark the beginning of the answer.
- Separate question-answer pairs with a blank line.

4. Content requirements:
- Ensure that the question-answer pairs closely revolve around the text's topic.
- Avoid adding information not mentioned in the text.
- If the text information is insufficient to answer a certain aspect, you can state "Cannot be determined based on the given information" in the answer.

Example structure (for reference only, actual content should be based on the given text):

Q: [Question 1]? [Question 2]? [Question 3]? ...... [Question K]?

A: [Comprehensive, detailed answer covering all angles of the questions]

Given text:
{chunk}

Please generate question-answer pairs based on this text.
"""

PDF_PAGE_TO_MARKDOWN_PROMPT = """
Convert the following PDF page to markdown.
Return only the markdown with no explanation text.
Do not exclude any content from the page. 
Please generate content in the following json format
{
    "markdown":xxxx
}

"""


GENERATOR_ANSWER_PROMPT_EN_2 = '''
The background knowledge is:
{unsupervised_knowledge_data}
Please answer the following question based on the
content of the article above:
{the_generated_question}
Please answer this question as thoroughly as possible,
but do not change the key information in the original
text, and do not include expressions such as “based
on the above article” in the answer.
You must answer in {language}, in a style that is clear and detailed in {language}.
No language other than {language} should be used. 
Please generate the corresponding answer in the following format:
Question: ...
Answer: ...

'''


GENERATOR_QA_PROMPT_ZH_1 = '''基于以下给定的文本，生成一组高质量的问答对。请遵循以下指南：

            1. 问题部分：
            - 为同一个主题创建尽你所能多的（如K个）不同表述的问题。
            - 问题应涵盖文本中的关键信息和主要概念。
            - 禁止使用本规范、该规范和该国家标准形式描述问题，可采用 GB_50203-2011砌体结构工程施工质量验收规范 5.2 主控项目 5.2.1条款主要讲了什么
            - 使用多种提问方式，如直接询问、请求确认、寻求解释等。

            2. 答案部分：
            - 提供一个全面、信息丰富的答案，涵盖问题的所有可能角度。
            - 答案应直接基于给定文本，确保准确性。
            - 包含相关的细节，如日期、名称、职位等具体信息。

            3. 格式：
            - 使用"Q:"标记问题集合的开始，所有问题应在一个段落内。
            - 使用"A:"标记答案的开始。
            - 问答对之间用空行分隔。

            4. 内容要求：
            - 确保问答对紧密围绕文本主题。
            - 避免添加文本中未提及的信息。
            - 如果文本信息不足以回答某个方面，可以在答案中说明"根据给定信息无法确定"。

            示例结构（仅供参考，实际内容应基于给定文本）：

            采用如下格式输出： 
            Q1:\nA1:\nQ2:\nA2:...\n

            给定文本：
            {chunk}

            请基于这个文本生成问答对。
'''
QUALITY_MAIN_STRUCTURE_PROMOPT_LABEL_UPDATE ='''
    <角色>你是一个建筑施工行业资深的质量检查员，你能够高精度判别出施工工地中施工质量风险
    <任务>请根据用户的场景图片和隐患描述识别出属于如下隐患列表：
一级隐患类别：主体结构
二级隐患类别：混凝土结构、钢体结构、钢结构、钢管混凝土结构、型钢混凝土结构、铝合金结构、木结构
1、混凝土结构-三级隐患内容
模板
钢筋
混凝土
预应力
现浇结构
装配式结构
2、钢体结构-三级隐患内容
砖砌体
混凝土小型空心砌块砌体
石砌体
配筋砌体
填充墙砌体
钢结构-三级隐患内容
钢结构焊接
紧固件连接
钢零部件加工
钢构件组装及预拼装
单层钢结构安装
多层及高层钢结构安装
钢管结构安装
网壳、网架及桁架结构
压型金属板
防腐涂料涂装
防火涂料涂装
3、钢管混凝土结构-三级隐患内容
构件现场拼装
构件安装
柱与混凝土梁连接
钢管内钢筋骨架
钢管内混凝土浇筑
型钢混凝土结构-三级隐患内容
型钢焊接
紧固件连接
型钢与钢筋连接
型钢构件组装
预埋件装
型钢安装
模板
混凝土
4、铝合金结构-三级隐患内容
铝合金焊接
紧固件连接
铝合金零部件加工
铝合金构件组装
铝合金构件预拼装
铝合金框架结构安装
铝合金空间网格结构安装
铝合金面板
铝合金幕墙结构安装
防腐处理
5、木结构-三级隐患内容
方木和原木结构
胶合木结构
轻型木结构
木结构防护
    根据图片信息和隐患描述，请一步步思考。
    第 1 步：根据图片和隐患描述{content}中识别出存在的所有质量隐患内容，并且提炼出关键的隐患内容\n
    第 2 步：判断图片中存在隐患内容属于隐患列表中那个隐患类别\n
    您必须用中文回答，并以中文的方式清晰详细地回答。
     不应使用 中文 以外的任何语言。\n
    如果无法判断图片中存在质量隐患内容，请根据自己的理解进行高质量回复
    请按如下json格式输出
    {
        "二级隐患类别"：xxxx,
        "三级隐患类别"：xxx
    }
'''


QUALITY_MAIN_STRUCTURE_RISK_LABEL=[
    "蜂窝麻面",
    "麻面",
    "漏筋"
]

QUALITY_MAIN_STRUCTURE_RISK_PROMPT='''
你是一个建筑施工行业资深的质量检查员，你能够高精度判别出施工工地中施工质量风险，请根据用户的输入建筑施工工地图片和隐患部位进行高质量的回复
隐患描述：{risk_des}
隐患部位：{risk_part}
'''

LATEXT_TO_MARKDOWN_PROMPT='''
你是一个智能助手，能够高质量将Latex格式转成Markdown格式，根据用户输入的Latex文本信息高质量的转换成对应的Markdown格式,如果Latex信息中出现格式错误，请在转markdown过程中进行自我纠正，正确转换，不要幻觉.
Latex:{latex_content}
请按如下json格式输出
{    
    "markdown": xxxxxxx
}

'''



#《建筑工程施工质量验收统一标准 GB 50300-2013》 附录 B建筑工程的分部工程、分项工程划分规定
QUALITY_MAIN_STRUCTURE_PROMOPT_LABEL = '''
你是一个建筑施工行业资深的质量检查员，你能够高精度判别出施工工地中施工质量风险，请根据用户的场景图片和隐患内容识别出属于如下隐患类别中类别:
一级隐患类别：主体结构
二级隐患类别：混凝土结构、钢体结构、钢结构、钢管混凝土结构、型钢混凝土结构、铝合金结构、木结构
1、混凝土结构-三级隐患内容
模板
钢筋
混凝土
预应力
现浇结构
装配式结构
2、钢体结构-三级隐患内容
砖砌体
混凝土小型空心砌块砌体
石砌体
配筋砌体
填充墙砌体
钢结构-三级隐患内容
钢结构焊接
紧固件连接
钢零部件加工
钢构件组装及预拼装
单层钢结构安装
多层及高层钢结构安装
钢管结构安装
网壳、网架及桁架结构
压型金属板
防腐涂料涂装
防火涂料涂装
3、钢管混凝土结构-三级隐患内容
构件现场拼装
构件安装
柱与混凝土梁连接
钢管内钢筋骨架
钢管内混凝土浇筑
型钢混凝土结构-三级隐患内容
型钢焊接
紧固件连接
型钢与钢筋连接
型钢构件组装
预埋件装
型钢安装
模板
混凝土
4、铝合金结构-三级隐患内容
铝合金焊接
紧固件连接
铝合金零部件加工
铝合金构件组装
铝合金构件预拼装
铝合金框架结构安装
铝合金空间网格结构安装
铝合金面板
铝合金幕墙结构安装
防腐处理
5、木结构-三级隐患内容
方木和原木结构
胶合木结构
轻型木结构
木结构防护
请根据用户输入的图片和隐患内容进行高质量的回复，如果识别出图片中质量隐患类别不在上述描述中,请根据自己理解指出具体的质量隐患类别
请按如下json格式输出
{
    "二级隐患类别"：xxxx,
    "三级隐患类别"：xxx
}
隐患内容：{content}

'''


QUALITY_MAIN_STRUCTURE_PROMOPT_LABEL_TEST = '''
你是一个建筑施工行业资深的质量检查员，你能够高精度判别出施工工地中施工质量风险，请根据用户的场景图片和隐患内容识别出属于如下隐患类别中类别:
一级隐患类别：主体结构
二级隐患类别：钢筋工程，模板工程，混凝土工程，钢结构工程，砌筑工程，装配式结构
1、钢筋工程-三级隐患内容：
钢筋原材复试检验不合格
钢筋原材、半成品无成品保护
直螺纹丝头未戴保护帽
箍筋加工形状、尺寸不符合设计要求
梁柱节点核心区箍筋缺失
梁钢筋分隔不到位
洞口加强筋缺失
搭接长度、百分比不符合要求
直螺纹丝头外露超过2个螺距
钢筋保护层超偏
2、模板工程-三级隐患内容：
模板承载力、强度、刚度不符合设计要求
模板加固与方案不符
封闭式模板无排气孔
模板拼装接缝不平整严密
模板未刷脱模剂
模板起拱高度与设计不符
县臂构件未设置独立支撑
混凝土接茬处模板未下挂
水平模板拆除无依据
墻柱底部无防漏浆措施
3、混凝土-三级隐患内容：
高低标号混凝土混浇
现场随意更改混凝土配合比
未按规走对砼结构进行养护
无依据、无措施提前拆模、上人或堆料
砼严重质量缺陷
抗渗构件出现渗漏
新旧混凝土接茬不顺直，错台
泛水高度砼构件未随主体一次性浇筑
施工缝结合面剔凿不到位
后浇带支撑提前拆除,或未独立搭设
4、钢结构工程-三级隐患内容：
钢材质量缺陷
钢杜安装精度和标高傧差
钢柱与钢梁、钢梁与钢梁连接错位
螺栓孔错位
地脚栓位移
焊接质量缺陷
焊缝错边
压型钢板安装漏縫
防火涂料厚度不足
涂装质量缺陷
5、砌筑工程-三级隐患内容：
构造柱、圈梁、过梁/压顶设置不符合要求
外墙施工不规范(灰缝、构造柱、孔洞封堵、窗台压顶)
应设导墙处未设导墙、设置高度不足断砖、瞎缝、通、假、透光縫
砌体留槎不规范
构造柱钢筋连接不规范
后塞口施工未间隔14d
砂浆无垫设措施
植筋不牢固，间距、长度不满足要求
砌块需使用专业开槽机开槽，刀砍斧劈
6、装配式结构-三级隐患内容：
预制构件外观尺寸不达标
预制构件预埋件定位错误
灌浆料性能不符合要求
注浆孔漏灌
楼梯预埋螺栓长度不符合要求
预制构件堆放高度过高
预制构件变形
构件粗糙面做法不合格
请根据用户输入的图片和隐患内容进行高质量的回复，如果识别出图片中质量隐患类别不在上述描述中,请指出具体的质量隐患类别，如果用户输入图片和文本与该场景不相关，请采用通识策略进行回复
请按如下json格式输出
{
    "二级隐患类别"：xxxx,
    "三级隐患类别"：xxx
}
隐患内容：{content}

'''
QUALITY_FOUNDATIONS_AND_FOUNDATION_ENGINEERING_PROMOPT_LABEL='''
你是一个建筑施工行业资深的质量检查员，你能够高精度判别出施工工地中施工质量风险，请根据用户的场景图片和隐患内容识别出属于如下隐患类别中类别:
一级隐患类别：地基与基础工程
二级隐患类别：土方，基坑支护，地基处理，桩基础，钢结构基础，筏板混凝土结构基础，型钢混凝土结构基础，型钢混凝土结构基础，地下防水
1、土方-三级隐患类别
土方开挖
土方回填
场地平整
2、基坑支护-三级隐患类别
灌注桩排桩围护墙
重力式挡土墙
板桩钢护墙
草袋水泥土模袱墙
土钉墙与复合土钉墙
地下连续墙
护坡
沉井与沉箱
钢或混凝土支撑
锚杆(索)
与主体结构相结合的基坑支护
降水与排水
3、地基处理-三级隐患类别
夯实、夯土填垫
砂和砂石垫层
土工合成材料垫层
粉煤灰垫层
强夯地基
注浆加固地基
置换地基
高压喷射注浆地基
水泥土搅拌桩地基
土和灰土挤密桩地基
水泥粉煤灰碎石桩地基
夯实水泥土桩地基
砂桩地基
振冲桩和振冲密实桩
碎石桩
挤淤置换法
塑料排水板
爆破挤淤法
4、桩基础-三级隐患类别
混凝土灌注桩
长螺旋钻孔灌注桩
沉管灌注桩
钻孔灌注桩
灌柱桩压桩
预制桩
钢筋、钢筋笼
混凝土
预应力
现浇结构
装配式结构
复合桩
混凝土小型预制桩
钢桩
木桩
5、钢结构基础-三级隐患类别
钢结构焊接
紧固件连接
钢结构制作
钢结构安装
防腐涂料涂装
6、筏板混凝土结构基础-三级隐患类别
钢筋进场验收
钢筋现场绑扎
柱脚锚固
钢筋安装
柱
与混凝土梁连接
钢管内嵌混凝土柱
钢管混凝土涂装
型钢混凝土结构基础-三级隐患类别
型钢焊接
紧固件连接
型钢与钢筋连接
结构件组装
预埋件装
模板
混凝土
7、地下防水-三级隐患类别
土体结构防水
细部构造防水
特殊施工法结构防水
注浆

请根据用户输入的图片和隐患内容进行高质量的回复，如果识别出图片中质量隐患类别不在上述描述中,请指出具体的质量隐患类别，如果用户输入图片和文本与该场景不相关，请采用通识策略进行回复
请按如下json格式输出
{
    "二级隐患类别"：xxxx,
    "三级隐患类别"：xxx
}
隐患内容：{content}
'''

QUALITY_BUILDING_DECORATION_AND_DECORATION_PROMOPT_LABEL='''
你是一个建筑施工行业资深的质量检查员，你能够高精度判别出施工工地中施工质量风险，请根据用户的场景图片和隐患内容识别出属于如下隐患类别中类别:
一级隐患类别：建筑装饰装修
二级隐患类别：建筑地面，抹灰，外墙防水，门窗，吊顶，轻质隔墙，饰面板，饰面砖，幕墙，涂饰，裱糊与软包，细部
1、建筑地面-三级隐患类别
基层铺设
整体面层铺设
板块面层铺设
木、竹面层铺设
抹灰-三级隐患类别
一般抹灰
保温层抹灰
装饰抹灰
清水面体勾缝
2、外墙防水-三级隐患类别
外墙防水防腐
涂膜防水
透气膜防水
3、门窗-三级隐患类别
木门窗安装
金属门窗安装
塑料门窗安装
特种门安装
门窗玻璃安装
4、吊顶-三级隐患类别
整体面层吊顶
板块面层吊顶
格栅吊顶
5、轻质隔墙-三级隐患类别
板材隔墙
骨架隔墙
活动隔墙
玻璃隔墙
6、饰面板-三级隐患类别
石板安装
陶瓷板安装
木板安装
金属板安装
塑料板安装
饰面砖-三级隐患类别
外墙饰面砖粘贴
内墙饰面砖粘贴
7、幕墙-三级隐患类别
玻璃幕墙安装
金属幕墙安装
石材幕墙安装
陶板幕墙安装
涂饰-三级隐患类别
水性涂料涂饰
溶剂型涂料涂饰
美术涂饰
8、裱糊与软包-三级隐患类别
裱糊
软包
9、细部-三级隐患类别
柜架制作与安装
窗帘盒和窗台板制作与安装
门窗套制作与安装
护栏和扶手安装
花饰制作与安装
请根据用户输入的图片和隐患内容进行高质量的回复，如果识别出图片中质量隐患类别不在上述描述中,请指出具体的质量隐患类别，如果用户输入图片和文本与该场景不相关，请采用通识策略进行回复
请按如下json格式输出
{
    "二级隐患类别"：xxxx,
    "三级隐患类别"：xxx
}
隐患内容：{content}

'''
