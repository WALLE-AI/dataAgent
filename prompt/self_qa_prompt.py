SELF_GENERATOR_QA_INSTRUCTION_PROMPT = '''
The background knowledge is:
{unsupervised_knowledge_data}

Please generate ten instruction questions as diverse
as possible based on the content of the above article.
These questions can be questions about facts or an
understanding and evaluation of relevant content.
Please assume that there is no corresponding article
to refer to when asking questions, so do not use
demonstrative pronouns such as “this” or “these” in
the question.
You must answer in {language}, in a style that is clear and detailed in {language}.
No language other than {language} should be used. 
Please generate questions in the following format:
1. Question: ...
2. Question: ...

'''

SELF_GENERATOR_ANSWER_PROMPT = '''
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