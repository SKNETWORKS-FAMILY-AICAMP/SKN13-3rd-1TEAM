# -*- coding: utf-8 -*-
"""
RAGAS 평가 자동화 스크립트 (main.ipynb의 llm 서비스 평가 목적)
- 5개 vector_store에서 context 추출
- LLM으로 50쌍의 질문-정답 쌍 생성
- RAG chain으로 답변 생성
- RAGAS로 평가
- 결과 저장 (RAGAS 디렉토리)
"""

import os
import random
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# 1. LLM 및 임베딩 모델 준비
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
llm = ChatOpenAI(model="gpt-4.1-mini")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# 2. vector_store 목록
VECTOR_STORE_DIRS = [
    "crawling/vector_store/vitamin_chroma",
    "crawling/vector_store/soomin",
    "crawling/vector_store/others",
    "crawling/vector_store/chroma_pubmed_v3_large",
    "crawling/vector_store/chroma"
]

# 3. 각 vector_store에서 context 추출
from langchain_chroma import Chroma
all_contexts = []
CONTEXTS_TOTAL = 50
MIN_CONTEXT_LEN = 200

# 각 store에서 골고루 추출 (총 50개)
per_store = CONTEXTS_TOTAL // len(VECTOR_STORE_DIRS)
for store_dir in VECTOR_STORE_DIRS:
    vector_store = Chroma(
        embedding_function=embedding_model,
        persist_directory=store_dir
    )
    docs = vector_store._collection.get(include=['documents'])['documents']
    contexts = [doc for doc in docs if len(doc) > MIN_CONTEXT_LEN]
    random.shuffle(contexts)
    all_contexts.extend(contexts[:per_store])
# 만약 50개가 안되면 추가로 채움
if len(all_contexts) < CONTEXTS_TOTAL:
    remain = CONTEXTS_TOTAL - len(all_contexts)
    extra_contexts = []
    for store_dir in VECTOR_STORE_DIRS:
        vector_store = Chroma(
            embedding_function=embedding_model,
            persist_directory=store_dir
        )
        docs = vector_store._collection.get(include=['documents'])['documents']
        contexts = [doc for doc in docs if len(doc) > MIN_CONTEXT_LEN]
        random.shuffle(contexts)
        extra_contexts.extend(contexts[per_store:per_store+remain])
        if len(extra_contexts) >= remain:
            break
    all_contexts.extend(extra_contexts[:remain])

print(f"총 추출된 context 개수: {len(all_contexts)} (목표: {CONTEXTS_TOTAL})")

# 4. context로부터 질문-정답 쌍 생성
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 질문-정답 스키마 정의
class EvalSchema(BaseModel):
    user_input: str = Field(..., description="사용자 질문")
    reference: str = Field(..., description="user_input(사용자 질문)에 대한 정답")
    qa_context: str = Field(..., description="질문, 답변 쌍을 만들때 참조한 context. 입력된 context를 수정하지 않고 그대로 넣는다.")

parser = JsonOutputParser(pydantic_object=EvalSchema)

# 질문-정답 생성 프롬프트
template = """# Instruction:
당신은 RAG 평가를 위해 질문과 정답 쌍을 생성하는 인공지능 비서입니다.
다음 [Context] 에 문서가 주어지면 해당 문서를 기반으로 {num_questions}개 질문-정답 쌍을 생성하세요. 

질문과 정답을 생성한 후 Output Indicator의 format으로 출력합니다.
질문은 반드시 Context 문서에 있는 정보를 바탕으로 생성해야 합니다. Context에 없는 내용을 가지고 질문-정답을 절대 만들면 안됩니다.
질문은 간결하게 작성합니다.
하나의 질문에는 한 가지씩만 내용만 작성합니다.
정답은 반드시 Context에 있는 정보를 바탕으로 작성합니다. 없는 내용을 추가하지 않습니다.
질문과 정답을 만들고 그 내용이 Context에 있는 항목인지 다시 한번 확인합니다.
생성된 질문-답변 쌍은 반드시 dictionary 형태로 정의하고 list로 묶어서 반환해야 합니다.
질문-답변 쌍은 반드시 {num_questions}개를 만들어야 합니다.

#Context:
{context}

Output Indicator:
{format_instructions}
"""

prompt_template = PromptTemplate(
    template=template, 
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
eval_model = ChatOpenAI(model="gpt-4.1")
eval_dataset_chain = prompt_template | eval_model | parser

# context당 1개씩만 질문-정답 쌍 생성(총 50개)
eval_dataset_list = []
num_questions = 1
for context in all_contexts:
    try:
        eval_data = eval_dataset_chain.invoke({"context": context, "num_questions": num_questions})
        if isinstance(eval_data, list):
            eval_dataset_list.extend(eval_data)
        else:
            eval_dataset_list.append(eval_data)
    except Exception as e:
        print(f"질문-정답 생성 실패: {e}")

print(f"생성된 질문-정답 쌍 개수: {len(eval_dataset_list)} (목표: {CONTEXTS_TOTAL})")

# DataFrame으로 변환
eval_df = pd.DataFrame(eval_dataset_list)

# 5. rag_chain 구성 및 답변/검색 context 생성
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# rag_chain 프롬프트(11_RAG_evaluation.ipynb 참고)
rag_template = """# Instruction:
당신은 정확한 정보 제공을 우선시하는 인공지능 어시스턴트입니다.
주어진 Context에 포함된 정보만 사용해서 질문에 답변하세요.
Context에 질문에 대한 명확한 정보가 있는 경우 그 내용을 바탕으로 답변하세요.
Context에 질문에 대한 명확한 정보없을 경우 "정보가 부족해서 답을 알 수 없습니다." 라고 대답합니다.
절대 Context에 없는 내용을 추측하거나 일반 상식을 이용해 답을 만들어서 대답하지 않습니다.

# Context:
{context}

# 질문:
{query}
"""
rag_prompt_template = PromptTemplate(template=rag_template)

# 평가 목적: 5개 store 모두에서 검색하도록 retriever를 각 store별로 순회
retrievers = [
    Chroma(
        embedding_function=embedding_model,
        persist_directory=store_dir
    ).as_retriever() for store_dir in VECTOR_STORE_DIRS
]

def get_rag_response(user_input):
    # 5개 store에서 검색된 context를 모두 합침
    all_contexts = []
    for retriever in retrievers:
        docs = retriever.get_relevant_documents(user_input)
        all_contexts.extend([doc.page_content if hasattr(doc, "page_content") else doc for doc in docs])
    # 중복 제거
    all_contexts = list(dict.fromkeys(all_contexts))
    context = "\n\n".join(all_contexts)
    prompt = rag_prompt_template.format(context=context, query=user_input)
    response = llm.invoke(prompt)
    return all_contexts, response.content

context_list = []
response_list = []
for user_input in eval_df['user_input']:
    try:
        retrieved_contexts, rag_response = get_rag_response(user_input)
        context_list.append(retrieved_contexts)
        response_list.append(rag_response)
    except Exception as e:
        print(f"rag_chain 실패: {e}")
        context_list.append([])
        response_list.append("")

# DataFrame에 추가
eval_df['retrieved_contexts'] = context_list
eval_df['response'] = response_list

# 6. RAGAS 평가
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    LLMContextRecall, Faithfulness, LLMContextPrecisionWithReference, AnswerRelevancy
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

eval_llm = LangchainLLMWrapper(llm)
eval_embedding = LangchainEmbeddingsWrapper(embedding_model)
metrics = [
    LLMContextPrecisionWithReference(llm=eval_llm),
    LLMContextRecall(llm=eval_llm),
    Faithfulness(llm=eval_llm),
    AnswerRelevancy(llm=eval_llm, embeddings=eval_embedding)
]

evaludation_dataset = EvaluationDataset.from_pandas(eval_df)
eval_result = evaluate(dataset=evaludation_dataset, metrics=metrics)

# 7. 결과 저장
eval_df.to_csv("RAGAS/ragas_eval_result.csv", index=False)
result_df = eval_result.to_pandas()
result_df.to_csv("RAGAS/ragas_score.csv", index=False)

print("RAGAS 평가 완료. 결과는 RAGAS/ragas_eval_result.csv, RAGAS/ragas_score.csv에 저장됨.") 