from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain.prompts import ChatPromptTemplate
from textwrap import dedent
from typing import TypedDict
from dotenv import load_dotenv
import os

env_path = "C:/Aicamp/SKN13_my/13_Langchain/.env"
load_dotenv(dotenv_path=env_path)

# 상태 정의
class CompareState(TypedDict):
    question: str
    social_knowledge: str
    latest_docs: list
    final_answer: str
    prompt: object

# 모델 정의
model = ChatOpenAI(model="gpt-4.1", streaming=True)  # 스트리밍 가능 모델
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# ChatPromptTemplate 정의
COMPARE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", dedent("""
        당신은 다양한 지식을 제공하는 AI 어시스턴트입니다.
        주요 목표는 사용자의 요청에 대한 정확한 정보를 이해하기 쉽게 설명하는 것입니다.

        Instruction:
        아래 사용자 질문, 사회통념 지식, 최신 논문 기반 정보를 바탕으로
        최신 논문이 기존 사회통념과 어떻게 다른지 비교하고 종합적인 요약을 제시하세요.
        모든 응답은 챗봇과 같은 대화형 스타일을 유지하세요. 친근하고 쉽고 자연스럽게 다변하되 전문성을 보이는 어조를 유지하세요.

        사회통념 지식, 최신 논문 기반 정보를 찾아본 결과가 없다면, 검색결과가 없다고 말해주세요.
        """)),
    ("human", dedent("""
        사용자 질문: {question}

        [사회통념 지식]
        {social_knowledge}

        [최신 논문 기반 정보]
        {chunks}

        위 내용을 바탕으로, 최신 논문이 기존 사회통념과 어떻게 다른지 비교하고, 종합적인 요약을 제시해 주세요.
        """))
])

# 노드: 질문 분석
def analyze_question(state: CompareState) -> CompareState:
    print("질문 분석:", state["question"])
    return state

# 노드: 사회통념 지식 검색 (Wikipedia)
def get_social_knowledge(state: CompareState) -> CompareState:
    print("사회통념 지식 검색 중...")
    wrapper = WikipediaAPIWrapper()
    wiki = WikipediaQueryRun(api_wrapper=wrapper)
    result = wiki.run(state["question"])
    state["social_knowledge"] = result
    return state

# 노드: 최신 논문 검색 (Chroma vector DB)
def get_latest_medical_papers(state: CompareState) -> CompareState:
    print("최신 논문 검색 중...")
    retriever = Chroma(
        collection_name="paper_db",
        persist_directory="crawling/Europe_PMC/vector_store/chroma",
        embedding_function=embedding_model
    ).as_retriever(search_kwargs={"k": 3})

    docs = retriever.invoke(state["question"])
    state["latest_docs"] = [doc.page_content for doc in docs]
    return state

# 노드: 두 지식 비교 및 통합 응답 생성
def compare_knowledge(state: CompareState) -> CompareState:
    print("지식 비교 중...")
    chunks = "\n\n".join(state["latest_docs"])
    prompt = COMPARE_PROMPT.format(
        question=state["question"],
        social_knowledge=state["social_knowledge"],
        chunks=chunks
    )
    # stream 반환용 객체 전달
    state["prompt"] = prompt
    return state

# 그래프 구성 함수
def build_compare_graph():
    workflow = StateGraph(CompareState)
    workflow.add_node("analyze_question", analyze_question)
    workflow.add_node("get_social_knowledge", get_social_knowledge)
    workflow.add_node("get_latest_medical_papers", get_latest_medical_papers)
    workflow.add_node("compare_knowledge", compare_knowledge)

    workflow.set_entry_point("analyze_question")
    workflow.add_edge("analyze_question", "get_social_knowledge")
    workflow.add_edge("get_social_knowledge", "get_latest_medical_papers")
    workflow.add_edge("get_latest_medical_papers", "compare_knowledge")
    workflow.add_edge("compare_knowledge", END)

    return workflow.compile()
