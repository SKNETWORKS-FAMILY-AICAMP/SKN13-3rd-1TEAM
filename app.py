import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI as LCChatOpenAI
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate as LCChatPromptTemplate
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain.agents import create_openai_functions_agent, Tool, AgentExecutor
from textwrap import dedent
from typing import TypedDict
import os
from dotenv import load_dotenv

# ✅ 환경 설정
env_path = "C:/Aicamp/SKN13_my/13_Langchain/.env"
load_dotenv(dotenv_path=env_path)

# ✅ 모델 및 설정
model = LCChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
COSINE_SIMILARITY_THRESHOLD = 0.1

CHROMA_DBS_CONFIG = [
    {"persist_directory": "Europe_PMC/vector_store/chroma", "collection_name": "antibiotic_overuse"},
    {"persist_directory": "Europe_PMC/vector_store/others", "collection_name": "others"},
    {"persist_directory": "Europe_PMC/vector_store/vitamin_chroma", "collection_name": "vitamin"}
]

class CompareState(TypedDict):
    question: str
    translated_question: str
    social_knowledge: str
    latest_docs: list
    k_value: int
    final_answer: str
    db_has_data: bool

COMPARE_PROMPT = LCChatPromptTemplate.from_messages([
    ("system", dedent("""
    당신은 전문 의료 정보 요약가입니다.
    아래 사용자 질문, 사회통념 정보, 최신 의학 논문 정보를 바탕으로,
    최신 연구 결과가 기존 사회통념과 어떻게 다른지 비교하고 종합적인 한글 요약을 작성하세요.
    """)),
    ("human", dedent("""
    사용자 질문: {question}

    [사회통념 정보]
    {social_knowledge}

    [최신 논문 정보]
    {latest_docs}

    {db_status_message}
    """))
])

# ✅ 노드 정의
def translate_question(state: CompareState) -> CompareState:
    translate_prompt = f"Translate the following Korean medical question into English: {state['question']}"
    translated = model.invoke(translate_prompt).content
    state["translated_question"] = translated
    return state

def agent_search(state: CompareState) -> CompareState:
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=3)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
    tools = [Tool(name="Wikipedia", func=wiki_tool.run, description="사회통념 검색")]
    AGENT_PROMPT = LCChatPromptTemplate.from_messages([
        ("system", dedent("""
        너는 똑똑한 전문가야. 사용자 질문과 관련된 사회통념 정보를 Wikipedia에서 검색하고,
        관련 정보를 한국어로 간결하게 요약해서 반환해야 해.
        없으면 '관련 사회통념 정보 없음'이라고 알려줘.
        """)),
        ("human", "{input}\n\n{agent_scratchpad}")
    ])
    agent = create_openai_functions_agent(model, tools, prompt=AGENT_PROMPT)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    search_query = f"Search for common knowledge about: {state['translated_question']}"
    try:
        result = executor.invoke({"input": search_query})
        if result and result.get("output"):
            content = result["output"].strip()
            if content.lower() in ["관련 사회통념 정보 없음", "정보 없음"]:
                state["social_knowledge"] = "정보 없음"
            else:
                state["social_knowledge"] = content
        else:
            state["social_knowledge"] = "정보 없음"
    except:
        state["social_knowledge"] = "정보 없음"
    return state

def get_latest_papers(state: CompareState) -> CompareState:
    all_docs = []
    found_data = False
    for db_config in CHROMA_DBS_CONFIG:
        try:
            db = Chroma(
                collection_name=db_config["collection_name"],
                persist_directory=db_config["persist_directory"],
                embedding_function=embedding_model
            )
            results = db.similarity_search_with_score(state["translated_question"], k=state["k_value"])
            for doc, score in results:
                if score < (1 - COSINE_SIMILARITY_THRESHOLD):
                    all_docs.append(doc.page_content)
                    found_data = True
        except:
            continue
    state["latest_docs"] = all_docs
    state["db_has_data"] = found_data
    return state

def compare_and_summarize(state: CompareState) -> CompareState:
    social_content = state["social_knowledge"] or "정보 없음"
    latest_content = "\n\n".join(state["latest_docs"]) or "정보 없음"
    db_status_msg = "" if state["db_has_data"] else "참고: 최신 의학 논문 데이터베이스에 정보가 부족하거나 존재하지 않습니다."
    prompt_value = COMPARE_PROMPT.format(
        question=state["question"],
        social_knowledge=social_content,
        latest_docs=latest_content,
        db_status_message=db_status_msg
    )
    chunks = model.stream(prompt_value)
    final_content = ""
    placeholder = st.empty()
    for chunk in chunks:
        final_content += chunk.content
        placeholder.markdown(final_content + "▌")
    placeholder.markdown(final_content)
    state["final_answer"] = final_content
    return state

def build_compare_graph():
    workflow = StateGraph(CompareState)
    workflow.add_node("translate_question", translate_question)
    workflow.add_node("agent_search", agent_search)
    workflow.add_node("get_latest_papers", get_latest_papers)
    workflow.add_node("compare_and_summarize", compare_and_summarize)

    workflow.set_entry_point("translate_question")
    workflow.add_edge("translate_question", "agent_search")
    workflow.add_edge("agent_search", "get_latest_papers")
    workflow.add_edge("get_latest_papers", "compare_and_summarize")
    workflow.add_edge("compare_and_summarize", END)

    return workflow.compile()

# ================================
# Streamlit 앱
# ================================
st.title("🩺 의학 논문 비교 요약 챗봇")

if "history" not in st.session_state:
    st.session_state.history = []

# 과거 대화 먼저 출력
for msg in st.session_state.history:
    if msg["type"] == "human":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

user_input = st.chat_input("궁금한 의학 질문을 입력하세요!")

if user_input:
    # 질문 먼저 표시 (추가 X)
    with st.chat_message("user"):
        st.markdown(user_input)

    graph = build_compare_graph()

    initial_state = {
        "question": user_input,
        "translated_question": "",
        "social_knowledge": "",
        "latest_docs": [],
        "k_value": 5,
        "final_answer": "",
        "db_has_data": False
    }

    # 응답 표시 및 streaming
    with st.chat_message("assistant"):
        result = graph.invoke(initial_state)
        final_answer = result["final_answer"]

    # 대화 history에 질문과 응답을 한 번에 추가
    st.session_state.history.append({"type": "human", "content": user_input})
    st.session_state.history.append({"type": "assistant", "content": final_answer})
