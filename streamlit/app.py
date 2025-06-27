import streamlit as st
import urllib.parse
from sqlalchemy import create_engine
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from modules.compare_agent_module import build_compare_graph

# ----------------------------
# DB 연결 (SQLite 또는 MySQL 사용 가능)
# ----------------------------
engine = create_engine("sqlite:///chat_history.sqlite")  # MySQL 사용 시 이 부분 수정

# ----------------------------
# Session ID 입력 + 시작 버튼
# ----------------------------
st.set_page_config(page_title="RAG 기반 의학 지식 비교", layout="wide")
st.title("사회통념 vs 최신 의학 논문 비교 서비스")

# 초기 세션 상태 설정
if "session_id" not in st.session_state:
    st.session_state["session_id"] = ""

# 사이드바: 세션 ID 입력
session_id_input = st.sidebar.text_input("Session ID", value=st.session_state["session_id"], placeholder="대화 ID를 입력하세요")

# 시작 버튼을 눌러야 세션 시작
if st.sidebar.button("시작하기"):
    if not session_id_input.strip():
        st.warning("Session ID를 입력하세요.")
        st.stop()
    st.session_state["session_id"] = session_id_input.strip()
    st.rerun()

# 세션 ID가 없다면 중단
if not st.session_state["session_id"]:
    st.warning("왼쪽에 Session ID를 입력해야 합니다.")
    st.stop()

# ----------------------------
# 세션 기반 DB 히스토리 객체 생성
# ----------------------------
history = SQLChatMessageHistory(session_id=st.session_state["session_id"], connection=engine)

# 이전 대화 출력
for msg in history.messages:
    role = "user" if isinstance(msg, HumanMessage) else "ai"
    with st.chat_message(role):
        st.write(msg.content)

# ----------------------------
# 질문 입력 + LangGraph 실행
# ----------------------------
question = st.chat_input("궁금한 의학 질문을 입력하세요")

if question:
    # 1. 사용자 질문 저장 + 출력
    history.add_user_message(question)
    with st.chat_message("user"):
        st.write(question)

    # 2. LangGraph 실행 (Wikipedia + 최신 논문 + 프롬프트 조립까지)
    graph = build_compare_graph()
    state = {
        "question": question,
        "social_knowledge": "",
        "latest_docs": [],
        "final_answer": "",
        "prompt": None  # 프롬프트가 여기로 전달됨
    }
    state = graph.invoke(state)

    # 3. 스트리밍 응답 출력
    with st.chat_message("ai"):
        message_placeholder = st.empty()
        model = ChatOpenAI(model="gpt-4.1", streaming=True)
        full_response = ""
        for chunk in model.stream(state["prompt"]):
            full_response += chunk.content
            message_placeholder.write(full_response)

    # 4. 응답 저장
    history.add_ai_message(full_response)
