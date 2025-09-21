import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from langgraph.graph import StateGraph, END
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI as LCChatOpenAI
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_chroma import Chroma
from langchain.agents import create_openai_functions_agent, Tool, AgentExecutor
from langchain.prompts import ChatPromptTemplate as LCChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from textwrap import dedent
from typing import TypedDict, List
from dotenv import load_dotenv
import os
import re

# 환경 설정
env_path = "C:/Aicamp/SKN13_my/13_Langchain/.env"
load_dotenv(dotenv_path=env_path)

COSINE_SIMILARITY_THRESHOLD = 0.1

CHROMA_DBS_CONFIG = [
    {"persist_directory": "crawling/vector_store/chroma", "collection_name": "antibiotic_overuse"},
    {"persist_directory": "crawling/vector_store/chroma", "collection_name": "vitamin"},
    {"persist_directory": "crawling/vector_store/others", "collection_name": "others"},
    {"persist_directory": "crawling/vector_store/vitamin_chroma", "collection_name": "vitamin"},
    {"persist_directory": "crawling/vector_store/soomin", "collection_name": "soomin"},
    {"persist_directory": "crawling/vector_store/chroma_pubmed_v3_large", "collection_name": "langchain"},
]

class CompareState(TypedDict):
    question: str
    translated_question: str
    social_knowledge: str
    latest_docs: list
    k_value: int
    final_answer: str
    db_has_data: bool
    chat_history: List[dict] # 대화 이력 추가

model = LCChatOpenAI(model="gpt-4.1", temperature=0)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

COMPARE_PROMPT = LCChatPromptTemplate.from_messages([
    ("system", dedent("""
    당신은 전문 의료 정보 요약가입니다.
    주어진 대화 이력, 사용자 질문, 사회통념 정보, 최신 의학 논문 정보를 바탕으로, 질문에 직접적으로 관련된 핵심 답변만 간결하고 명확하게 작성하세요.
    - 최신 연구 결과가 기존 사회통념과 어떻게 다른지 비교하고 종합적인 한글 요약을 작성하세요.
    - 사회통념이나 최신 논문 정보가 없다면 해당 부분은 '정보 없음'이라고 작성하세요.
    - 최신 의학 논문 데이터베이스에 관련 정보가 없는 경우, 그 사실을 명확히 언급해주세요.
    답변은 아래와 같이 3문단으로 구성해서 한글로 최종 답변해주세요.
    1. 사회통념 소개
    2. 최신 논문 소개
    3. 간단한 비교글
    """)),
    ("human", dedent("""
    대화 이력: {chat_history}
    사용자 질문: {question}
    [사회통념 정보]
    {social_knowledge}
    [최신 논문 정보]
    {latest_docs}
    {db_status_message}
    위 내용을 바탕으로 상세하고 쉽게 이해할 수 있는 한글 요약을 작성하세요.
    """))
])

@database_sync_to_async
def get_chroma_instance(collection_name, persist_dir):
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )

async def translate_question(state: CompareState) -> CompareState:
    translate_prompt = f"Translate the following Korean medical question into English: {state['question']}"
    translated = await model.ainvoke(translate_prompt)
    state["translated_question"] = translated.content
    return state

async def agent_search(state: CompareState) -> CompareState:
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=3)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

    tools = [Tool(name="Wikipedia", func=wiki_tool.run, description="사회통념 검색")]

    AGENT_PROMPT = LCChatPromptTemplate.from_messages([
        ("system", dedent("""
        너는 똑똑한 전문가야. 사용자 질문과 관련된 사회통념 정보를 Wikipedia에서 검색해야 해.
        검색된 내용을 바탕으로 **사용자 질문에 직접적으로 관련된 핵심 사회통념**을 한국어로 간결하게 요약해서 반환해야 해.
        만약 검색 결과가 없거나 관련 정보를 찾을 수 없다면, '관련 사회통념 정보 없음'이라고 알려줘.
        """)),
        ("human", "{input}\n\n{agent_scratchpad}")
    ])

    agent = create_openai_functions_agent(model, tools, prompt=AGENT_PROMPT)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    search_query = f"Search for common knowledge about: {state['translated_question']}"
    result = await executor.ainvoke({"input": search_query})

    if result and result.get("output"):
        content = result["output"].strip()
        if content.lower() in ["관련 사회통념 정보 없음", "정보 없음", "no information found"]:
            state["social_knowledge"] = "정보 없음"
        else:
            state["social_knowledge"] = content
    else:
        state["social_knowledge"] = "정보 없음"

    return state

async def get_latest_papers(state: CompareState) -> CompareState:
    all_docs = []
    found_any = False

    for db_config in CHROMA_DBS_CONFIG:
        db = await get_chroma_instance(db_config["collection_name"], db_config["persist_directory"])
        try:
            results = db.similarity_search_with_score(state["translated_question"], k=state["k_value"])
            for doc, score in results:
                if score < (1 - COSINE_SIMILARITY_THRESHOLD):
                    all_docs.append(doc.page_content)
                    found_any = True
        except Exception as e:
            print(f"❌ DB 오류: {e}")

    state["latest_docs"] = all_docs
    state["db_has_data"] = found_any
    return state

async def compare_and_summarize(state: CompareState) -> CompareState:
    social_content = state["social_knowledge"] if state["social_knowledge"] else "정보 없음"
    latest_content = "\n\n".join(state["latest_docs"]) if state["latest_docs"] else "정보 없음"

    db_status_message = "" if state["db_has_data"] else "참고: 최신 의학 논문 데이터베이스에 해당 정보가 부족합니다."

    prompt_value = COMPARE_PROMPT.format(
        chat_history="\n".join([f"{msg['type']}: {msg['content']}" for msg in state['chat_history']]),
        question=state["question"],
        social_knowledge=social_content,
        latest_docs=latest_content,
        db_status_message=db_status_message
    )
    response = await model.ainvoke(prompt_value)
    final = response.content

    # 문단 구분을 위해 번호 기준으로 개행 처리
    final = re.sub(r"2.", "\n\n2.", final)
    final = re.sub(r"3.", "\n\n3.", final)

    state["final_answer"] = final
    return state

async def build_compare_graph():
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

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        from .models import ChatSession
        self.user = self.scope["user"]
        if not self.user.is_authenticated:
            await self.close()
            return

        session_id = self.scope['url_route']['kwargs']['session_id']
        self.room_group_name = f"chat_{session_id}"

        try:
            self.session = await database_sync_to_async(
                ChatSession.objects.get
            )(id=session_id, user=self.user)
        except ChatSession.DoesNotExist:
            await self.close()
            return

        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        from .models import ChatMessage
        text_data_json = json.loads(text_data)
        user_message = text_data_json['message']

        # 사용자 메시지 저장 및 전송
        await self.save_message(self.session, 'user', user_message)
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': user_message,
                'message_type': 'user',
                'username': self.user.username
            }
        )

        # 과거 대화 이력 불러오기
        chat_history_objects = await database_sync_to_async(list)(
            self.session.messages.order_by('created_at').all()
        )
        chat_history = []
        for msg_obj in chat_history_objects:
            if msg_obj.message_type == 'user':
                chat_history.append(HumanMessage(content=msg_obj.content))
            elif msg_obj.message_type == 'assistant':
                chat_history.append(AIMessage(content=msg_obj.content))

        try:
            graph = await build_compare_graph()
            state = {
                "question": user_message,
                "k_value": 5,
                "chat_history": chat_history # 대화 이력 전달
            }

            # LangGraph 스트리밍 및 상태 업데이트
            async for event in graph.astream_events(state, version="v1"):
                kind = event["event"]
                if kind == "on_chain_start":
                    if event["name"] == "translate_question":
                        await self.send_status_update("질문을 번역하는 중...")
                    elif event["name"] == "agent_search":
                        await self.send_status_update("사회통념 정보를 검색하는 중...")
                    elif event["name"] == "get_latest_papers":
                        await self.send_status_update("최신 의학 논문을 검색하는 중...")
                    elif event["name"] == "compare_and_summarize":
                        await self.send_status_update("정보를 종합하여 답변을 생성하는 중...")
                elif kind == "on_chain_end":
                    if event["name"] == "compare_and_summarize":
                        # 최종 답변이 생성되면 상태 메시지 숨김
                        await self.send_status_update("") 

            # 최종 상태에서 답변 가져오기
            final_state = await graph.ainvoke(state)
            ai_response = final_state.get("final_answer", "답변을 생성하지 못했습니다.")

            # AI 답변 저장 및 전송
            await self.save_message(self.session, 'assistant', ai_response)
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'chat_message',
                    'message': ai_response,
                    'message_type': 'assistant',
                    'username': 'AI 어시스턴트'
                }
            )

        except Exception as e:
            error_message = f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"
            await self.save_message(self.session, 'assistant', error_message)
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'chat_message',
                    'message': error_message,
                    'message_type': 'assistant',
                    'username': 'AI 어시스턴트'
                }
            )

    async def send_status_update(self, status_message):
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': status_message,
                'message_type': 'status',
                'username': 'AI 어시스턴트'
            }
        )

    @database_sync_to_async
    def save_message(self, session, message_type, content):
        from .models import ChatMessage
        return ChatMessage.objects.create(
            session=session,
            message_type=message_type,
            content=content
        )
