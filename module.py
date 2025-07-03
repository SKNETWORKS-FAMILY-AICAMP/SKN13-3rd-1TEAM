from langgraph.graph import StateGraph, END
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_chroma import Chroma
from langchain.agents import create_openai_functions_agent, Tool, AgentExecutor
from langchain.prompts import ChatPromptTemplate as LCChatPromptTemplate
from textwrap import dedent
from typing import TypedDict
from IPython.display import Image
from dotenv import load_dotenv
import os

# 환경 설정
load_dotenv()

COSINE_SIMILARITY_THRESHOLD = 0.1


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))  # 루트 경로
CHROMA_DBS_CONFIG = [
    {
        "persist_directory": os.path.join(root_dir, "crawling/vector_store/chroma"),
        "collection_name": "antibiotic_overuse"
    },
    {
        "persist_directory": os.path.join(root_dir, "crawling/vector_store/chroma"),
        "collection_name": "vitamin"
    },
    {
        "persist_directory": os.path.join(root_dir, "crawling/vector_store/others"),
        "collection_name": "others"
    },
    {
        "persist_directory": os.path.join(root_dir, "crawling/vector_store/vitamin_chroma"),
        "collection_name": "vitamin"
    },
    {
        "persist_directory": os.path.join(root_dir, "crawling/vector_store/soomin"),
        "collection_name": "soomin"
    },
    {
        "persist_directory": os.path.join(root_dir, "crawling/vector_store/chroma_pubmed_v3_large"),
        "collection_name": "langchain"
    }
]



# ✅ 상태 정의
class CompareState(TypedDict):
    """
    비교 및 요약 작업을 위한 상태를 정의합니다.
    """
    question: str
    translated_question: str
    social_knowledge: str
    latest_docs: list
    k_value: int
    final_answer: str
    db_has_data: bool 

# ✅ 모델 정의
model = ChatOpenAI(model="gpt-4.1", temperature=0)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# ✅ 프롬프트 정의
COMPARE_PROMPT = LCChatPromptTemplate.from_messages([
    ("system", dedent("""
    당신은 전문 의료 정보 요약가입니다.
    아래 사용자 질문, 사회통념 정보, 최신 의학 논문 정보를 바탕으로, 질문에 직접적으로 관련된 핵심 답변만 간결하고 명확하게 작성하세요.
    - 최신 연구 결과가 기존 사회통념과 어떻게 다른지 비교하고 종합적인 한글 요약을 작성하세요.
    - 사회통념이나 최신 논문 정보가 없다면 해당 부분은 '정보 없음'이라고 작성하세요.
    - 최신 의학 논문 데이터베이스에 관련 정보가 없는 경우, 그 사실을 명확히 언급해주세요.

    
    답변은 아래와 같이 3문단으로 구성해서 한글로 최종 답변해주세요.

    1. 사회통념 소개
    2. 최신 논문 소개
    3. 간단한 비교글
    """)),
    ("human", dedent("""
    사용자 질문: {question}

    [사회통념 정보]
    {social_knowledge}

    [최신 논문 정보]
    {latest_docs}

    {db_status_message} 
    
    위 내용을 바탕으로 상세하고 쉽게 이해할 수 있는 한글 요약을 작성하세요.
    """
    ))
])

# ✅ 노드 정의

def translate_question(state: CompareState) -> CompareState:
    """
    사용자 질문을 영어로 번역하는 노드.
    """
    print(f"\n--- translate_question ---")
    translate_prompt = f"Translate the following Korean medical question into English: {state['question']}"
    translated = model.invoke(translate_prompt).content
    state["translated_question"] = translated
    print(f"원본 질문: {state['question']}")
    print(f"번역된 질문: {state['translated_question']}")
    return state

def agent_search(state: CompareState) -> CompareState:
    """
    Wikipedia를 통해 사회 통념 정보를 검색하는 Agent 노드.
    """
    print(f"\n--- agent_search ---")
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=3) # 검색 결과 개수 늘려볼 수 있음
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

    tools = [Tool(name="Wikipedia", func=wiki_tool.run, description="사회통념 검색")]

    # 에이전트 프롬프트 개선: 더 명확하게 검색 후 요약하도록 지시
    AGENT_PROMPT = LCChatPromptTemplate.from_messages([
        ("system", dedent("""
        너는 똑똑한 전문가야. 사용자 질문과 관련된 사회통념 정보를 Wikipedia에서 검색해야 해.
        검색된 내용을 바탕으로 **사용자 질문에 직접적으로 관련된 핵심 사회통념**을 한국어로 간결하게 요약해서 반환해야 해.
        만약 검색 결과가 없거나 관련 정보를 찾을 수 없다면, '관련 사회통념 정보 없음'이라고 명확히 알려줘.
        """)),
        ("human", "{input}\n\n{agent_scratchpad}")
    ])

    agent = create_openai_functions_agent(model, tools, prompt=AGENT_PROMPT)
   
    executor = AgentExecutor(agent=agent, tools=tools, verbose=False) 

    search_query = f"Search for common knowledge about: {state['translated_question']}"
    print(f"Wikipedia 검색 쿼리: {search_query}")

    try:
        result = executor.invoke({"input": search_query})
        
        # 에이전트의 'output'을 바로 사용하기 전에 유효성 검사
        if result and result.get("output"):
            social_knowledge_found = result["output"].strip()
            # LLM이 '정보 없음'이라고 반환하는 경우도 처리
            if social_knowledge_found.lower() in ["관련 사회통념 정보 없음", "정보 없음", "no information found", "nothing found", "no relevant information"]:
                state["social_knowledge"] = "정보 없음"
                print(f"Wikipedia 검색 결과: {social_knowledge_found} (LLM이 정보 없음을 반환)")
            else:
                state["social_knowledge"] = social_knowledge_found
                print(f"Wikipedia 검색 성공! 사회통념: {state['social_knowledge']}")
        else:
            state["social_knowledge"] = "정보 없음"
            print("Wikipedia 검색 실패 또는 LLM이 유효한 output을 반환하지 않음.")
    except Exception as e:
        state["social_knowledge"] = "정보 없음"
        print(f"Wikipedia Agent 실행 중 오류 발생: {e}")
    
    return state

def get_latest_papers(state: CompareState) -> CompareState:
    """
    여러 Chroma Vector Store에서 코사인 유사도 임계값을 적용하여 최신 논문을 검색하는 노드.
    """
    print(f"\n--- get_latest_papers ---")
    all_retrieved_docs_contents = []
    found_any_data = False

    for db_config in CHROMA_DBS_CONFIG:
        current_persist_dir = db_config["persist_directory"]
        current_collection_name = db_config["collection_name"]
        
        # print(f"🔎 {current_persist_dir}/{current_collection_name} 에서 문서 검색 중 (k={state['k_value']}, 유사도 임계값={COSINE_SIMILARITY_THRESHOLD})...")
        try:
            current_db = Chroma(
                collection_name=current_collection_name,
                persist_directory=current_persist_dir,
                embedding_function=embedding_model
            )
            
            search_results_with_scores = current_db.similarity_search_with_score(
                state["translated_question"],
                k=state["k_value"]
            )
            
            filtered_docs_count = 0
            for doc, score in search_results_with_scores:
                if score < (1 - COSINE_SIMILARITY_THRESHOLD): 
                    all_retrieved_docs_contents.append(doc.page_content)
                    found_any_data = True
                    filtered_docs_count += 1
            #         print(f"   - 문서 발견 (점수: {score:.4f}, 내용 일부: {doc.page_content[:50]}...)")
            #     else:
            #         print(f"   - 문서 필터링됨 (점수: {score:.4f}, 임계값 {1 - COSINE_SIMILARITY_THRESHOLD:.4f} 초과)")

            # if filtered_docs_count > 0:
            #     print(f"✅ {current_persist_dir}/{current_collection_name} 에서 임계값 통과 문서 {filtered_docs_count}개 발견.")
            # else:
            #     print(f"❗ {current_persist_dir}/{current_collection_name} 에서 임계값 통과 문서 발견 안됨.")

        except Exception as e:
            print(f"❌ {current_persist_dir}/{current_collection_name} 로드 또는 검색 중 오류 발생: {e}")
            continue

    state["latest_docs"] = all_retrieved_docs_contents
    state["db_has_data"] = found_any_data # 하나라도 유효한 문서가 발견되면 True

    if not found_any_data:
        print("❗ 모든 Chroma DB에서 해당 질문에 대한 관련 문서가 (임계값 기준) 발견되지 않았습니다.")
    
    return state


def compare_and_summarize(state: CompareState) -> CompareState:
    """
    사회 통념과 최신 논문 정보를 비교하여 최종 요약을 생성하는 노드.
    """
    print(f"\n--- compare_and_summarize ---")
    social_knowledge_content = state["social_knowledge"] if state["social_knowledge"] else "정보 없음"
    latest_docs_content = "\n\n".join(state["latest_docs"]) if state["latest_docs"] else "정보 없음"

    db_status_message = ""

    if not state["db_has_data"]:
        db_status_message = "참고: 최신 의학 논문 데이터베이스에 해당 주제에 대한 관련 정보가 현재 부족하거나 존재하지 않습니다."
    else:
        db_status_message = ""

    prompt_value = COMPARE_PROMPT.format(
        question=state["question"],
        social_knowledge=social_knowledge_content,
        latest_docs=latest_docs_content,
        db_status_message=db_status_message 
    )
    response = model.invoke(prompt_value)
    state["final_answer"] = response.content
    print(f"최종 요약 생성 완료.")
    return state


def build_compare_graph():
    """
    LangGraph 워크플로우를 구성합니다.
    """
    workflow = StateGraph(CompareState)

    # 노드 추가
    workflow.add_node("translate_question", translate_question)
    workflow.add_node("agent_search", agent_search)
    workflow.add_node("get_latest_papers", get_latest_papers)
    workflow.add_node("compare_and_summarize", compare_and_summarize)

    # 시작점 설정
    workflow.set_entry_point("translate_question")
    workflow.add_edge("translate_question", "agent_search")
    workflow.add_edge("agent_search", "get_latest_papers")
    workflow.add_edge("get_latest_papers", "compare_and_summarize")
    workflow.add_edge("compare_and_summarize", END)

    return workflow.compile()