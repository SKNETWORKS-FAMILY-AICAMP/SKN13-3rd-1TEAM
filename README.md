![image](https://github.com/user-attachments/assets/37d717b3-9d0d-4d57-bed3-b97e3f0d2f98)


# Project 3

## 프로젝트 개요

- 인터넷에는 정확하지 않거나 과장된 건강 정보가 많아 사용자들이 스스로 정확한 정보를 판단하기 어렵기 때문에, **신뢰 가능한 비교 시스템**이 필요합니다.
- 최신 의학 논문과 사회적 통념(위키백과 등)을 자동으로 수집·정제·저장하고, 사용자의 의료 질문에 대해 논문 근거와 일반적 상식의 차이를 비교·요약해주는 한국어 챗봇 서비스입니다
- **일반 상식(Wikipedia, 블로그, 지식인 등등) + 최신 논문 기반(PubMed, Europe PMC)** 정보를 통합해 **정확하고 풍부한 한글 응답**을 생성합니다
- 헬스/의학 외에도 법률, 교육, 정책 등 다양한 분야로 확장 가능합니다

## **핵심 기술:**

- 대규모 논문 크롤링 및 벡터DB 구축
    - RAG 기반 실시간 검색·요약
    - 위키백과에서 사회통념 정보 추출
    - Streamlit UI 및 LangGraph로 워크플로우 자동화
- **적용 분야:**
    - 최신 의학정보 검증
        - 건강정보 팩트체크
        - 의료 상담 보조

## ✅ 기술 스택 요약

| 컴포넌트 | 기술 |
| --- | --- |
| LLM | `ChatOpenAI` 또는 `gpt-4.1` |
| 번역 | `PromptTemplate` + `LLMChain` |
| 위키 검색 | `WikipediaQueryRun` |
| 논문 검색 | `Chroma`, `OpenAIEmbeddings`, `retriever` |
| 요약 출력 | `LLMChain` + `PromptTemplate` |
| Pipeline 통합 | `LangChain` |

# System Architecture

![image.png](attachment:0ef143c4-f6b9-4c04-8984-b34405714aca:image.png)

# Datasets

1. PubMed(https://pubmed.ncbi.nlm.nih.gov/)
2. Europe PMC(https://europepmc.org/)
3. MedRxiv(https://www.medrxiv.org/)

# ✅ 전체 구조: 파이프라인 흐름 요약

---

```
[1] 사용자의 한글 질의 입력
      ↓
[2] 질의를 영어로 번역 (LLM 또는 Translator Chain)
      ↓
[3] 영어 질의로 Wikipedia 검색 (Agent Tool / Tool Calling)
      ↓
[4] 영어 질의로 벡터DB 검색 (논문 정보 검색: 최신 RAG DB)
      ↓
[5] Wikipedia + 논문 결과를 기반으로 LLM이 한국어로 통합 요약 생성
      ↓
[6] 최종 한글 답변 출력

```

---

## 🔍 각 단계 상세 설명

### 1. **사용자 질의 입력 (예: "비타민 C는 피부에 좋은가요?")**

- 한글로 질문을 입력받음.
- 자연어 기반 자유 질의 가능.
- 

---

### 2. **질의 영어 번역 (LLM 또는 TranslatorChain)**

- 이유: Wikipedia나 논문 검색은 대부분 영어 기반.

---

### 3. **Wikipedia 검색 (일반상식)**

- LangChain의 `Tool`, `Agent`를 활용
- `WikipediaAPIWrapper` 과  `WikipediaQueryRun`을 툴로 등록.

---

### 4. **RAG 벡터 DB 검색 (최신 논문 기반)**

- 사전 수집한 논문(PubMed 등)을 임베딩하여 Chroma 등 벡터DB에 저장.
- LangChain의 `retriever`로 검색
- 예: retriever_output = retriever.get_relevant_documents(translated_query)

---

### 5. **LLM이 Wikipedia + 논문 기반 한국어 요약 생성**

- Prompt에서 `위키 지식 + 논문 내용`을 통합해 한글로 생성 지시.

---

### 6. **한글 답변 출력**

- 최종 `final_answer`를 사용자에게 출력.
- csv 응답으로 제공, 그리고 vector store에 바로 저장한 것을 이용해 챗봇 UI에 렌더링 가능.

---

## 🧠 이 구조의 장점

| 기능 | 설명 |
| --- | --- |
| 다단계 근거 기반 | 위키 + 논문 → 신뢰성 있는 복합 답변 |
| LLM 보완 | 번역 + 요약에 LLM을 사용하되, 검색 기반 지식으로 보완 |
| 최신 정보 대응 | 논문 DB를 주기적으로 업데이트하면 실시간성 확보 |
| 한글 대응 | 유저 입출력은 전부 한글 → 사용성 뛰어남 |

---

# 시연



## “상식 수준의 의학정보 vs 최신 의학 논문 기반 정보 비교” 를 자동으로 해주는 도구

일반 상식과 최신 논문의 의학정보를 비교함으로 최신 일반 상식의 과학적 오류를 알아 볼 수 있는 챗 봇 시스템입니다.

기존 gpt 4.1 모델에 의학 논문데이터를 보강했습니다.

### 1. 데이터 수집
- PMC , Europe PMC API 크롤링 `건강 상식` 키워드 논문 크롤링
- medirxiv `medicine` 키워드 논문 크롤링
- VectorDB 저장 (ChromaDB)
- AI Model : gpt-4.1
- Embedding Model : text-embedding-3-large



![image](https://github.com/user-attachments/assets/8361f1dd-c0b5-4cd2-ba96-27e5fe9b9714)

![image](https://github.com/user-attachments/assets/405fa90a-910e-48d3-8009-a5736d553923)
