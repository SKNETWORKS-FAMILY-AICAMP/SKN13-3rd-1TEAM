# SKN13-3RD-1TEAM: 상식 vs 최신 의학 논문 기반 챗봇 & RAG 평가

## 팀원 소개
<table align=center>
  <tbody>
   <tr>
      <td align=center><b>남궁건우</b></td>
      <td align=center><b>모지호</b></td>
      <td align=center><b>민경재</b></td>
      <td align=center><b>안수민</b></td>
    </tr>
    <tr>
      <td align="center">
          <img alt="Image" src="images/gw.jpeg" width="200px;" alt="남궁건우"/>
      </td>
      <td align="center">
          <img alt="Image" src="images/jh.jpeg" width="200px;" alt="모지호"/>
      </td>
      <td align="center">
        <img alt="Image" src="images/gj.jpeg" width="200px;" alt="민경재" />
      </td>
      <td align="center">
        <img alt="Image" src="images/sm.jpg" width="200px;" alt="안수민"/>
      </td>
    </tr>
    <tr>
       <td align="center">
       <a href="https://github.com/NGGW519">
         <img src="https://img.shields.io/badge/GitHub-NGGW519-FEFFAB?logo=github" alt="남궁건우 GitHub"/>
       </a>
       </td>
       <td align="center">
       <a href="https://github.com/mojiho7">
         <img src="https://img.shields.io/badge/GitHub-mojiho-B9FF92?logo=github" alt="모지호 GitHub"/>
       </a>
       </td>
       <td align="center">
       <a href="https://github.com/rudwo524">
         <img src="https://img.shields.io/badge/GitHub-rudwo524-FFAFB0?logo=github" alt="민경재 GitHub"/>
       </a>
       </td>
       <td align="center">
       <a href="https://github.com/tnalsdk111">
         <img src="https://img.shields.io/badge/GitHub-tnalsdk111-BD9FFF?logo=github" alt="안수민 GitHub"/>
       </a>
       </td>
    </tr>
  </tbody>
</table>
<br>
<br/><br/>


## 프로젝트 개요

- **목표:**  
  일반 상식과 최신 의학 논문 정보를 비교하여, 최신 상식의 과학적 오류를 자동으로 진단하는 챗봇 및 평가 시스템 구축
- **핵심:**  
  - GPT-4.1 기반 LLM + 최신 논문 벡터스토어(ChromaDB)
  - RAG 파이프라인 및 RAGAS 평가 자동화

---

## 산출물

- **데이터 수집 및 전처리 문서**
  - `crawling/report.ipynb` : 데이터 수집, 전처리, 메타데이터 정제, 중복 처리 등 상세 설명
- **시스템 아키텍처 구성도**
  - Mermaid Diagram
    ```mermaid
    flowchart TD
        subgraph User
            U["사용자\n질문 입력"]
        end
        subgraph WebApp
            ST["Streamlit 챗봇(app.py)"]
            G["LLM 파이프라인(main.ipynb, module.py)"]
        end
        subgraph RAG
            RET["Retriever\n(ChromaDB)"]
            LLM["LLM (GPT-4.1)"]
            RAGAS["RAGAS 평가(RAGAS/ragas_eval_all.py)"]
        end
        subgraph Data
            CR["논문 크롤링/전처리\n(crawling/module.py, crawling.ipynb)"]
            VS["Vector Store\n(ChromaDB, crawling/vector_store/)"]
        end
    
        U -->|질문| ST
        ST --> G
        G -->|질문/프롬프트| RET
        RET -->|관련 논문 context| LLM
        LLM -->|답변| ST
        CR --> VS
        RET --> VS
        G --> RAGAS
        RAGAS --> VS
        RAGAS --> LLM
        RAGAS -->|평가결과| ST
    ``` 
- **수집한 데이터셋, 전처리한 데이터셋**
  - 원본/전처리 데이터: `crawling/vector_store/` 내 ChromaDB 벡터스토어(논문 임베딩)
- **구현 코드**
    - **데이터 수집, 전처리 코드**: `crawling/crawling.ipynb`, `crawling/module.py`
    - **RAG Application 구현 코드**: `app.py`, `main.ipynb`, `module.py`, `RAGAS/ragas_eval_all.py` 등
- **RAG 평가 score 해석 안내**
    - 일반상식이나 의학논문을 찾은 결과, 관련 없는 내용과 관련 있는 내용을 정확히 구분합니다. 
    - 관련 없는 내용의 경우 문서를 가져오지 않으므로, 평가 점수(예: context recall 등)가 0점이 나올 수 있습니다.
    - 이는 시스템이 불필요한 문서를 근거로 사용하지 않도록 설계된 결과입니다.
    - 또한, 정확한 관련 문서만을 검색하기 위해 `COSINE_SIMILARITY_THRESHOLD = 0.1`로 설정하여, 임계값보다 유사도가 낮은 문서는 검색 결과에서 제외합니다.
- **데이터 전처리 과정 설명**
    - 본 프로젝트는 논문 크롤링 시 각 논문의 URL에서 필요한 부분(주로 abstract 등)만을 HTML 파싱하여 곧바로 임베딩 벡터로 저장합니다.
    - 논문 원문의 의미와 구조를 최대한 보존하기 위해, 불필요한 전처리는 최소화하였습니다.
    - AI 관점에서의 전처리 단계로는 다음이 포함됩니다:
        - HTML 파싱을 통한 논문 본문/abstract 추출
        - 불필요한 섹션(예: references, funding 등) 제거
        - 최소 길이(200자 등) 미만 텍스트를 가진 논문은, 신뢰성 부족으로 필터링
        - 논문 메타데이터(pmcid, title, doi, 저자 등) 정제 및 부착
    - 이로써, 논문 내용의 신뢰성과 활용도를 높이면서도, RAG 기반 검색/질의응답에 최적화된 데이터셋을 구축하였습니다.

---

## 주요 기능

1. **의학 질의응답 챗봇**
   - 사용자의 질문을 한글로 입력 → 영어 번역 → 사회통념(위키피디아) & 최신 논문 검색 → 비교 요약
   - Streamlit 기반 웹 UI (`app.py`)

2. **최신 논문 기반 RAG 파이프라인**
   - Europe PMC, PubMed, MedRxiv 등에서 논문 크롤링
   - 논문 텍스트를 임베딩 후 ChromaDB에 저장
   - LLM이 논문 context 기반으로 답변 생성

3. **RAGAS 평가 자동화**
   - 5개 vector store에서 context 추출, LLM으로 50쌍의 질의/응답 쌍 생성
   - RAGAS로 정량적 평가(정밀도, 재현율, faithfulness, relevancy 등)
   - 평가 결과 csv로 저장

---

## 프로젝트 구조

```
SKN13-3RD-1TEAM/
│
├─ app.py                # Streamlit 기반 챗봇 서비스
├─ main.ipynb            # LLM 서비스 및 평가 예시
├─ module.py             # 파이프라인/그래프/노드 정의
├─ crawling/
│   ├─ crawling.ipynb    # 논문 크롤링 및 벡터스토어 저장
│   ├─ module.py         # 크롤링/임베딩/저장 함수
│   ├─ report.ipynb      # 데이터 수집/전처리/정제 문서
│   └─ vector_store/     # ChromaDB 벡터스토어(논문 임베딩)
│       ├─ chroma/
│       ├─ chroma_pubmed_v3_large/
│       ├─ others/
│       ├─ soomin/
│       └─ vitamin_chroma/
├─ RAGAS/
│   ├─ ragas_eval_all.py         # RAGAS 평가 자동화 스크립트
│   ├─ ragas_eval_result.csv     # 평가용 질문/정답/context/답변 등 전체 데이터셋
│   ├─ ragas_score.csv           # 각 샘플별 RAGAS 평가 점수
│   ├─ view_ragas_score.ipynb    # 평가 점수 csv 확인용 노트북
│   └─ check_all_vectorstores.py # 벡터스토어 저장 점검 스크립트
├─ chat_history/         # 대화 로그/DB
├─ README.md             # 프로젝트 설명서
└─ ...
```

---

## 실행 방법

### 1. 환경 준비

```bash
conda activate lang_env
pip install -r requirements.txt  # 필요시
```

### 2. 데이터 수집 및 벡터스토어 구축

- 새 데이터 수집： `crawling/crawling.ipynb` 또는 `crawling/module.py` 실행
- 기존 데이터베이스 사용： `main.ipynb` 실행 → zip 파일 자동 다운로드 및 압축 해제

### 3. 챗봇 서비스 실행

```bash
streamlit run app.py
```
- 웹 UI에서 한글로 질문 → 사회통념/최신 논문 비교 답변

### 4. RAGAS 평가 자동화

```bash
python RAGAS/ragas_eval_all.py
```
- 5개 store에서 context 추출, 50쌍 질의/응답 생성, RAGAS 평가
- 결과: `RAGAS/ragas_eval_result.csv`, `RAGAS/ragas_score.csv`

### 5. 평가 결과 확인

- `RAGAS/view_ragas_score.ipynb`에서 평가 점수 csv 바로 확인

---

## 주요 파일 설명

- **app.py**: Streamlit 기반 챗봇, LLM+RAG+사회통념 비교
- **main.ipynb**: 파이프라인 예시, 평가 흐름
- **module.py**: LLM 파이프라인, 그래프, 노드 정의
- **crawling/module.py**: 논문 크롤링, 임베딩, 벡터스토어 저장 함수
- **crawling/report.ipynb**: 데이터 수집, 전처리, 정제, 메타데이터 관리 문서
- **RAGAS/ragas_eval_all.py**: RAGAS 평가 자동화(50쌍 질의/응답)
- **RAGAS/ragas_eval_result.csv**: 평가용 전체 데이터셋
- **RAGAS/ragas_score.csv**: 각 샘플별 RAGAS 평가 점수

---

## 참고/인용

- PubMed: https://pubmed.ncbi.nlm.nih.gov/
- Europe PMC: https://europepmc.org/
- MedRxiv: https://www.medrxiv.org/

---

**최신 논문 기반의 신뢰도 높은 의학 챗봇 및 RAG 평가 자동화!**
