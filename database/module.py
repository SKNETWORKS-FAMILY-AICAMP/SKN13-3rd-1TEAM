import os
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import tqdm

# ======================================
# ✅ 환경 설정
# ======================================

load_dotenv(dotenv_path="C:/Aicamp/SKN13_my/13_Langchain/.env")

PERSIST_DIR = "vector_store/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-large")
LLM = ChatOpenAI(model="gpt-4o", temperature=0.3)

# ======================================
# ✅ 유틸리티 함수
# ======================================

def safe_request_get(url, params=None, headers=None, retries=5, backoff=2, timeout=10):
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"요청 실패 ({attempt + 1}/{retries}): {e}")
            time.sleep(backoff * (attempt + 1))
    raise Exception(f"최대 재시도 초과: {url}")

def extract_pmc_fulltext(pmcid):
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = safe_request_get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    paragraphs = soup.select("p")
    text_blocks = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
    full_text = "\n\n".join(text_blocks).strip()
    if len(full_text) < 300:
        raise Exception(f"본문 길이 부족: {len(full_text)}자")
    return full_text, url

def chunk_text_to_documents(text, source_url, r):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    metadata = {
        "source": source_url,
        "pmcid": r.get("pmcid"),
        "title": r.get("title"),
        "doi": r.get("doi"),
        "pubYear": r.get("pubYear"),
        "journal": r.get("journalTitle"),
        "authors": r.get("authorString"),
        "abstract": r.get("abstractText")
    }
    for doc in docs:
        doc.metadata = metadata
    return docs

def get_existing_pmcids(collection_name, persist_dir):
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=EMBEDDING_MODEL,
        persist_directory=persist_dir
    )
    metadatas = vector_store.get().get("metadatas", [])
    return {m.get("pmcid") for m in metadatas if m and m.get("pmcid")}

def store_documents_batch(docs, collection_name, persist_dir, batch_size=10):
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=EMBEDDING_MODEL,
        persist_directory=persist_dir
    )
    for i in range(0, len(docs), batch_size):
        chunk = docs[i:i+batch_size]
        vector_store.add_documents(chunk)

# ======================================
# ✅ 크롤링 + 요약 + 저장
# ======================================

def process_topic(query, collection_name, persist_dir, batch_size, max_docs):
    cursor = "*"
    page_size = 20
    existing_pmcs = get_existing_pmcids(collection_name, persist_dir)

    if len(existing_pmcs) >= max_docs:
        print(f"✅ {collection_name}: 이미 {len(existing_pmcs)}/{max_docs}개 논문 저장됨 → 이번 라운드 패스")
        return

    all_new_docs = []
    new_pmcs = set()
    pbar = tqdm.tqdm(desc=f"{collection_name} 처리", unit="논문")
    processed_count = 0

    while True:
        res = safe_request_get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": query, "format": "json", "pageSize": page_size, "cursorMark": cursor}
        )
        data = res.json()
        results = data.get("resultList", {}).get("result", [])
        if not results:
            break

        filtered = [r for r in results if r.get("pmcid") not in existing_pmcs and r.get("pmcid") not in new_pmcs]

        for r in filtered:
            pmcid = r.get("pmcid")
            if not pmcid:
                continue

            if len(existing_pmcs) + len(new_pmcs) >= max_docs:
                print(f"⚡ {collection_name}: 목표 {max_docs}개 논문 채움 → 종료")
                pbar.close()
                if all_new_docs:
                    store_documents_batch(all_new_docs, collection_name, persist_dir, batch_size)
                    print(f"\n총 {len(new_pmcs)}개 논문 {collection_name}에 저장 완료 (batch size: {batch_size})")
                return

            try:
                text, url = extract_pmc_fulltext(pmcid)

                # ✅ LLM 요약
                print(f"\n>>> {pmcid} 요약 중...")
                summary = LLM.invoke(f"아래 논문 본문을 한국어로 간결하게 요약해 주세요:\n\n{text}")
                summary_text = summary.content

                docs = chunk_text_to_documents(summary_text, url, r)
                all_new_docs.extend(docs)
                new_pmcs.add(pmcid)

                processed_count += 1
                print(f"[+{processed_count}] {pmcid} 처리 완료 (논문 기준)")
                pbar.update(1)
            except Exception as e:
                print(f"[+{processed_count}] {pmcid} 처리 실패: {str(e)}")

        cursor = data.get("nextCursorMark")
        if not cursor:
            break

    if all_new_docs:
        store_documents_batch(all_new_docs, collection_name, persist_dir, batch_size)
        print(f"\n총 {len(new_pmcs)}개 논문 {collection_name}에 저장 완료 (batch size: {batch_size})")
    else:
        print("\n저장할 새 논문이 없습니다.")

    pbar.close()


