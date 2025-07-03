import os
import time
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import tqdm
import gc
from textwrap import dedent
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# Vector Store 저장 경로 설정
PERSIST_DIR = "vector_store/others"
os.makedirs(PERSIST_DIR, exist_ok=True)

# 임베딩 모델 초기화
EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-large")


def safe_request_get(url, params=None, headers=None, retries=3, backoff=1, timeout=5):
    """
    지정된 URL에 안전하게 GET 요청을 보내고, 실패 시 재시도(backoff)를 수행합니다.
    """
    for attempt in range(retries):
        try:
            res = requests.get(url, params=params, headers=headers, timeout=timeout)
            res.raise_for_status()
            return res
        except requests.exceptions.RequestException as e:
            print(f"요청 실패 ({attempt + 1}/{retries}): {e}")
            time.sleep(backoff * (attempt + 1))
    raise Exception(f"요청 실패: {url}")


def get_existing_pmcids(collection_name, persist_dir):
    """
    지정된 Vector Store에서 이미 저장된 PMCID 목록을 가져옵니다.
    """
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=EMBEDDING_MODEL,
        persist_directory=persist_dir
    )
    metadatas = vector_store.get().get("metadatas", [])
    return {m.get("pmcid") for m in metadatas if m and m.get("pmcid")}


def extract_abstract_html(pmcid):
    """
    PMC 웹페이지에서 주어진 PMCID의 abstract 내용을 추출합니다.
    """
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = safe_request_get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    abstract_section = soup.find("section", class_="abstract")
    if not abstract_section:
        print(f"abstract section 없음 (PMCID: {pmcid}) → 건너뜀")
        return None

    paragraphs = abstract_section.find_all("p")
    abstract_text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    if not abstract_text:
        print(f"abstract 내용 없음 (PMCID: {pmcid}) → 건너뜀")
        return None

    return abstract_text


def chunk_text_to_documents(text, r, url):
    """
    abstract 텍스트를 chunk 단위로 나눠 LangChain Document 객체 리스트로 변환합니다.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([text])
    metadata = {
        "pmcid": r.get("pmcid", ""),
        "title": r.get("title", ""),
        "doi": r.get("doi", ""),
        "pubYear": r.get("pubYear", ""),
        "journal": r.get("journalTitle", ""),
        "authors": r.get("authorString", ""),
        "source": url
    }
    for doc in docs:
        doc.metadata = metadata
    return docs


def store_documents_batch(docs, collection_name, persist_dir, batch_size=5):
    """
    지정된 Vector Store에 문서 리스트를 batch 단위로 저장합니다.
    """
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=EMBEDDING_MODEL,
        persist_directory=persist_dir
    )
    for i in range(0, len(docs), batch_size):
        chunk = docs[i:i + batch_size]
        vector_store.add_documents(chunk)


def process_topic(query, collection_name, persist_dir, batch_size):
    """
    EuropePMC API에서 논문 검색 후, abstract를 크롤링하고 Vector Store에 저장합니다.
    """
    cursor = "*"
    page_size = 5

    # 기존에 저장된 PMCID 목록 가져오기
    existing_pmcs = get_existing_pmcids(collection_name, persist_dir)

    # 총 검색 결과 개수 확인
    init_res = safe_request_get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": query, "format": "json", "pageSize": 1},
        headers={"User-Agent": "Mozilla/5.0"}
    )
    init_data = init_res.json()
    total_count = init_data.get("hitCount", 0)
    print(f"{collection_name}: EuropePMC 검색 결과 개수: {total_count}개")

    if total_count == 0:
        print(f"{collection_name}: 검색 결과 없음 → 패스")
        return

    all_new_docs = []
    new_pmcs = set()
    progress_index = 1
    pbar = tqdm.tqdm(desc=f"{collection_name} 진행", unit="논문", total=total_count)

    try:
        while True:
            # EuropePMC API 요청 (cursor 기반 페이징)
            res = safe_request_get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={"query": query, "format": "json", "pageSize": page_size, "cursorMark": cursor},
                headers={"User-Agent": "Mozilla/5.0"}
            )
            data = res.json()
            results = data.get("resultList", {}).get("result", [])
            if not results:
                break

            for r in results:
                pmcid = r.get("pmcid", "")

                # PMCID 없으면 건너뜀
                if not pmcid:
                    print(f"[{progress_index}/{total_count}] PMCID 없음 → 건너뜀")
                    progress_index += 1
                    pbar.update(1)
                    continue

                # 이미 저장된 PMCID면 건너뜀
                if pmcid in existing_pmcs or pmcid in new_pmcs:
                    print(f"[{progress_index}/{total_count}] 이미 저장된 논문 건너뜀 (PMCID: {pmcid})")
                    progress_index += 1
                    pbar.update(1)
                    continue

                # Abstract HTML 추출
                abstract_text = extract_abstract_html(pmcid)
                if not abstract_text:
                    progress_index += 1
                    pbar.update(1)
                    continue

                url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                docs = chunk_text_to_documents(abstract_text, r, url)
                all_new_docs.extend(docs)
                new_pmcs.add(pmcid)

                print(f"[{progress_index}/{total_count}] 저장 완료 (PMCID: {pmcid})")

                # 일정 개수 이상 쌓이면 batch 저장
                if len(all_new_docs) >= batch_size:
                    store_documents_batch(all_new_docs, collection_name, persist_dir, batch_size)
                    all_new_docs = []
                    gc.collect()

                progress_index += 1
                pbar.update(1)

            # cursor 업데이트 및 종료 조건 체크
            prev_cursor = cursor
            cursor = data.get("nextCursorMark")
            if not cursor or cursor == prev_cursor:
                break

    except KeyboardInterrupt:
        print("\n중단 감지! 지금까지의 데이터를 저장합니다...")
        if all_new_docs:
            store_documents_batch(all_new_docs, collection_name, persist_dir, batch_size)
            print(f"중단 시점까지 {len(new_pmcs)}개 논문 저장 완료")
        else:
            print("저장할 새 문서 없음")
        pbar.close()
        gc.collect()
        raise

    # 남은 문서 저장
    if all_new_docs:
        store_documents_batch(all_new_docs, collection_name, persist_dir, batch_size)
        print(f"\n총 {len(new_pmcs)}개 논문 {collection_name}에 저장 완료 (batch size: {batch_size})")
    else:
        print("\n저장할 새 논문이 없습니다.")

    pbar.close()
    gc.collect()
