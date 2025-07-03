from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

# 각 store의 collection_name과 persist_directory를 명시적으로 지정
VECTOR_STORES = [
    {"name": "vitamin_chroma", "collection_name": "vitamin", "persist_directory": "crawling/vector_store/vitamin_chroma"},
    {"name": "soomin", "collection_name": "soomin", "persist_directory": "crawling/vector_store/soomin"},
    {"name": "others", "collection_name": "others", "persist_directory": "crawling/vector_store/others"},
    {"name": "chroma_pubmed_v3_large", "collection_name": "langchain", "persist_directory": "crawling/vector_store/chroma_pubmed_v3_large"},
    {"name": "chroma", "collection_name": "vitamin", "persist_directory": "crawling/vector_store/chroma"},
]

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

for store in VECTOR_STORES:
    print(f"=== {store['name']} ===")
    try:
        vector_store = Chroma(
            collection_name=store["collection_name"],
            persist_directory=store["persist_directory"],
            embedding_function=embedding_model
        )
        docs = vector_store._collection.get(include=['documents'])['documents']
        print(f"문서 개수: {len(docs)}")
        for i, doc in enumerate(docs[:3]):
            print(f"--- 문서 {i+1} ---")
            if hasattr(doc, "page_content"):
                print(doc.page_content)
            else:
                print(doc)
            print()
    except Exception as e:
        print(f"오류 발생: {e}")
    print() 