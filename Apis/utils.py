# search/utils.py
from langchain_milvus import Milvus
from langchain_cohere import CohereEmbeddings
from uuid import uuid4
from langchain_core.documents import Document
import pandas as pd
import json
from tqdm import tqdm

def load_csv_data_as_json(csv_path):
    df = pd.read_csv(csv_path, usecols=['tweets', 'sentiment'])
    json_data = df.to_json(orient='records')
    json_data = json.loads(json_data)
    return json_data

def prepare_documents(json_data):
    documents = []
    for data in tqdm(json_data):
        document = Document(
            page_content=f"tweets - {data['tweets']} and sentiment - {data['sentiment']}",
            metadata={"sentiment": data['sentiment']}
        )
        documents.append(document)
    uuids = [str(uuid4()) for _ in range(len(documents))]
    return documents, uuids

def load_embedding_model(model_name, api_key):
    embeddings = CohereEmbeddings(
        model=model_name, cohere_api_key=api_key
    )
    return embeddings

def get_milvus_instance(embeddings, uri, collection_name):
    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": uri},
        collection_name=collection_name,
    )
    return vector_store

def store_documents_in_vectordb(vector_store, documents, uuids):
    vector_store.add_documents(documents=documents, ids=uuids)
    print("Documents stored successfully.")
    return True

def similarity_search(vector_store, text, k):
    results = vector_store.similarity_search_with_score(
        text, k=k,
    )  
    for res, score in results:
        return res.metadata, score
