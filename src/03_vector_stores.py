
from chromadb.utils import sparse_embedding_utils
from cohere.audio.transcriptions.types import audio_transcriptions_create_response
from langchain_chroma import Chroma
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from uuid import uuid4
import hashlib
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
load_dotenv()


def add_to_vector_store(vector_store, documents:List[Document]):
    """
    adds docs to the given vectore store
    uses hashed to prevent adding duplicate documents
    """
    ids=[]
    for doc in documents:
        source=doc.metadata.get("source","unkown")
        payload=f"{doc.page_content}::{source}".encode("utf-8")

        # generate a unique hash for the doc
        doc_id=hashlib.sha256(payload).hexdigest()
        ids.append(doc_id)
    # add data into the vectore store
    vector_store.add_documents(documents=documents, ids=ids)
    # check how many docs in the collection
    print(f"ThE vectore store contain: {vector_store._collection.count()} documents")
    return vector_store
    


def clear_vector_store(vector_store):
    docs= vector_store._collection.get()
    ids=docs['ids']
    vector_store.delete(ids=ids)
    return vector_store

def similarity_search_with_relevance_score(vector_store, 
                                            query: str, 
                                            k: int =3, 
                                            filter: Optional[Dict[str, Any]]= None):
    results=vector_store.similarity_search_with_relevance_scores(query, k=k, filter=filter)
    return results

if __name__== '__main__':
    # connect to the running docker chroma db
    chroma_client= chromadb.HttpClient(host='localhost', port=8000)

    embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")


    # define vectore store
    vector_store=Chroma(
        client=chroma_client,
        collection_name='my_collection',
        embedding_function=embeddings
    )


    document_1 = Document(
        page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
        metadata={"source": "tweet"},
        id=1,
    )

    document_2 = Document(
        page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
        metadata={"source": "news"},
        id=2,
    )

    document_3 = Document(
        page_content="Building an exciting new project with LangChain - come check it out!",
        metadata={"source": "tweet"},
        id=3,
    )

    document_4 = Document(
        page_content="Robbers broke into the city bank and stole $1 million in cash.",
        metadata={"source": "news"},
        id=4,
    )

    document_5 = Document(
        page_content="Wow! That was an amazing movie. I can't wait to see it again.",
        metadata={"source": "tweet"},
        id=5,
    )

    document_6 = Document(
        page_content="Is the new iPhone worth the price? Read this review to find out.",
        metadata={"source": "website"},
        id=6,
    )

    document_7 = Document(
        page_content="The top 10 soccer players in the world right now.",
        metadata={"source": "website"},
        id=7,
    )

    document_8 = Document(
        page_content="LangGraph is the best framework for building stateful, agentic applications!",
        metadata={"source": "tweet"},
        id=8,
    )

    document_9 = Document(
        page_content="The stock market is down 500 points today due to fears of a recession.",
        metadata={"source": "news"},
        id=9,
    )

    document_10 = Document(
        page_content="I have a bad feeling I am going to get deleted :(",
        metadata={"source": "tweet"},
        id=10,
    )

    documents = [
        document_1,
        document_2,
        document_3,
        document_4,
        document_5,
        document_6,
        document_7,
        document_8,
        document_9,
        document_10,
    ]
    vectore_store= add_to_vector_store(vector_store, documents)
    res = similarity_search_with_relevance_score(vector_store, 'what is langgraph?', filter={'source': 'website'})
    print(res)