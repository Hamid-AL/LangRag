"""
text splitters and chunking strategies
Optimizing document chunks for RAG
"""

from h11._abnf import chunk_size
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def document_spliter(path:Path):
    # load the full pdf
    loader=PyPDFLoader(path)
    docs = loader.load()

    # define the splitter
    splitter= RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=50)
    split_docs=splitter.split_documents(docs)

    return split_docs


if __name__== '__main__':
    file_path= Path("./docs/langchain_demo.pdf")
    split_documents= document_spliter(file_path)
    print(f"Split into {len(split_documents)} chunks")
    print(f"\nFirst chunk metadata: {split_documents[0].metadata}")
    print(f"First chunk content: {split_documents[0].page_content[:200]}...")
    print(f"\nLast chunk metadata: {split_documents[-1].metadata}")
