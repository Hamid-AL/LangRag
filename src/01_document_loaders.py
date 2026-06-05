
import os
import tempfile
#import pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    PyPDFLoader
)
from dotenv import load_dotenv
load_dotenv()


def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
        temp_file.write(b"hello, this is a sample text file. \nThis file is used for demenstraion porpuses")
        temp_file_path=temp_file.name
    try:
        # load the text file using the loader
        text_loader=TextLoader(temp_file_path)
        documents=text_loader.load()

        # Print the documents
        for doc in documents:
            print('Document content: ')
            print(doc)
            print(doc.page_content)
    finally:
        # clean the temp fil
        os.remove(temp_file_path)


def pdf_loader(pdf_path:str):
    pdf_loader=PyPDFLoader(pdf_path)
    documents=pdf_loader.load()

    print(f'loaded {len(documents)} document(s) from pdf')
    for i, doc in enumerate(documents):
        print(f'document {i+1} content preview : {doc.page_content[:100]}')
        print(f'matadata: {doc.metadata}')


pdf_loader("./docs/langchain_demo.pdf")