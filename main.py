from dotenv import load_dotenv
load_dotenv()

from langchain_core import __version__ as core_version
#from langgraph import __version__ as lg_version
from langchain_cohere import ChatCohere


print (f'langchain-core version {core_version}')
#print(f'langgraph vesion {lg_version}')


def main(): 
    llm = ChatCohere(
        model="command-a-plus-05-2026",
        temperature=0.7
    )

    response = llm.invoke("hello")

    print(response)

if __name__ == "__main__":
    main()
