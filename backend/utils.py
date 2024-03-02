
from langchain_community.embeddings import OpenAIEmbeddings
import weaviate
from weaviate.embedded import EmbeddedOptions
from langchain_community.vectorstores import Weaviate
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.memory import ( ConversationBufferWindowMemory)
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

import requests

def translate_text(text, from_lang, to_lang):
    url = "https://translate.googleapis.com/translate_a/single"
    headers = {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
        "Content-Type": "application/json; charset=utf-8",
        "Origin": "https://www.easyamharictyping.com",
        "Referer": "https://www.easyamharictyping.com/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "cross-site",
        "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
    }

    params = {
        "client": "gtx",
        "sl": from_lang,
        "tl": to_lang,
        "dt": "t",
        "q": text,
    }


    response = requests.get(url, params=params, headers=headers)
    answer = response.json()
    answer = "".join([answer[0][i][0] for i in range(len(answer[0]))])
    answer
    return answer


def create_rag_pipeline():
    llm = ChatOpenAI(temperature=0.1, model = 'gpt-3.5-turbo')
    file_path = "../prompts/system_message.txt"
    with open(file_path, 'r') as txt_file:
        template = txt_file.read()

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")

    client = weaviate.Client(embedded_options=EmbeddedOptions())
    vectorstore = Weaviate.from_documents(client=client, documents=[], embedding=OpenAIEmbeddings(), by_text=False)
    retriever = vectorstore.as_retriever()
    chat_memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=chat_memory,
    combine_docs_chain_kwargs={
        "prompt": ChatPromptTemplate.from_messages(
            [
                system_message_prompt,
                human_message_prompt,
            ]
        ),
    },
    )
    
    return conversation_chain