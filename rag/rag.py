from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser

def chat_update(chat, role, text):
    chat.append({"role": role, "text": text})
    return chat

def startRAG(bd_dir, api_key):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectordb = Chroma(persist_directory=bd_dir, embedding_function=embedding_model)

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 35, "lambda_mult": 0.3}
        )
    
    llm_generation = ChatOpenAI(
        model="google/gemma-3-27b-it:free",
        temperature=0.2,
        max_tokens = 2000,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    llm_structurizing = ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct:free",
        temperature=0,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    prompt_identify = ChatPromptTemplate.from_template(
    """Given a chat history and the latest user question which might reference context in the chat history,
    formulate a standalone question which can be understood without the chat history. Do NOT answer the question,
    just reformulate it if needed and otherwise return it as is.

        Chat History:
        {chat_history}

        Latest Question: 
        {question}

        Standalone Question:
    """
    )

    prompt_generate = ChatPromptTemplate.from_template(
    """You are a scientific assistant specializing in Alzheimer's disease research. Your task is to answer questions based ONLY on the provided context from scientific articles.

    STRICT RULES:
    1. Answer ONLY using information explicitly stated in the context below
    2. If the context does not contain enough information to answer fully, say "The provided documents do not contain sufficient information about [topic]"
    3. Never use external knowledge or make assumptions beyond the given text
    4. Always cite the sources: include article titles and DOI/URLs when mentioning findings

    CONTEXT:
    {docs}

    QUESTION: {question}

    ANSWER (cite sources with titles and links):"""
    )

    prompt_structurize = ChatPromptTemplate.from_template(
    """You are a formatting assistant. 
    Input: 
    1. An "Answer" text.
    2. A list of "Source Documents" with metadata.

    Task:
    Return a JSON object with two keys:
    - "message": The exact "Answer" text provided below.
    - "sources": A list of objects ["title": "...", "file": "..."] for every document from the "Source Documents" list that supports the answer.

    Answer:
    {initial_answer}

    Source Documents:
    {docs}
    """
    )

    identified_answer_chain = prompt_identify | llm_generation | StrOutputParser()
    answer_chain = prompt_generate | llm_generation | StrOutputParser()
    structurized_answer_chain = prompt_structurize | llm_structurizing | JsonOutputParser()

    return identified_answer_chain, answer_chain, structurized_answer_chain, retriever

def answer_question(identified_answer_chain, answer_chain, structurized_answer_chain, retriever, chat):
    
    last_messages = chat[-5:]

    user_query = (identified_answer_chain.invoke({"chat_history": str(last_messages), "question": last_messages[-1]["text"]})).strip()
    
    docs_retrieved = retriever.invoke(user_query)

    initial_answer = answer_chain.invoke({"docs": docs_retrieved, "question": user_query})
    result = structurized_answer_chain.invoke({"initial_answer": initial_answer, "docs": docs_retrieved})

    message = dict(result)["message"]
    sources_list = dict(result)["sources"]
    
    file_dict = {source.get("file"): source.get("title") for source in sources_list}

    chat = chat_update(chat, "ai", message)

    return chat, file_dict