import streamlit as st 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    """
    Get an array of pdf files and return all the raw text
    """
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(raw_text):
    """
    Takes all the raw text of the pdfs and return a list of chunks of text
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def create_vectorstore(text_chunks):
    """
    Create envedings of the text chunks and create vectorstore
    """
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")#model_name="hkunlp/instructor-xl"
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore 

def get_conversation_chain(vectorstore):
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.1})
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_message=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    import time
    start = time.time()
    response = st.session_state.conversation({'question': user_question})
    end = time.time() - start
    print(end)

    if st.session_state.firts_chat:
        st.session_state.firts_chat = False
        response['chat_history'] = 

    
    st.session_state.chat_history = response['chat_history']
    

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
    
    breakpoint()


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    st.header("Chat:books:")
    user_question = st.text_input("Ask a question about the docs:")
    if user_question:
        handle_userinput(user_question)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "firts_chat" not in st.session_state:
        st.session_state.firts_chat = True

    with st.sidebar:
        st.subheader("Your docs")
        pdf_docs = st.file_uploader("Upload your docs", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get pdf text
                raw_text = get_pdf_text(pdf_docs)

                #Get text chunks
                text_chunks = get_text_chunks(raw_text)

                #Create vector store
                vectorstore = create_vectorstore(text_chunks)

                #Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                

if __name__=='__main__':
    main()