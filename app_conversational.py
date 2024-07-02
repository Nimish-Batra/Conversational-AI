
import streamlit as st
import tempfile
import os
import hashlib
import uuid
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain,LLMChain,SequentialChain
from htmlTemplates import css, bot_template, user_template
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate

load_dotenv()

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://ea-openai.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "2355a247f79f4b8ea2adaa0929cd32c2"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

llm = AzureChatOpenAI(azure_deployment="gpt-35-turbo", model_name="gpt-4", temperature=0.50)
# model = SentenceTransformer('all-MiniLM-L6-v2')

def get_chunks(file_obj):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(file_obj.read())

    if '.pdf' in file_obj.name:
        loader = PyPDFLoader(temp_path)
    elif '.docx' in file_obj.name or '.doc' in file_obj.name:
        loader = Docx2txtLoader(temp_path)
    elif '.xlsx' in file_obj.name or '.xls' in file_obj.name:
        loader = UnstructuredExcelLoader(temp_path)

    chunks = loader.load_and_split()
    return chunks

def get_vectostore(text_chunks):
    embeddings = AzureOpenAIEmbeddings(azure_deployment='text-embedding')
    vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key="chat_history", input_key='question', return_messages=True,
                                      output_key="answer")
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,
        return_generated_question=True
    )
    return conversation_chain

def generate_response(question, conversation_chain):
    response = conversation_chain({'question': question})

    # Generate follow-up question
    follow_up_prompt = PromptTemplate(
        input_variables=['answer'],
        template="Based on the answer: {answer}, suggest a follow-up question that could be asked to further the conversation.follow up question should be of the form Do you also want to know this?"
    )
    follow_up_chain = LLMChain(llm=llm, prompt=follow_up_prompt)
    follow_up_response = follow_up_chain.run(answer=response['answer'])

    return response, follow_up_response


def handle_userinput():
    question = st.session_state.current_question


    if question.strip().lower() == "yes":
            question = st.session_state.follow_up_question

    st.session_state["conversation_history"].append(question)
    response, follow_up = generate_response(question, st.session_state.conversation)


    # Ensure the response contains the expected structure
    if 'source_documents' in response and response['source_documents']:
        st.session_state.doc_source.append(response['source_documents'][0].metadata.get("source", "Unknown source"))
        st.session_state.doc_page_num.append(response['source_documents'][0].metadata.get("page", 0) + 1)

    final_chat_hist = []
    for i in range(len(response['chat_history']) // 2):
        j = i * 2
        lis_obj = response['chat_history'][j:j + 2]
        lis_obj.append(st.session_state.doc_source[i])
        lis_obj.append(st.session_state.doc_page_num[i])
        final_chat_hist.append(lis_obj)

    st.session_state.chat_history = response['chat_history']
    for ele in final_chat_hist[::-1]:
        st.write(user_template.replace("{{MSG}}", ele[0].content), unsafe_allow_html=True)
        st.write(
            bot_template.replace("{{MSG}}", ele[1].content).replace("{{source_url}}", ele[2]).replace("{{page_number}}",
                                                                                                      str(ele[3])),
            unsafe_allow_html=True
        )

    st.session_state.follow_up_question = follow_up
    st.session_state.follow_up_needed = True
    st.write(follow_up, unsafe_allow_html=True)


#
def main():
    st.header("Chat with Documents")
    st.write(css, unsafe_allow_html=True)

    with st.sidebar:
        final_chunks = []
        docs = st.file_uploader("Upload your files here and click on Process",
                                type=['docx', 'doc', 'pdf', 'xlsx', 'xls'], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                for f_obj in docs:
                    chunks = get_chunks(f_obj)
                    final_chunks.extend(chunks)
                vectorstore = get_vectostore(final_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

        if len(final_chunks) > 0:
            st.subheader("Processing complete. Ready!")

    if "doc_source" not in st.session_state:
        st.session_state.doc_source = []
        st.session_state.doc_page_num = []
    if "conversation_history" not in st.session_state:
        st.session_state['conversation_history'] = [
            SystemMessage(
                content='You are a Human Conversational agent. Your primary and only task is to generate a follow-up question. This question should be related to the initial user question or introduce a related topic to keep the conversation engaging. Ensure each response offers value and prompts further discussion. You always have to generate the related topic by yourself and then make a follow-up question of the form: "Do you also want to know about this related topic or are you interested in knowing about related topic ?" Always have empathy and compassion. Only return follow-up questions as output, no need to provide any response to the user question.')
        ]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    if 'follow_up_needed' not in st.session_state:
        st.session_state.follow_up_needed = False
    if 'follow_up_question' not in st.session_state:
        st.session_state.follow_up_question = ""

    # Ensure text input updates with the follow-up question if one is selected
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        st.session_state.current_question = user_question
        handle_userinput()



if __name__ == '__main__':
    main()


