
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st


text_loader = TextLoader('stdata.txt')
document = text_loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(document)

# Initialize embeddings and ChatOpenAI model
embeddings = OpenAIEmbeddings()
chat = ChatOpenAI(model_name='gpt-3.5-turbo')

# Initialize the vector store index
index = VectorstoreIndexCreator(
    vectorstore_cls=FAISS,
    embedding=embeddings,
).from_loaders([text_loader])

# Save the index to a file
index.vectorstore.save_local("faiss-nj")

# Streamlit app
st.title('Helper for Exchange Student')
question = st.text_input('answer your question')

if st.button('request question'):
    with st.spinner('Wait a minute...'):
        # Query the index with the user's question and get the answer
        answer = index.query(question, llm=chat, verbose=True)
        st.write('answer:', answer)

#실행 명령어 streamlit run C:/Users/tymus/PycharmProjects/pythonProject4/sme_prac2.py

