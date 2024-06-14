import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import tempfile
import pandas as pd

# Load environment variables
load_dotenv()

# Load the GROQ and Google API keys from Streamlit secrets
groq_api_key = st.secrets["GROQ_API_KEY"]
google_api_key = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = google_api_key

# Function to create vector embeddings
@st.cache_resource
def vector_embedding(pdf_files):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    all_docs = []
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)  # Data Ingestion
        docs = loader.load()  # Document Loading
        all_docs.extend(docs)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)  # Chunk Creation
    final_documents = text_splitter.split_documents(all_docs)  # Splitting
    vectors = FAISS.from_documents(final_documents, embeddings)  # Vector Embeddings
    return vectors

# Function to save feedback
def save_feedback(feedback_text, email):
    feedback_data = {'Feedback': [feedback_text], 'Email': [email]}
    feedback_df = pd.DataFrame(feedback_data)
    try:
        feedback_df.to_csv('feedback.csv', mode='a', index=False, header=not os.path.isfile('feedback.csv'))
    except Exception as e:
        st.error(f"Error saving feedback: {e}")

# Function to download feedback
def download_feedback():
    try:
        return open('feedback.csv', 'rb').read()
    except FileNotFoundError:
        return None

# Feedback page
def feedback_page():
    st.title("User Feedback")
    st.write("We value your feedback. Please let us know how we can improve this app.")

    feedback_text = st.text_area("Enter your feedback here")
    email = st.text_input("Your Email (optional)")

    if st.button("Submit Feedback"):
        if feedback_text:
            save_feedback(feedback_text, email)
            st.success("Thank you for your feedback, we appreciate your input!")
        else:
            st.error("Feedback cannot be empty.")

# Admin page
def admin_page():
    st.title("Admin Page")
    st.write("Download feedback data. This user feedback data can only be downloaded by the admin or their delegates.")

    password = st.text_input("Enter password", type="password")
    submit_button = st.button("Submit")

    if submit_button:
        if password == st.secrets["admin"]["password"]:  # Replace with your secure password
            feedback_csv = download_feedback()
            if feedback_csv:
                st.download_button(
                    label="Download Feedback as CSV",
                    data=feedback_csv,
                    file_name="feedback.csv",
                    mime="text/csv"
                )
            else:
                st.error("No feedback data available.")
        else:
            st.error("Incorrect password, contact the admin")

# Main page for document Q&A
def main_page():
    # Set Streamlit application title
    st.title("RAG-Powered LLM Model For Document Q&A")

    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader("Upload your PDF documents", type="pdf", accept_multiple_files=True)

    # Initialize chat history in the session state if it doesn't exist
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Check if files have been uploaded
    if uploaded_files:
        # Load the vectors (this will cache the result)
        vectors = vector_embedding(uploaded_files)

        # Initialize the ChatGroq language model
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

        # Define the prompt template
        prompt_template = ChatPromptTemplate.from_template("""
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Questions: {input}
        """)

        # Chat input
        prompt = st.chat_input("Ask a question based on the uploaded documents")
        if prompt:
            # Append the user's question to the chat history and display it
            st.session_state.history.append({
                'role': 'user',
                'content': prompt
            })

            for exchange in st.session_state.history:
                with st.chat_message(exchange['role']):
                    st.markdown(exchange['content'])

            # Show a spinner while processing the response
            with st.spinner('💡 Thinking...'):
                # Create document chain and retriever
                document_chain = create_stuff_documents_chain(llm, prompt_template)
                retriever = vectors.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                # Run the QA chain
                response = retrieval_chain.invoke({'input': prompt})
                answer_text = response['answer'] if response['answer'] else "Sorry, I couldn't find an answer in the documents."

                # Append the assistant's response to the chat history
                st.session_state.history.append({
                    'role': 'Assistant',
                    'content': answer_text
                })

                # Display the assistant's response
                with st.chat_message('Assistant'):
                    st.markdown(answer_text)

                # Display relevant document chunks in an expander
                with st.expander("Document Similarity Search"):
                    for i, doc in enumerate(response["context"]):
                        st.write(doc.page_content)
                        st.write("--------------------------------")
    else:
        st.sidebar.warning("Please upload PDF documents to proceed.")

# Page navigation
page = st.sidebar.selectbox("Choose a page", ["Main", "Feedback", "Admin"])

if page == "Main":
    main_page()
elif page == "Feedback":
    feedback_page()
elif page == "Admin":
    admin_page()
