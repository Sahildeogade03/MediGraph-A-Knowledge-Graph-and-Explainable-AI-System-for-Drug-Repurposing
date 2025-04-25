# import streamlit as st

# def main():
#     st.title("Ask Chatbot!")
    
#     prompt = st.chat_input("Pass your prompt here:")
    
#     if 'messages' not in st.session_state:
#         st.session_state.messages = [] 
    
#     for message in st.session_state.messages:
#         st.chat_message(message['role']).markdown(message['content']) # show list 
#         st.session_state.messages.append({'role':'user', 'content' : prompt}) #append list
#     #session state variables are used to save the chat in streamlit as it refershses the project

#     if prompt:
#         st.write("User input received:", prompt)  # Debugging line
#         st.chat_message("user").markdown(prompt)
#         st.session_state.messages.append({'role':'user', 'content' : prompt})
        
#         response = "Hi, I am Medibot!"
#         st.chat_message("assistant").markdown(response)
#         st.session_state.messages.append({'role':'assistant', 'content' : prompt})
#         # st.write("Response sent!")  # Debugging line

# if __name__ == "__main__":
#     main()


# import streamlit as st

# def main():
#     st.title("Ask Chatbot!")
    
#     # Initialize chat history if not present
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     # Display existing chat messages without modifying the list
#     for message in st.session_state.messages:
#         st.chat_message(message['role']).markdown(message['content'])  

#     # Get user input
#     prompt = st.chat_input("Pass your prompt here:")

#     if prompt:  # Process only if user has entered something
#         # Store and display user's message
#         st.session_state.messages.append({'role': 'user', 'content': prompt})
#         st.chat_message("user").markdown(prompt)

#         # Generate and store bot response
#         response = "Hi, I am Medibot!"
#         st.session_state.messages.append({'role': 'assistant', 'content': response})
#         st.chat_message("assistant").markdown(response)

# if __name__ == "__main__":
#     main()





import os

import streamlit as st

from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

DB_FAISS_PATH = "D:/EDAI/model/vectorestore/db_faiss"


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt



@st.cache_resource # model load then store in cache
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# HF_TOKEN = os.environ.get("HF_TOKEN")

HF_TOKEN= "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"  # Replace with your actual token



def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id, 
        temperature=0.5, 
        huggingfacehub_api_token=HF_TOKEN, 
        model_kwargs={"max_length": 512}  
    )
    return llm



def main():
    st.title("Ask Chatbot!")

    # Initialize chat history if not present
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display existing chat messages without modifying the list
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])  

       # Get user input
    prompt = st.chat_input("Pass your prompt here:")

    if prompt:  # Process only if user has entered something
        # Store and display user's message
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.chat_message("user").markdown(prompt)


        CUSTOM_PROMPT_TEMPLATE = """
            Usen the pieces of information provided in the context to answer user's question.
            Answer as you are a Scientist trying to repurpose a drug or connect different medicines with different diseases.


            Conext :{context}
            Question :{question}

            Start the answer directly. No small talk please.
            """

        HUGGING_FACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"



        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGING_FACE_REPO_ID), 
                chain_type="stuff", 
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True, 
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}  
            )

            response = qa_chain.invoke({'query': prompt})

            # result = response['result']
            # souce_documents = response['source_documents']

            # Generate and store bot response
            # response = "Hi, I am Medibot!"
            # result_to_show = result + "\n\nSource Docs:\n" + str(souce_documents)

            result = response['result']
            source_documents = response['source_documents']

            # Format source document details
            formatted_sources = "\n".join([
                f"- **Page {doc.metadata.get('page', 'N/A')}** from *{doc.metadata.get('source', 'Unknown Source')}*"
                for doc in source_documents
            ])

            # Generate and store bot response with better formatting
            result_to_show = f"""
            **Response:**
            {result}

            **Source Documents:**
            {formatted_sources if formatted_sources else 'No relevant documents found.'}
            """

            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})
            st.chat_message("assistant").markdown(result_to_show)

        except Exception as e:  
            st.error(f"Error: {e}")  

if __name__ == "__main__":
    main()
