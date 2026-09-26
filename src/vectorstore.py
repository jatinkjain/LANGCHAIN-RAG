from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma



def create_vector_store(chunks):
    
   embeddings = HuggingFaceEmbeddings(
       model_name="sentence-transformers/all-MiniLM-L6-v2"
   )

   vector_store = Chroma.from_documents(
       documents=chunks,
       embedding=embeddings,
       collection_name="rag_chatbot"
   )

   
   retriever = vector_store.as_retriever(
       search_kwargs= {"k":3}
   )

   return retriever