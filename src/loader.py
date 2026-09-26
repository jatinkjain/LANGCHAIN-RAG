from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



def load_and_split_doc(file_path):
    
   loader = PyPDFLoader("data/rag_chatbot_practice_notes.pdf")
   text_splitter = RecursiveCharacterTextSplitter(
       chunk_size = 500,
       chunk_overlap = 50
)
   documents = loader.load()

   chunks = text_splitter.split_documents(documents)

   return chunks 