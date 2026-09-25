from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from huggingface_hub import InferenceClient
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()

loader = PyPDFLoader("data/rag_chatbot_practice_notes.pdf")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)
documents = loader.load()

chunks = text_splitter.split_documents(documents)

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



question = "What is RAG?"

formatted_prompt = prompt.invoke({
    "context": "RAG stands for Retrieval-Augmented Generation.",
    "question": question
})


llm = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-R1-0528",
    task = "conversational",
    max_new_tokens=512,
    temperature= 0.1,
    provider= "auto",
    timeout = 120
)

ll2 = ChatHuggingFace(llm=llm)



result = ll2.invoke("what is RAG?")

print(result)