from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace


def create_llm():
    llm = HuggingFaceEndpoint(
       repo_id = "openai/gpt-oss-120b",
       task = "conversational",
       max_new_tokens=512,
       temperature= 0.1,
       provider= "auto",
       timeout = 120
    )

    chat_llm = ChatHuggingFace(llm=llm)

    return chat_llm