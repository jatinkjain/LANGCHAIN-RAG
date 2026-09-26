from langchain_core.runnables import RunnableLambda
from LLM import create_llm
from prompts import create_rag_prompt,create_rewrite_prompt



def format_data(data):
    return "\n\n".join(doc.page_content for doc in data)



def create_rag_chain(retriever):
    chat_llm = create_llm()

    rag_prompt = create_rag_prompt()
    rewrite_prompt = create_rewrite_prompt()

    question_rewriter = rewrite_prompt | chat_llm

    
    def retrieve_docs(x):
        rewritten_question = question_rewriter.invoke({
            "chat_history": x["chat_history"],
            "question": x["question"]
        })
    
        return retriever.invoke(rewritten_question.content)
    

    rag_chain = (
        {
            "context": RunnableLambda(retrieve_docs) | RunnableLambda(format_data),
            "question": RunnableLambda(lambda x: x["question"]),
            "chat_history":RunnableLambda(lambda x: x["chat_history"])
        }
        | rag_prompt
        | chat_llm
    )

    return rag_chain
