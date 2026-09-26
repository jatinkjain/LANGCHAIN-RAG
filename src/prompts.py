from langchain_core.prompts import ChatPromptTemplate


def create_rag_prompt():

   prompt = ChatPromptTemplate.from_template("""
   You are a helpful RAG assistant.

   Answer the question using ONLY the provided context.

   Give a short, direct answer.
   Do not show your reasoning or thinking process.
   Do not add information that is not present in the context.

   If the answer is not present in the context, say:
   "I don't know based on the provided document."

   Conversation History:
   {chat_history}

   Context:
   {context}

   Question:
   {question}

   Answer:
   """)

   return prompt


def create_rewrite_prompt():
   
   rewrite_prompt = ChatPromptTemplate.from_template("""
    Given the conversation history and the user's latest question,
    rewrite the latest question into a standalone question.

    The standalone question must be understandable without the conversation history.

    Do not answer the question.
    Only return the rewritten question.

    Conversation History:
    {chat_history}

    Latest Question:
    {question}
    """)

   return rewrite_prompt

