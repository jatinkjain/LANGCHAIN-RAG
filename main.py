from dotenv import load_dotenv


load_dotenv()

chat_history = []





while True:
    question = input("YOU: ").strip()

    if not question:
        print("Please enter your question")
        continue

    if question.lower() == "exit":
        print("BOT: GOODBYE!")
        break

    result = rag_chain.invoke({
        "question":question,
        "chat_history":chat_history
    })

    print(f"BOT: {result.content}\n")

    chat_history.append(("user",question))
    chat_history.append(("assistant",result.content))

