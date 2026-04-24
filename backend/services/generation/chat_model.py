from langchain_cohere import ChatCohere


def get_chat_model(model: str = "command-a-03-2025", temperature: float = 0.7) -> ChatCohere:
    return ChatCohere(model=model, temperature=temperature)


def generate_response(chat_model: ChatCohere, prompt: str) -> str:
    from langchain_core.messages import HumanMessage

    response = chat_model.invoke([HumanMessage(content=prompt)])
    return response.content


__all__ = ["get_chat_model", "generate_response"]
