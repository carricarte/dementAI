from functools import lru_cache


@lru_cache(maxsize=1)
def get_llm():
    from langchain_anthropic import ChatAnthropic

    from backend.config import settings

    return ChatAnthropic(
        model=settings.claude_model,
        anthropic_api_key=settings.anthropic_api_key,
        max_tokens=4096,
    )
