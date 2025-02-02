from sapie.models.llm.openai_service import OpenAIService

class SGLangService(OpenAIService):
    """
    VLLM은 OpenAI-Compatible이므로 OpenAIService를 상속하여 재사용.
    필요 시 추가 설정 가능.
    """
    pass