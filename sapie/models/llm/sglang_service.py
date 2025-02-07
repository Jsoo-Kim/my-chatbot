from sapie.models.llm.openai_service import OpenAIService

class SGLangService(OpenAIService):
    """
    SGLang OpenAI-Compatible이므로 OpenAIService를 상속하여 재사용.
    필요 시 추가 설정 가능.
    """
    def call_api(self, messages: list):
        response = self.client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            temperature=0.0,
            max_tokens=32680,
            top_p=0.7,
            presence_penalty=0.2,
            stream=self.streaming,
        )
        if self.streaming:
            # 스트리밍 데이터 생성 (제너레이터)
            for chunk in response:
                if chunk.choices[0].delta.content:
                    # print(f"openai_service 쪽 청크: {chunk.choices[0].delta.content}")
                    yield chunk.choices[0].delta.content
        else:
            # 단일 응답 처리
            return response.choices[0].message.content