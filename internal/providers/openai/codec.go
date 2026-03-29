package openai

import (
	"github.com/jiayx/llmio/internal/llm"
	protocolopenai "github.com/jiayx/llmio/internal/protocols/openai"
)

func llmToOpenAI(req llm.ChatRequest) protocolopenai.ChatCompletionRequest {
	return protocolopenai.ChatCompletionRequestFromLLM(req)
}

func openAIToLLMResponse(resp protocolopenai.ChatCompletionResponse, raw []byte) *llm.ChatResponse {
	return protocolopenai.ChatCompletionResponseToLLM(resp, raw)
}

func openAIStreamChunkToLLMEvents(payload string) ([]llm.StreamEvent, error) {
	return protocolopenai.StreamChunkPayloadToLLMEvents(payload)
}
