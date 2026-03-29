package anthropic

import (
	"github.com/jiayx/llmio/internal/llm"
	protocolanthropic "github.com/jiayx/llmio/internal/protocols/anthropic"
)

func llmToAnthropic(req llm.ChatRequest) protocolanthropic.MessagesRequest {
	return protocolanthropic.MessagesRequestFromLLM(req)
}

func anthropicToLLMResponse(resp protocolanthropic.MessagesResponse, raw []byte) *llm.ChatResponse {
	return protocolanthropic.MessagesResponseToLLM(resp, raw)
}

func anthropicSSEToLLMEvents(eventName, payload string, blockState map[int]protocolanthropic.ContentBlock, out chan<- llm.StreamEvent) error {
	return protocolanthropic.SSEEventToLLMEvents(eventName, payload, blockState, out)
}
