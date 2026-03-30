package openai

import (
	"testing"

	"github.com/jiayx/llmio/internal/llm"
)

func TestStreamChunkPayloadToLLMEventsSupportsReasoningContentAndChoiceUsage(t *testing.T) {
	events, err := StreamChunkPayloadToLLMEvents(`{
		"id":"chatcmpl-1",
		"object":"chat.completion.chunk",
		"model":"kimi-k2.5",
		"choices":[
			{
				"index":0,
				"delta":{"reasoning_content":"thinking","content":"hello"},
				"finish_reason":"stop",
				"usage":{"prompt_tokens":12,"completion_tokens":7,"total_tokens":19}
			}
		]
	}`)
	if err != nil {
		t.Fatalf("StreamChunkPayloadToLLMEvents() error = %v", err)
	}
	if len(events) != 4 {
		t.Fatalf("events = %#v", events)
	}
	if events[0].Type != llm.StreamEventDelta || events[0].Part.Type != llm.ContentTypeText || events[0].TextDelta != "hello" {
		t.Fatalf("event0 = %#v", events[0])
	}
	if events[1].Type != llm.StreamEventDelta || events[1].Part.Type != llm.ContentTypeReasoning || events[1].TextDelta != "thinking" {
		t.Fatalf("event1 = %#v", events[1])
	}
	if events[2].Type != llm.StreamEventStop || events[2].FinishReason != "stop" {
		t.Fatalf("event2 = %#v", events[2])
	}
	if events[3].Type != llm.StreamEventUsage || events[3].InputTokens != 12 || events[3].OutputTokens != 7 {
		t.Fatalf("event3 = %#v", events[3])
	}
}
