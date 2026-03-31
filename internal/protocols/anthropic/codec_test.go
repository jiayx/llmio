package anthropic

import (
	"testing"

	"github.com/jiayx/llmio/internal/llm"
)

func TestMessagesResponseToLLMIncludesCacheTokens(t *testing.T) {
	got := MessagesResponseToLLM(MessagesResponse{
		ID:    "msg_1",
		Model: "claude-sonnet",
		Usage: Usage{
			InputTokens:              100,
			CacheReadInputTokens:     60,
			CacheCreationInputTokens: 20,
			OutputTokens:             15,
		},
	}, []byte(`{}`))

	if got.InputTokens != 100 || got.CacheReadInputTokens != 60 || got.CacheCreationInputTokens != 20 || got.CachedInputTokens != 60 || got.OutputTokens != 15 {
		t.Fatalf("response = %#v", got)
	}
}

func TestSSEEventToLLMEventsIncludesCacheTokens(t *testing.T) {
	events := make(chan llm.StreamEvent, 4)
	blockState := make(map[int]ContentBlock)

	if err := SSEEventToLLMEvents("message_start", `{"type":"message_start","message":{"usage":{"input_tokens":100,"cache_read_input_tokens":60,"cache_creation_input_tokens":20}}}`, blockState, events); err != nil {
		t.Fatalf("message_start error = %v", err)
	}
	if err := SSEEventToLLMEvents("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15,"cache_read_input_tokens":60,"cache_creation_input_tokens":20}}`, blockState, events); err != nil {
		t.Fatalf("message_delta error = %v", err)
	}

	first := <-events
	second := <-events

	if first.Type != llm.StreamEventUsage || first.CacheReadInputTokens != 60 || first.CacheCreationInputTokens != 20 || first.CachedInputTokens != 60 {
		t.Fatalf("first = %#v", first)
	}
	if second.Type != llm.StreamEventUsage || second.OutputTokens != 15 || second.CacheReadInputTokens != 60 || second.CacheCreationInputTokens != 20 {
		t.Fatalf("second = %#v", second)
	}
}
