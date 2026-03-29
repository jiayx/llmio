package providers

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/jiayx/llmio/internal/core"
	anthropicproto "github.com/jiayx/llmio/internal/protocols/anthropic"
)

func TestCoreToAnthropic(t *testing.T) {
	temp := 0.3
	req := core.ChatRequest{
		Model:       "claude-3-7-sonnet",
		System:      []core.ContentPart{{Type: core.ContentTypeText, Text: "be concise"}},
		Messages:    []core.Message{{Role: "user", Content: []core.ContentPart{{Type: core.ContentTypeText, Text: "hello"}}}},
		MaxTokens:   256,
		Temperature: &temp,
		User:        "u-1",
	}

	got := coreToAnthropic(req)
	if got.Model != "claude-3-7-sonnet" || got.MaxTokens != 256 {
		t.Fatalf("request = %#v", got)
	}
	if got.Metadata["user_id"] != "u-1" {
		t.Fatalf("metadata = %#v", got.Metadata)
	}
	system, ok := got.System.([]anthropicproto.ContentBlock)
	if !ok || len(system) != 1 || system[0].Text != "be concise" {
		t.Fatalf("system = %#v", got.System)
	}
	if len(got.Messages) != 1 {
		t.Fatalf("messages = %#v", got.Messages)
	}
}

func TestAnthropicToCoreResponse(t *testing.T) {
	got := anthropicToCoreResponse(anthropicproto.MessagesResponse{
		ID:         "msg_1",
		Model:      "claude-3-7-sonnet",
		StopReason: "end_turn",
		Content: []anthropicproto.ContentBlock{
			{Type: "text", Text: "hello"},
		},
		Usage: anthropicproto.Usage{
			InputTokens:  9,
			OutputTokens: 4,
		},
	}, []byte(`{}`))

	if got.ID != "msg_1" || got.OutputText != "hello" || got.InputTokens != 9 || got.OutputTokens != 4 {
		t.Fatalf("response = %#v", got)
	}
}

func TestAnthropicSSEToCoreEvents(t *testing.T) {
	events := make(chan core.StreamEvent, 8)
	blockState := make(map[int]anthropicproto.ContentBlock)

	if err := anthropicSSEToCoreEvents("message_start", `{"type":"message_start","message":{"usage":{"input_tokens":12}}}`, blockState, events); err != nil {
		t.Fatalf("message_start error = %v", err)
	}
	if err := anthropicSSEToCoreEvents("content_block_delta", `{"type":"content_block_delta","delta":{"type":"text_delta","text":"hel"}}`, blockState, events); err != nil {
		t.Fatalf("content_block_delta error = %v", err)
	}
	if err := anthropicSSEToCoreEvents("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`, blockState, events); err != nil {
		t.Fatalf("message_delta error = %v", err)
	}

	first := <-events
	second := <-events
	third := <-events
	fourth := <-events

	if first.Type != core.StreamEventUsage || first.InputTokens != 12 {
		t.Fatalf("first = %#v", first)
	}
	if second.Type != core.StreamEventDelta || second.TextDelta != "hel" {
		t.Fatalf("second = %#v", second)
	}
	if third.Type != core.StreamEventUsage || third.OutputTokens != 5 {
		t.Fatalf("third = %#v", third)
	}
	if fourth.Type != core.StreamEventStop || fourth.FinishReason != "end_turn" {
		t.Fatalf("fourth = %#v", fourth)
	}
}

func TestAnthropicSSEToCoreToolEvents(t *testing.T) {
	events := make(chan core.StreamEvent, 8)
	blockState := make(map[int]anthropicproto.ContentBlock)

	if err := anthropicSSEToCoreEvents("content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call_1","name":"lookup","input":{"q":"hel"}}}`, blockState, events); err != nil {
		t.Fatalf("content_block_start error = %v", err)
	}
	if err := anthropicSSEToCoreEvents("content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"lo"}}`, blockState, events); err != nil {
		t.Fatalf("content_block_delta error = %v", err)
	}

	first := <-events
	second := <-events
	third := <-events
	if first.Type != core.StreamEventContentStart || first.Part.Type != core.ContentTypeToolCall {
		t.Fatalf("first = %#v", first)
	}
	if second.Type != core.StreamEventTool || second.ToolCallID != "call_1" || second.ToolName != "lookup" {
		t.Fatalf("second = %#v", second)
	}
	if third.Type != core.StreamEventTool || third.ToolInput != "lo" {
		t.Fatalf("third = %#v", third)
	}
}

func TestAnthropicSSEToCoreImageLifecycle(t *testing.T) {
	events := make(chan core.StreamEvent, 8)
	blockState := make(map[int]anthropicproto.ContentBlock)

	if err := anthropicSSEToCoreEvents("content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"image","source":{"type":"base64","media_type":"image/png","data":"abc"}}}`, blockState, events); err != nil {
		t.Fatalf("content_block_start error = %v", err)
	}
	if err := anthropicSSEToCoreEvents("content_block_stop", `{"index":2}`, blockState, events); err != nil {
		t.Fatalf("content_block_stop error = %v", err)
	}

	first := <-events
	second := <-events
	if first.Type != core.StreamEventContentStart || first.Part.Type != core.ContentTypeImage {
		t.Fatalf("first = %#v", first)
	}
	if second.Type != core.StreamEventContentStop || second.Part.Type != core.ContentTypeImage {
		t.Fatalf("second = %#v", second)
	}
}

func TestAnthropicNativeChat(t *testing.T) {
	provider := &AnthropicNative{
		name: "anthropic",
		httpClient: newProviderHTTPClient("http://example.com", nil, &http.Client{
			Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
				if req.URL.Path != "/messages" {
					t.Fatalf("path = %s", req.URL.Path)
				}
				if req.Header.Get("x-api-key") != "secret" {
					t.Fatalf("api key header missing")
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Body: io.NopCloser(strings.NewReader(`{
						"id":"msg_1",
						"type":"message",
						"role":"assistant",
						"model":"claude-3-7-sonnet",
						"content":[{"type":"text","text":"hello"}],
						"stop_reason":"end_turn",
						"usage":{"input_tokens":7,"output_tokens":3}
					}`)),
					Header: make(http.Header),
				}, nil
			}),
		}, func(req *http.Request) {
			req.Header.Set("x-api-key", "secret")
			req.Header.Set("anthropic-version", "2023-06-01")
		}),
	}

	resp, err := provider.Chat(context.Background(), core.ChatRequest{
		Model:    "claude-3-7-sonnet",
		Messages: []core.Message{{Role: "user", Content: []core.ContentPart{{Type: core.ContentTypeText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if resp.OutputText != "hello" || resp.InputTokens != 7 || resp.OutputTokens != 3 {
		t.Fatalf("resp = %#v", resp)
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}
