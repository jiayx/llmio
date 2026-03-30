package openai

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/jiayx/llmio/internal/llm"
	providershared "github.com/jiayx/llmio/internal/providers/shared"
)

func TestChatStreamDelaysStopUntilAfterTrailingUsage(t *testing.T) {
	provider := &OpenAICompatible{
		name: "moonshot",
		httpClient: providershared.NewHTTPClient("http://example.com", nil, &http.Client{
			Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
					Body: io.NopCloser(strings.NewReader(
						"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n" +
							"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
							"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":7,\"total_tokens\":19}}\n\n" +
							"data: [DONE]\n\n")),
				}, nil
			}),
		}, nil),
	}

	stream, err := provider.ChatStream(context.Background(), llm.ChatRequest{
		Model:    "kimi-k2.5",
		Stream:   true,
		Messages: []llm.Message{{Role: "user", Content: []llm.ContentPart{{Type: llm.ContentTypeText, Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("ChatStream() error = %v", err)
	}
	defer func() { _ = stream.Close() }()

	var events []llm.StreamEvent
	for event := range stream.Events {
		events = append(events, event)
	}
	if len(events) != 3 {
		t.Fatalf("events = %#v", events)
	}
	if events[0].Type != llm.StreamEventDelta || events[0].TextDelta != "hello" {
		t.Fatalf("event0 = %#v", events[0])
	}
	if events[1].Type != llm.StreamEventUsage || events[1].InputTokens != 12 || events[1].OutputTokens != 7 {
		t.Fatalf("event1 = %#v", events[1])
	}
	if events[2].Type != llm.StreamEventStop || events[2].FinishReason != "stop" {
		t.Fatalf("event2 = %#v", events[2])
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}
