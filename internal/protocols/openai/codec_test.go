package openai

import (
	"encoding/json"
	"net/http/httptest"
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

func TestWriteErrorUsesOfficialEnvelope(t *testing.T) {
	rec := httptest.NewRecorder()
	WriteError(rec, 400, "bad request")

	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("unmarshal = %v", err)
	}
	if _, ok := body["error"]; !ok {
		t.Fatalf("body = %#v", body)
	}
	if _, ok := body["choices"]; ok {
		t.Fatalf("body = %#v", body)
	}
}

func TestWriteChatCompletionResponseToolCallUsesNullContent(t *testing.T) {
	rec := httptest.NewRecorder()
	WriteChatCompletionResponse(rec, "gpt-proxy", &llm.ChatResponse{
		ID: "resp_1",
		Output: []llm.ContentPart{{
			Type:       llm.ContentTypeToolCall,
			Name:       "lookup",
			ToolCallID: "call_1",
			Input:      `{"q":"hi"}`,
		}},
	})

	var body map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("unmarshal = %v", err)
	}
	choices := body["choices"].([]any)
	message := choices[0].(map[string]any)["message"].(map[string]any)
	if content, ok := message["content"]; ok && content != nil {
		t.Fatalf("message = %#v", message)
	}
}

func TestChatCompletionRequestToLLMPreservesReasoningContent(t *testing.T) {
	req := ChatCompletionRequest{
		Model: "kimi-k2.5",
		Messages: []Message{
			{
				Role:             "assistant",
				ReasoningContent: "think first",
				ToolCalls: []ToolCall{{
					ID:   "call_1",
					Type: "function",
					Function: FunctionCall{
						Name:      "lookup",
						Arguments: `{"q":"hello"}`,
					},
				}},
			},
		},
	}

	got, err := ChatCompletionRequestToLLM(req)
	if err != nil {
		t.Fatalf("ChatCompletionRequestToLLM() error = %v", err)
	}
	if len(got.Messages) != 1 {
		t.Fatalf("messages = %#v", got.Messages)
	}
	parts := got.Messages[0].Content
	if len(parts) != 2 {
		t.Fatalf("parts = %#v", parts)
	}
	if parts[0].Type != llm.ContentTypeReasoning || parts[0].Text != "think first" {
		t.Fatalf("parts = %#v", parts)
	}
	if parts[1].Type != llm.ContentTypeToolCall || parts[1].Name != "lookup" {
		t.Fatalf("parts = %#v", parts)
	}
}

func TestChatCompletionRequestFromLLMIncludesReasoningContent(t *testing.T) {
	req := llm.ChatRequest{
		Model: "kimi-k2.5",
		Messages: []llm.Message{{
			Role: "assistant",
			Content: []llm.ContentPart{
				{Type: llm.ContentTypeReasoning, Text: "think first"},
				{Type: llm.ContentTypeToolCall, ToolCallID: "call_1", Name: "lookup", Input: `{"q":"hello"}`},
			},
		}},
	}

	got := ChatCompletionRequestFromLLM(req)
	if len(got.Messages) != 1 {
		t.Fatalf("messages = %#v", got.Messages)
	}
	if got.Messages[0].ReasoningContent != "think first" {
		t.Fatalf("message = %#v", got.Messages[0])
	}
	if len(got.Messages[0].ToolCalls) != 1 || got.Messages[0].ToolCalls[0].Function.Name != "lookup" {
		t.Fatalf("message = %#v", got.Messages[0])
	}
}
