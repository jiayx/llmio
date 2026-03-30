package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jiayx/llmio/internal/apikeys"
	"github.com/jiayx/llmio/internal/config"
	"github.com/jiayx/llmio/internal/llm"
	"github.com/jiayx/llmio/internal/policy"
	protocols "github.com/jiayx/llmio/internal/protocols"
	anthropicproto "github.com/jiayx/llmio/internal/protocols/anthropic"
	openaiproto "github.com/jiayx/llmio/internal/protocols/openai"
	providerapi "github.com/jiayx/llmio/internal/providers/api"
	"github.com/jiayx/llmio/internal/routing"
	"github.com/jiayx/llmio/internal/usage"
)

func TestAnthropicToLLM(t *testing.T) {
	temp := 0.2
	req := anthropicproto.MessagesRequest{
		Model: "claude-proxy",
		System: []any{
			map[string]any{"type": "text", "text": "You are concise."},
		},
		Messages: []anthropicproto.Message{
			{Role: "user", Content: "hello"},
			{Role: "assistant", Content: []any{map[string]any{"type": "text", "text": "hi"}}},
		},
		MaxTokens:   128,
		Temperature: &temp,
		Metadata: map[string]string{
			"user_id": "u-1",
		},
	}

	got, err := anthropicproto.MessagesRequestToLLM(req)
	if err != nil {
		t.Fatalf("MessagesRequestToLLM() error = %v", err)
	}
	if got.Model != "claude-proxy" {
		t.Fatalf("model = %q", got.Model)
	}
	if llm.ContentText(got.System) != "You are concise." {
		t.Fatalf("system = %#v", got.System)
	}
	if len(got.Messages) != 2 {
		t.Fatalf("messages len = %d", len(got.Messages))
	}
	if got.User != "u-1" {
		t.Fatalf("user = %q", got.User)
	}
}

func TestLLMResponseToAnthropic(t *testing.T) {
	got := anthropicproto.MessagesResponseFromLLM("claude-compatible", &llm.ChatResponse{
		ID:           "chatcmpl-1",
		Model:        "deepseek-chat",
		OutputText:   "world",
		FinishReason: "stop",
		InputTokens:  11,
		OutputTokens: 7,
	})
	if got.Model != "claude-compatible" {
		t.Fatalf("model = %q", got.Model)
	}
	if got.StopReason != "end_turn" {
		t.Fatalf("stop_reason = %q", got.StopReason)
	}
	if len(got.Content) != 1 || got.Content[0].Text != "world" {
		t.Fatalf("content = %#v", got.Content)
	}
	if got.Usage.InputTokens != 11 || got.Usage.OutputTokens != 7 {
		t.Fatalf("usage = %#v", got.Usage)
	}
}

func TestWithAuth(t *testing.T) {
	s := &Server{adminAPIKeys: map[string]apikeys.Principal{"secret": {ID: "admin:secret", Name: "admin"}}}
	handler := s.withAuth(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer secret")
	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusNoContent {
		t.Fatalf("status = %d", rec.Code)
	}
}

func TestManagedAPIKeyLifecycleAndUsage(t *testing.T) {
	store, err := apikeys.Open(filepath.Join(t.TempDir(), "apikeys.json"))
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}

	providersMap := map[string]chatProvider{
		"primary": fakeProvider{chatResp: &llm.ChatResponse{
			ID:           "resp_1",
			Model:        "backend-model",
			OutputText:   "hello",
			InputTokens:  11,
			OutputTokens: 7,
		}},
	}
	router, err := routing.New(&config.Config{
		Providers: []config.ProviderConfig{{
			Name:    "primary",
			Type:    "openai-compatible",
			BaseURL: "https://example.com/v1",
		}},
		ModelRoutes: []config.ModelRoute{{
			ExternalModel: "gpt-proxy",
			Targets: []config.Target{{
				Provider:     "primary",
				BackendModel: "backend-model",
			}},
		}},
	}, providersMap)
	if err != nil {
		t.Fatalf("routing.New() error = %v", err)
	}

	s := &Server{
		adminAPIKeys: map[string]apikeys.Principal{
			"admin-secret": {ID: "admin:admin-secret", Name: "admin"},
		},
		apiKeyStore:      store,
		providers:        providersMap,
		router:           router,
		policy:           policy.NewWithRecorder(policy.DefaultConfig(), providersMap, usage.MultiRecorder{store}),
		protocolAdapters: protocols.DefaultAdapters(),
	}
	handler := s.Handler()

	createReq := httptest.NewRequest(http.MethodPost, "/admin/api-keys", strings.NewReader(`{"name":"crm-system"}`))
	createReq.Header.Set("Authorization", "Bearer admin-secret")
	createReq.Header.Set("Content-Type", "application/json")
	createRec := httptest.NewRecorder()
	handler.ServeHTTP(createRec, createReq)
	if createRec.Code != http.StatusCreated {
		t.Fatalf("create status = %d body=%s", createRec.Code, createRec.Body.String())
	}

	var created struct {
		Key struct {
			ID string `json:"id"`
		} `json:"key"`
		Secret string `json:"secret"`
	}
	if err := json.Unmarshal(createRec.Body.Bytes(), &created); err != nil {
		t.Fatalf("unmarshal create = %v", err)
	}
	if created.Key.ID == "" || created.Secret == "" {
		t.Fatalf("create body = %s", createRec.Body.String())
	}

	callReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-proxy","messages":[{"role":"user","content":"hi"}]}`))
	callReq.Header.Set("Authorization", "Bearer "+created.Secret)
	callReq.Header.Set("Content-Type", "application/json")
	callRec := httptest.NewRecorder()
	handler.ServeHTTP(callRec, callReq)
	if callRec.Code != http.StatusOK {
		t.Fatalf("call status = %d body=%s", callRec.Code, callRec.Body.String())
	}

	getReq := httptest.NewRequest(http.MethodGet, "/admin/api-keys/"+created.Key.ID, nil)
	getReq.Header.Set("Authorization", "Bearer admin-secret")
	getRec := httptest.NewRecorder()
	handler.ServeHTTP(getRec, getReq)
	if getRec.Code != http.StatusOK {
		t.Fatalf("get status = %d body=%s", getRec.Code, getRec.Body.String())
	}

	var key apikeys.KeySummary
	if err := json.Unmarshal(getRec.Body.Bytes(), &key); err != nil {
		t.Fatalf("unmarshal get = %v", err)
	}
	if key.Usage.RequestCount != 1 || key.Usage.InputTokens != 11 || key.Usage.OutputTokens != 7 || key.Usage.TotalTokens != 18 {
		t.Fatalf("usage = %#v", key.Usage)
	}

	usageReq := httptest.NewRequest(http.MethodGet, "/admin/api-keys/"+created.Key.ID+"/usage", nil)
	usageReq.Header.Set("Authorization", "Bearer admin-secret")
	usageRec := httptest.NewRecorder()
	handler.ServeHTTP(usageRec, usageReq)
	if usageRec.Code != http.StatusOK {
		t.Fatalf("usage status = %d body=%s", usageRec.Code, usageRec.Body.String())
	}

	var singleUsage apikeys.UsageReport
	if err := json.Unmarshal(usageRec.Body.Bytes(), &singleUsage); err != nil {
		t.Fatalf("unmarshal usage = %v", err)
	}
	if singleUsage.KeyID != created.Key.ID || singleUsage.Usage.TotalTokens != 18 {
		t.Fatalf("singleUsage = %#v", singleUsage)
	}

	adminUsageReq := httptest.NewRequest(http.MethodGet, "/admin/usage", nil)
	adminUsageReq.Header.Set("Authorization", "Bearer admin-secret")
	adminUsageRec := httptest.NewRecorder()
	handler.ServeHTTP(adminUsageRec, adminUsageReq)
	if adminUsageRec.Code != http.StatusOK {
		t.Fatalf("admin usage status = %d body=%s", adminUsageRec.Code, adminUsageRec.Body.String())
	}

	var usageList struct {
		Total apikeys.UsageTotals   `json:"total"`
		Data  []apikeys.UsageReport `json:"data"`
	}
	if err := json.Unmarshal(adminUsageRec.Body.Bytes(), &usageList); err != nil {
		t.Fatalf("unmarshal usage list = %v", err)
	}
	if usageList.Total.TotalTokens != 18 || len(usageList.Data) != 1 || usageList.Data[0].KeyID != created.Key.ID {
		t.Fatalf("usageList = %#v", usageList)
	}
}

func TestDisabledManagedAPIKeyCannotCallLLM(t *testing.T) {
	store, err := apikeys.Open(filepath.Join(t.TempDir(), "apikeys.json"))
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	created, err := store.Create("worker")
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if _, err := store.Disable(created.Key.ID); err != nil {
		t.Fatalf("Disable() error = %v", err)
	}

	s := &Server{
		apiKeyStore:      store,
		protocolAdapters: protocols.DefaultAdapters(),
	}
	handler := s.withAuth(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("Authorization", "Bearer "+created.Secret)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d", rec.Code)
	}
}

func TestRecoveryReturnsJSON500AndLogsPanic(t *testing.T) {
	var logBuf strings.Builder
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logBuf, nil)))
	defer slog.SetDefault(prev)

	s := &Server{}
	handler := s.withRecovery(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		panic("boom")
	}))

	req := httptest.NewRequest(http.MethodGet, "/admin/usage", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"error":"internal server error"`) {
		t.Fatalf("body = %s", rec.Body.String())
	}
	if !strings.Contains(logBuf.String(), "request panic") || !strings.Contains(logBuf.String(), "boom") || !strings.Contains(logBuf.String(), "stack=") {
		t.Fatalf("log = %s", logBuf.String())
	}
}

func TestRecoveryReturnsOpenAI500(t *testing.T) {
	s := &Server{}
	handler := s.withRecovery(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		panic("boom")
	}))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"message":"internal server error"`) {
		t.Fatalf("body = %s", rec.Body.String())
	}
}

func TestRecoveryReturnsAnthropic500(t *testing.T) {
	s := &Server{}
	handler := s.withRecovery(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		panic("boom")
	}))

	req := httptest.NewRequest(http.MethodPost, "/anthropic/v1/messages", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"type":"error"`) || !strings.Contains(rec.Body.String(), `"message":"internal server error"`) {
		t.Fatalf("body = %s", rec.Body.String())
	}
}

func TestDispatchChatFallback(t *testing.T) {
	s := &Server{
		providers: map[string]chatProvider{
			"primary": fakeProvider{chatErr: errors.New("provider primary returned status 503: busy")},
			"secondary": fakeProvider{chatResp: &llm.ChatResponse{
				ID:         "ok",
				Model:      "deepseek-chat",
				OutputText: "hi",
			}},
		},
	}

	resp, err := s.dispatchChat(context.Background(), modelRoute{
		Targets: []routeTarget{
			{ProviderName: "primary", BackendModel: "a"},
			{ProviderName: "secondary", BackendModel: "b"},
		},
	}, llm.ChatRequest{
		Model:    "proxy",
		Messages: []llm.Message{{Role: "user", Content: []llm.ContentPart{{Type: llm.ContentTypeText, Text: "hello"}}}},
	})
	if err != nil {
		t.Fatalf("dispatchChat() error = %v", err)
	}
	if resp.ID != "ok" {
		t.Fatalf("response = %#v", resp)
	}
}

func TestOpenAIRequestToLLM(t *testing.T) {
	maxTokens := 64
	req := openaiproto.ChatCompletionRequest{
		Model: "gpt-proxy",
		Messages: []openaiproto.Message{
			{Role: "system", Content: "sys"},
			{Role: "user", Content: "hello"},
		},
		MaxTokens: &maxTokens,
		Stream:    true,
		User:      "u1",
	}

	got, err := openaiproto.ChatCompletionRequestToLLM(req)
	if err != nil {
		t.Fatalf("ChatCompletionRequestToLLM() error = %v", err)
	}
	if llm.ContentText(got.System) != "sys" {
		t.Fatalf("system = %#v", got.System)
	}
	if got.MaxTokens != 64 || !got.Stream || got.User != "u1" {
		t.Fatalf("unexpected request = %#v", got)
	}
}

func TestOpenAIRequestToLLMRejectsUnsupportedContent(t *testing.T) {
	_, err := openaiproto.ChatCompletionRequestToLLM(openaiproto.ChatCompletionRequest{
		Model: "gpt-proxy",
		Messages: []openaiproto.Message{{
			Role: "user",
			Content: []any{
				map[string]any{"type": "audio_url", "audio_url": "https://example.com/a.mp3"},
			},
		}},
	})
	if err == nil || !strings.Contains(err.Error(), "unsupported content type") {
		t.Fatalf("error = %v", err)
	}
}

func TestWriteLLMResponseAsOpenAI(t *testing.T) {
	rec := httptest.NewRecorder()
	openaiproto.WriteChatCompletionResponse(rec, "gpt-proxy", &llm.ChatResponse{
		ID:           "1",
		OutputText:   "hello",
		FinishReason: "stop",
		InputTokens:  1,
		OutputTokens: 2,
	})

	var body openaiproto.ChatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("unmarshal = %v", err)
	}
	if body.Model != "gpt-proxy" || body.Choices[0].Message.Content != "hello" {
		t.Fatalf("body = %#v", body)
	}
	if body.Created == 0 {
		t.Fatalf("created = %#v", body)
	}
}

func TestOpenAIResponsesRequestToLLM(t *testing.T) {
	maxOutputTokens := 128
	req := openaiproto.ResponsesRequest{
		Model:           "gpt-proxy",
		Instructions:    "be brief",
		MaxOutputTokens: &maxOutputTokens,
		Input: []any{
			map[string]any{
				"role": "developer",
				"content": []any{
					map[string]any{"type": "input_text", "text": "Follow policy."},
				},
			},
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "input_text", "text": "hello"},
				},
			},
		},
	}

	got, err := openaiproto.ResponsesRequestToLLM(req)
	if err != nil {
		t.Fatalf("ResponsesRequestToLLM() error = %v", err)
	}
	if llm.ContentText(got.System) != "be brief\nFollow policy." {
		t.Fatalf("system = %#v", got.System)
	}
	if got.MaxTokens != 128 {
		t.Fatalf("max_tokens = %d", got.MaxTokens)
	}
	if len(got.Messages) != 1 || got.Messages[0].Role != "user" || llm.ContentText(got.Messages[0].Content) != "hello" {
		t.Fatalf("messages = %#v", got.Messages)
	}
}

func TestOpenAIResponsesRequestToLLMFunctionCallOutput(t *testing.T) {
	req := openaiproto.ResponsesRequest{
		Model: "gpt-proxy",
		Input: []any{
			map[string]any{
				"type":      "function_call",
				"name":      "lookup",
				"call_id":   "call_1",
				"arguments": `{"q":"hello"}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": "call_1",
				"output":  "done",
			},
		},
	}

	got, err := openaiproto.ResponsesRequestToLLM(req)
	if err != nil {
		t.Fatalf("ResponsesRequestToLLM() error = %v", err)
	}
	if len(got.Messages) != 2 {
		t.Fatalf("messages = %#v", got.Messages)
	}
	if got.Messages[0].Role != "assistant" || got.Messages[0].Content[0].Type != llm.ContentTypeToolCall {
		t.Fatalf("message0 = %#v", got.Messages[0])
	}
	if got.Messages[1].Role != "user" || got.Messages[1].Content[0].Type != llm.ContentTypeToolResult || got.Messages[1].Content[0].Output != "done" {
		t.Fatalf("message1 = %#v", got.Messages[1])
	}
}

func TestLLMResponseToOpenAIResponse(t *testing.T) {
	got := openaiproto.ResponsesResponseFromLLM("gpt-proxy", &llm.ChatResponse{
		ID:           "resp_1",
		OutputText:   "hello",
		InputTokens:  10,
		OutputTokens: 5,
	})
	if got.Object != "response" || got.Model != "gpt-proxy" || got.Status != "completed" {
		t.Fatalf("response = %#v", got)
	}
	if got.OutputText != "hello" {
		t.Fatalf("output_text = %q", got.OutputText)
	}
	if len(got.Output) != 1 || got.Output[0].Content[0].Text != "hello" {
		t.Fatalf("output = %#v", got.Output)
	}
	if got.Usage == nil || got.Usage.TotalTokens != 15 {
		t.Fatalf("usage = %#v", got.Usage)
	}
}

func TestHandleOpenAIResponses(t *testing.T) {
	s := &Server{
		providers: map[string]chatProvider{
			"primary": fakeProvider{chatResp: &llm.ChatResponse{
				ID:           "resp_1",
				OutputText:   "world",
				InputTokens:  3,
				OutputTokens: 4,
			}},
		},
		router: mustTestRouter(t, map[string]chatProvider{
			"primary": fakeProvider{},
		}, "gpt-proxy", "primary", "backend-model"),
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"model": "gpt-proxy",
		"input": "hello"
	}`))
	rec := httptest.NewRecorder()

	s.handleOpenAIResponses(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}

	var body openaiproto.ResponsesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("unmarshal = %v", err)
	}
	if body.Model != "gpt-proxy" || body.OutputText != "world" {
		t.Fatalf("body = %#v", body)
	}
}

func TestAnthropicToLLMToolAndImage(t *testing.T) {
	req := anthropicproto.MessagesRequest{
		Model: "claude-proxy",
		Messages: []anthropicproto.Message{
			{
				Role: "user",
				Content: []any{
					map[string]any{
						"type": "image",
						"source": map[string]any{
							"type":       "base64",
							"media_type": "image/png",
							"data":       "abc",
						},
					},
					map[string]any{
						"type": "tool_result", "tool_use_id": "call_1", "content": "done",
					},
				},
			},
			{
				Role: "assistant",
				Content: []any{
					map[string]any{
						"type": "tool_use", "id": "call_1", "name": "lookup", "input": map[string]any{"q": "hi"},
					},
				},
			},
		},
	}

	got, err := anthropicproto.MessagesRequestToLLM(req)
	if err != nil {
		t.Fatalf("MessagesRequestToLLM() error = %v", err)
	}
	if got.Messages[0].Content[0].Type != llm.ContentTypeImage {
		t.Fatalf("message0 = %#v", got.Messages[0].Content)
	}
	if got.Messages[0].Content[1].Type != llm.ContentTypeToolResult {
		t.Fatalf("message0 = %#v", got.Messages[0].Content)
	}
	if got.Messages[1].Content[0].Type != llm.ContentTypeToolCall {
		t.Fatalf("message1 = %#v", got.Messages[1].Content)
	}
}

func TestOpenAIRequestToLLMToolCalls(t *testing.T) {
	req := openaiproto.ChatCompletionRequest{
		Model: "gpt-proxy",
		Messages: []openaiproto.Message{
			{
				Role: "assistant",
				ToolCalls: []openaiproto.ToolCall{{
					ID:   "call_1",
					Type: "function",
					Function: openaiproto.FunctionCall{
						Name:      "lookup",
						Arguments: `{"q":"hello"}`,
					},
				}},
			},
			{
				Role:       "tool",
				ToolCallID: "call_1",
				Content:    "done",
			},
		},
		Tools: []openaiproto.Tool{{
			Type: "function",
			Function: openaiproto.FunctionDefinition{
				Name:       "lookup",
				Parameters: json.RawMessage(`{"type":"object"}`),
			},
		}},
	}

	got, err := openaiproto.ChatCompletionRequestToLLM(req)
	if err != nil {
		t.Fatalf("ChatCompletionRequestToLLM() error = %v", err)
	}
	if len(got.Tools) != 1 || got.Tools[0].Name != "lookup" {
		t.Fatalf("tools = %#v", got.Tools)
	}
	if got.Messages[0].Content[0].Type != llm.ContentTypeToolCall {
		t.Fatalf("message0 = %#v", got.Messages[0].Content)
	}
	if got.Messages[1].Content[0].Type != llm.ContentTypeToolResult {
		t.Fatalf("message1 = %#v", got.Messages[1].Content)
	}
}

func TestNewServerAnthropicNativeProvider(t *testing.T) {
	server, err := NewServer(&config.Config{
		Providers: []config.ProviderConfig{
			{
				Name:    "anthropic",
				Type:    "anthropic-native",
				BaseURL: "https://api.anthropic.com/v1",
				APIKey:  "secret",
			},
		},
		ModelRoutes: []config.ModelRoute{
			{
				ExternalModel: "claude-proxy",
				Targets: []config.Target{
					{Provider: "anthropic", BackendModel: "claude-3-7-sonnet"},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("NewServer() error = %v", err)
	}
	if _, ok := server.providers["anthropic"]; !ok {
		t.Fatalf("provider not registered")
	}
}

func TestHandleOpenAIChatCompletionsPassthrough(t *testing.T) {
	p := &fakePassthroughProvider{
		openAIResponse: &http.Response{
			StatusCode: http.StatusOK,
			Header: http.Header{
				"Content-Type": []string{"application/json"},
				"X-Request-Id": []string{"upstream-id"},
			},
			Body: io.NopCloser(strings.NewReader(`{"id":"1","model":"backend-model","choices":[]}`)),
		},
	}
	s := &Server{
		providers: map[string]chatProvider{"p": p},
		router:    mustTestRouter(t, map[string]chatProvider{"p": p}, "gpt-proxy", "p", "backend-model"),
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"gpt-proxy","messages":[{"role":"user","content":"hello"}]}`))
	rec := httptest.NewRecorder()

	s.handleOpenAIChatCompletions(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if p.chatCalled {
		t.Fatalf("normalized path should not be called")
	}
	if p.openAIPath != "/chat/completions" || !strings.Contains(string(p.openAIBody), `"model":"backend-model"`) {
		t.Fatalf("passthrough request = path=%s body=%s", p.openAIPath, string(p.openAIBody))
	}
	if !strings.Contains(rec.Body.String(), `"model":"gpt-proxy"`) {
		t.Fatalf("response = %s", rec.Body.String())
	}
	if rec.Header().Get("X-Request-Id") != "" {
		t.Fatalf("unexpected passthrough header: %#v", rec.Header())
	}
	if rec.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("headers = %#v", rec.Header())
	}
}

func TestHandleOpenAIResponsesPassthrough(t *testing.T) {
	p := &fakePassthroughProvider{
		openAIResponse: &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"id":"resp_1","object":"response","model":"backend-model","status":"completed","output":[]}`)),
		},
	}
	s := &Server{
		providers: map[string]chatProvider{"p": p},
		router:    mustTestRouter(t, map[string]chatProvider{"p": p}, "gpt-proxy", "p", "backend-model"),
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"gpt-proxy","input":"hello"}`))
	rec := httptest.NewRecorder()

	s.handleOpenAIResponses(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if p.openAIPath != "/responses" || !strings.Contains(string(p.openAIBody), `"model":"backend-model"`) {
		t.Fatalf("passthrough request = path=%s body=%s", p.openAIPath, string(p.openAIBody))
	}
}

func TestHandleOpenAIResponsesFallsBackWhenAPITypeUnsupported(t *testing.T) {
	p := &fakePassthroughProvider{
		chatResp: &llm.ChatResponse{
			ID:           "resp_1",
			OutputText:   "fallback",
			InputTokens:  3,
			OutputTokens: 4,
		},
		openAIAPIs: map[string]struct{}{
			providerapi.OpenAIAPIChatCompletions: {},
		},
	}
	s := &Server{
		providers: map[string]chatProvider{"p": p},
		router:    mustTestRouter(t, map[string]chatProvider{"p": p}, "gpt-proxy", "p", "backend-model"),
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"gpt-proxy","input":"hello"}`))
	rec := httptest.NewRecorder()

	s.handleOpenAIResponses(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !p.chatCalled {
		t.Fatalf("normalized path should be called")
	}
	if p.openAIPath != "" {
		t.Fatalf("unexpected passthrough path = %s", p.openAIPath)
	}

	var body openaiproto.ResponsesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("unmarshal = %v", err)
	}
	if body.OutputText != "fallback" {
		t.Fatalf("body = %#v", body)
	}
}

func TestHandleOpenAIChatCompletionsMethodNotAllowed(t *testing.T) {
	s := &Server{}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)

	s.handleOpenAIChatCompletions(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if rec.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("allow = %q", rec.Header().Get("Allow"))
	}
}

func TestHandleAnthropicMessagesMethodNotAllowed(t *testing.T) {
	s := &Server{}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/anthropic/v1/messages", nil)

	s.handleAnthropicMessages(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"type":"error"`) {
		t.Fatalf("body = %s", rec.Body.String())
	}
}

func TestHandleAnthropicMessagesPassthrough(t *testing.T) {
	p := &fakePassthroughProvider{
		anthropicResponse: &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"id":"msg_1","type":"message","model":"backend-model","role":"assistant","content":[],"usage":{"input_tokens":1,"output_tokens":1}}`)),
		},
	}
	s := &Server{
		providers: map[string]chatProvider{"p": p},
		router:    mustTestRouter(t, map[string]chatProvider{"p": p}, "claude-proxy", "p", "backend-model"),
	}

	req := httptest.NewRequest(http.MethodPost, "/anthropic/v1/messages", strings.NewReader(`{"model":"claude-proxy","max_tokens":16,"messages":[{"role":"user","content":"hello"}]}`))
	rec := httptest.NewRecorder()

	s.handleAnthropicMessages(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if p.chatCalled {
		t.Fatalf("normalized path should not be called")
	}
	if p.anthropicPath != "/messages" || !strings.Contains(string(p.anthropicBody), `"model":"backend-model"`) {
		t.Fatalf("passthrough request = path=%s body=%s", p.anthropicPath, string(p.anthropicBody))
	}
	if !strings.Contains(rec.Body.String(), `"model":"claude-proxy"`) {
		t.Fatalf("response = %s", rec.Body.String())
	}
}

func TestWritePassthroughResponseSSEModelRewrite(t *testing.T) {
	rec := httptest.NewRecorder()
	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header: http.Header{
			"Content-Type": []string{"text/event-stream"},
			"Server":       []string{"provider"},
		},
		Body: io.NopCloser(strings.NewReader("event: message\ndata: {\"model\":\"backend-model\"}\n\ndata: [DONE]\n\n")),
	}

	openaiproto.WritePassthroughResponse(rec, resp, "gpt-proxy")
	body := rec.Body.String()
	if !strings.Contains(body, `"model":"gpt-proxy"`) {
		t.Fatalf("body = %s", body)
	}
	if rec.Header().Get("Server") != "" {
		t.Fatalf("headers = %#v", rec.Header())
	}
}

func TestHandleOpenAIStreamToolCall(t *testing.T) {
	stream := testStream(
		llm.StreamEvent{Type: llm.StreamEventTool, ToolIndex: 0, ToolCallID: "call_1", ToolName: "lookup", ToolInput: `{"q":"hel`},
		llm.StreamEvent{Type: llm.StreamEventTool, ToolIndex: 0, ToolCallID: "call_1", ToolInput: `lo"}`},
		llm.StreamEvent{Type: llm.StreamEventStop, FinishReason: "tool_calls"},
	)
	s := &Server{
		providers: map[string]chatProvider{"p": fakeProvider{stream: stream}},
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	rec := httptest.NewRecorder()

	s.handleOpenAIStream(rec, req, modelRoute{Targets: []routeTarget{{ProviderName: "p", BackendModel: "x"}}}, llm.ChatRequest{Model: "gpt-proxy"})
	body := rec.Body.String()
	if !strings.Contains(body, `"tool_calls"`) || !strings.Contains(body, `"lookup"`) || !strings.Contains(body, `[DONE]`) {
		t.Fatalf("body = %s", body)
	}
}

func TestHandleOpenAIStreamReasoning(t *testing.T) {
	stream := testStream(
		llm.StreamEvent{Type: llm.StreamEventDelta, Part: llm.ContentPart{Type: llm.ContentTypeReasoning}, TextDelta: "think"},
		llm.StreamEvent{Type: llm.StreamEventStop, FinishReason: "stop"},
	)
	s := &Server{
		providers: map[string]chatProvider{"p": fakeProvider{stream: stream}},
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	rec := httptest.NewRecorder()

	s.handleOpenAIStream(rec, req, modelRoute{Targets: []routeTarget{{ProviderName: "p", BackendModel: "x"}}}, llm.ChatRequest{Model: "gpt-proxy"})
	body := rec.Body.String()
	if !strings.Contains(body, `"reasoning":"think"`) || !strings.Contains(body, `[DONE]`) {
		t.Fatalf("body = %s", body)
	}
}

func TestHandleOpenAIResponsesStreamToolCall(t *testing.T) {
	stream := testStream(
		llm.StreamEvent{Type: llm.StreamEventTool, ToolIndex: 0, ToolCallID: "call_1", ToolName: "lookup", ToolInput: `{"q":"hi"}`},
		llm.StreamEvent{Type: llm.StreamEventStop, FinishReason: "tool_calls"},
	)
	s := &Server{
		providers: map[string]chatProvider{"p": fakeProvider{stream: stream}},
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	rec := httptest.NewRecorder()

	s.handleOpenAIResponsesStream(rec, req, "gpt-proxy", modelRoute{Targets: []routeTarget{{ProviderName: "p", BackendModel: "x"}}}, llm.ChatRequest{Model: "gpt-proxy"})
	body := rec.Body.String()
	if !strings.Contains(body, "response.function_call_arguments.delta") || !strings.Contains(body, "response.output_item.done") || !strings.Contains(body, `"call_id":"call_1"`) {
		t.Fatalf("body = %s", body)
	}
}

func TestHandleOpenAIResponsesStreamReasoningAndImage(t *testing.T) {
	stream := testStream(
		llm.StreamEvent{Type: llm.StreamEventContentStart, BlockIndex: 1, Part: llm.ContentPart{Type: llm.ContentTypeImage, URL: "https://example.com/image.png"}},
		llm.StreamEvent{Type: llm.StreamEventDelta, Part: llm.ContentPart{Type: llm.ContentTypeReasoning}, TextDelta: "think"},
		llm.StreamEvent{Type: llm.StreamEventContentStop, BlockIndex: 1, Part: llm.ContentPart{Type: llm.ContentTypeImage, URL: "https://example.com/image.png"}},
		llm.StreamEvent{Type: llm.StreamEventStop, FinishReason: "stop"},
	)
	s := &Server{
		providers: map[string]chatProvider{"p": fakeProvider{stream: stream}},
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	rec := httptest.NewRecorder()

	s.handleOpenAIResponsesStream(rec, req, "gpt-proxy", modelRoute{Targets: []routeTarget{{ProviderName: "p", BackendModel: "x"}}}, llm.ChatRequest{Model: "gpt-proxy"})
	body := rec.Body.String()
	if !strings.Contains(body, "response.reasoning.delta") || !strings.Contains(body, "response.reasoning.done") || !strings.Contains(body, "image.png") {
		t.Fatalf("body = %s", body)
	}
}

func TestHandleOpenAIResponsesStreamContextCanceledDoesNotWrite502(t *testing.T) {
	events := make(chan llm.StreamEvent)
	errs := make(chan error, 1)
	errs <- context.Canceled
	close(errs)

	s := &Server{
		providers: map[string]chatProvider{"p": fakeProvider{stream: &providerapi.StreamReader{
			Events: events,
			Err:    errs,
			Close: func() error {
				return nil
			},
		}}},
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	rec := httptest.NewRecorder()

	s.handleOpenAIResponsesStream(rec, req, "gpt-proxy", modelRoute{Targets: []routeTarget{{ProviderName: "p", BackendModel: "x"}}}, llm.ChatRequest{Model: "gpt-proxy"})
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if strings.Contains(rec.Body.String(), `"error"`) {
		t.Fatalf("body = %s", rec.Body.String())
	}
}

func TestHandleAnthropicStreamToolCall(t *testing.T) {
	stream := testStream(
		llm.StreamEvent{Type: llm.StreamEventTool, ToolCallID: "call_1", ToolName: "lookup", ToolInput: `{"q":"hi"}`},
		llm.StreamEvent{Type: llm.StreamEventStop, FinishReason: "tool_use"},
	)
	s := &Server{
		providers: map[string]chatProvider{"p": fakeProvider{stream: stream}},
	}
	req := httptest.NewRequest(http.MethodPost, "/anthropic/v1/messages", nil)
	rec := httptest.NewRecorder()

	s.handleAnthropicStream(rec, req, "claude-proxy", modelRoute{Targets: []routeTarget{{ProviderName: "p", BackendModel: "x"}}}, llm.ChatRequest{Model: "claude-proxy"})
	body := rec.Body.String()
	if !strings.Contains(body, `"type":"tool_use"`) || !strings.Contains(body, `"input_json_delta"`) || !strings.Contains(body, `"message_stop"`) {
		t.Fatalf("body = %s", body)
	}
}

func TestHandleAnthropicStreamImageBlock(t *testing.T) {
	stream := testStream(
		llm.StreamEvent{Type: llm.StreamEventContentStart, BlockIndex: 2, Part: llm.ContentPart{Type: llm.ContentTypeImage, MediaType: "image/png", Data: "abc"}},
		llm.StreamEvent{Type: llm.StreamEventContentStop, BlockIndex: 2, Part: llm.ContentPart{Type: llm.ContentTypeImage, MediaType: "image/png", Data: "abc"}},
		llm.StreamEvent{Type: llm.StreamEventStop, FinishReason: "end_turn"},
	)
	s := &Server{
		providers: map[string]chatProvider{"p": fakeProvider{stream: stream}},
	}
	req := httptest.NewRequest(http.MethodPost, "/anthropic/v1/messages", nil)
	rec := httptest.NewRecorder()

	s.handleAnthropicStream(rec, req, "claude-proxy", modelRoute{Targets: []routeTarget{{ProviderName: "p", BackendModel: "x"}}}, llm.ChatRequest{Model: "claude-proxy"})
	body := rec.Body.String()
	if !strings.Contains(body, `"type":"image"`) || !strings.Contains(body, `"content_block_stop"`) || !strings.Contains(body, `"message_stop"`) {
		t.Fatalf("body = %s", body)
	}
}

func TestHandlerAnthropicStreamWithLoggingWrapper(t *testing.T) {
	stream := testStream(
		llm.StreamEvent{Type: llm.StreamEventDelta, Part: llm.ContentPart{Type: llm.ContentTypeText}, TextDelta: "hello"},
		llm.StreamEvent{Type: llm.StreamEventStop, FinishReason: "end_turn"},
	)
	s := &Server{
		providers: map[string]chatProvider{"p": fakeProvider{stream: stream}},
		router:    mustTestRouter(t, map[string]chatProvider{"p": fakeProvider{}}, "claude-proxy", "p", "x"),
	}

	req := httptest.NewRequest(http.MethodPost, "/anthropic/v1/messages", strings.NewReader(`{
		"model":"claude-proxy",
		"max_tokens":16,
		"stream":true,
		"messages":[{"role":"user","content":"hello"}]
	}`))
	rec := httptest.NewRecorder()

	s.Handler().ServeHTTP(rec, req)
	body := rec.Body.String()
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, body)
	}
	if strings.Contains(body, "streaming is not supported by this server") {
		t.Fatalf("unexpected streaming support error: %s", body)
	}
	if !strings.Contains(body, `"content_block_delta"`) || !strings.Contains(body, `"message_stop"`) {
		t.Fatalf("body = %s", body)
	}
}

type fakeProvider struct {
	chatResp  *llm.ChatResponse
	chatErr   error
	stream    *providerapi.StreamReader
	streamErr error
}

func (f fakeProvider) Name() string { return "fake" }

func (f fakeProvider) Chat(context.Context, llm.ChatRequest) (*llm.ChatResponse, error) {
	return f.chatResp, f.chatErr
}

func (f fakeProvider) ChatStream(context.Context, llm.ChatRequest) (*providerapi.StreamReader, error) {
	if f.streamErr != nil {
		return nil, f.streamErr
	}
	if f.stream != nil {
		return f.stream, nil
	}
	return nil, errors.New("not implemented")
}

func testStream(events ...llm.StreamEvent) *providerapi.StreamReader {
	ch := make(chan llm.StreamEvent, len(events))
	for _, event := range events {
		ch <- event
	}
	close(ch)
	return &providerapi.StreamReader{
		Events: ch,
		Err:    nil,
		Close: func() error {
			return nil
		},
	}
}

type fakePassthroughProvider struct {
	chatCalled bool
	chatResp   *llm.ChatResponse
	chatErr    error

	openAIPath     string
	openAIBody     []byte
	openAIResponse *http.Response
	openAIErr      error
	openAIAPIs     map[string]struct{}

	anthropicPath     string
	anthropicBody     []byte
	anthropicResponse *http.Response
	anthropicErr      error
	anthropicAPIs     map[string]struct{}
}

func (f *fakePassthroughProvider) Name() string { return "fake-passthrough" }

func (f *fakePassthroughProvider) Chat(context.Context, llm.ChatRequest) (*llm.ChatResponse, error) {
	f.chatCalled = true
	if f.chatResp != nil || f.chatErr != nil {
		return f.chatResp, f.chatErr
	}
	return nil, errors.New("normalized path should not be called")
}

func (f *fakePassthroughProvider) ChatStream(context.Context, llm.ChatRequest) (*providerapi.StreamReader, error) {
	f.chatCalled = true
	return nil, errors.New("normalized path should not be called")
}

func (f *fakePassthroughProvider) ForwardOpenAI(_ context.Context, path string, body []byte, _ http.Header) (*http.Response, error) {
	f.openAIPath = path
	f.openAIBody = append([]byte(nil), body...)
	return f.openAIResponse, f.openAIErr
}

func (f *fakePassthroughProvider) ForwardAnthropic(_ context.Context, path string, body []byte, _ http.Header) (*http.Response, error) {
	f.anthropicPath = path
	f.anthropicBody = append([]byte(nil), body...)
	return f.anthropicResponse, f.anthropicErr
}

func (f *fakePassthroughProvider) SupportsOpenAIAPI(apiType string) bool {
	if len(f.openAIAPIs) == 0 {
		return true
	}
	_, ok := f.openAIAPIs[apiType]
	return ok
}

func (f *fakePassthroughProvider) SupportsAnthropicAPI(apiType string) bool {
	if len(f.anthropicAPIs) == 0 {
		return true
	}
	_, ok := f.anthropicAPIs[apiType]
	return ok
}

func (f *fakePassthroughProvider) SupportsPassthrough(protocol, apiType string) bool {
	switch protocol {
	case "openai":
		return f.SupportsOpenAIAPI(apiType)
	case "anthropic":
		return f.SupportsAnthropicAPI(apiType)
	default:
		return false
	}
}

func (f *fakePassthroughProvider) Forward(ctx context.Context, protocol, path string, body []byte, headers http.Header) (*http.Response, error) {
	switch protocol {
	case "openai":
		return f.ForwardOpenAI(ctx, path, body, headers)
	case "anthropic":
		return f.ForwardAnthropic(ctx, path, body, headers)
	default:
		return nil, errors.New("unsupported protocol")
	}
}

func mustTestRouter(t *testing.T, providers map[string]chatProvider, externalModel, providerName, backendModel string) *routing.Router {
	t.Helper()

	router, err := routing.New(&config.Config{
		ModelRoutes: []config.ModelRoute{{
			ExternalModel: externalModel,
			Targets: []config.Target{{
				Provider:     providerName,
				BackendModel: backendModel,
			}},
		}},
	}, providers)
	if err != nil {
		t.Fatalf("routing.New() error = %v", err)
	}
	return router
}
