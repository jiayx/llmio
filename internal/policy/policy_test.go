package policy

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/jiayx/llmio/internal/llm"
	"github.com/jiayx/llmio/internal/protocols"
	providerapi "github.com/jiayx/llmio/internal/providers/api"
	"github.com/jiayx/llmio/internal/routing"
	"github.com/jiayx/llmio/internal/usage"
)

func TestExecuteChatFallsBackOnRetryableError(t *testing.T) {
	recorder := &usageSpy{}
	p := NewWithRecorder(Config{
		RequestTimeout:       time.Second,
		StreamSetupTimeout:   time.Second,
		BreakerFailureThresh: 3,
		BreakerOpenFor:       time.Second,
	}, map[string]providerapi.ProviderAdapter{
		"primary":   &policyStubProvider{chatErr: errors.New("provider primary returned status 503: busy")},
		"secondary": &policyStubProvider{chatResp: &llm.ChatResponse{ID: "ok", InputTokens: 11, OutputTokens: 7}},
	}, recorder)

	resp, err := p.ExecuteChat(context.Background(), routing.Route{
		Targets: []routing.Target{
			{ProviderName: "primary", BackendModel: "a"},
			{ProviderName: "secondary", BackendModel: "b"},
		},
	}, llm.ChatRequest{Model: "proxy"})
	if err != nil {
		t.Fatalf("ExecuteChat() error = %v", err)
	}
	if resp.ID != "ok" {
		t.Fatalf("resp = %#v", resp)
	}
	if len(recorder.events) != 1 {
		t.Fatalf("usage events = %#v", recorder.events)
	}
	if recorder.events[0].ProviderName != "secondary" || recorder.events[0].BackendModel != "b" || recorder.events[0].ExternalModel != "proxy" {
		t.Fatalf("usage event = %#v", recorder.events[0])
	}
	if recorder.events[0].InputTokens != 11 || recorder.events[0].OutputTokens != 7 || !recorder.events[0].UsageKnown {
		t.Fatalf("usage event = %#v", recorder.events[0])
	}
}

func TestExecuteChatStripsIgnoredRequestFields(t *testing.T) {
	temp := 0.7
	topP := 0.9
	primary := &policyStubProvider{
		chatResp: &llm.ChatResponse{ID: "ok"},
	}
	p := New(DefaultConfig(), map[string]providerapi.ProviderAdapter{
		"primary": primary,
	})

	_, err := p.ExecuteChat(context.Background(), routing.Route{
		Targets: []routing.Target{{
			ProviderName:        "primary",
			BackendModel:        "kimi-k2.5",
			IgnoreRequestFields: []string{"temperature", "top_p", "max_output_tokens"},
		}},
	}, llm.ChatRequest{
		Model:       "proxy-model",
		MaxTokens:   256,
		Temperature: &temp,
		TopP:        &topP,
	})
	if err != nil {
		t.Fatalf("ExecuteChat() error = %v", err)
	}
	if primary.lastChatReq.Model != "kimi-k2.5" {
		t.Fatalf("model = %q", primary.lastChatReq.Model)
	}
	if primary.lastChatReq.Temperature != nil || primary.lastChatReq.TopP != nil || primary.lastChatReq.MaxTokens != 0 {
		t.Fatalf("request = %#v", primary.lastChatReq)
	}
}

func TestExecuteChatUsesRewrittenRawBodyForSameProtocolTransform(t *testing.T) {
	primary := &policyStubProvider{
		chatResp:       &llm.ChatResponse{ID: "ok"},
		nativeProtocol: "openai",
	}
	p := New(DefaultConfig(), map[string]providerapi.ProviderAdapter{
		"primary": primary,
	})

	_, err := p.ExecuteChat(context.Background(), routing.Route{
		Targets: []routing.Target{{
			ProviderName:        "primary",
			BackendModel:        "kimi-k2.5",
			IgnoreRequestFields: []string{"temperature"},
		}},
	}, llm.ChatRequest{
		Model:          "proxy-model",
		SourceProtocol: "openai",
		SourceAPIType:  "chat_completions",
		RawRequestBody: []byte(`{"model":"proxy-model","temperature":0.7,"reasoning_content":"think first","vendor_field":{"x":1}}`),
	})
	if err != nil {
		t.Fatalf("ExecuteChat() error = %v", err)
	}
	if !bytes.Contains(primary.lastChatReq.RawRequestBody, []byte(`"model":"kimi-k2.5"`)) {
		t.Fatalf("raw = %s", string(primary.lastChatReq.RawRequestBody))
	}
	if bytes.Contains(primary.lastChatReq.RawRequestBody, []byte(`"temperature"`)) {
		t.Fatalf("raw = %s", string(primary.lastChatReq.RawRequestBody))
	}
	if !bytes.Contains(primary.lastChatReq.RawRequestBody, []byte(`"vendor_field":{"x":1}`)) {
		t.Fatalf("raw = %s", string(primary.lastChatReq.RawRequestBody))
	}
}

func TestExecutePassthroughUsesBreakerAfterRetryableFailures(t *testing.T) {
	primary := &policyStubProvider{
		statuses: []int{http.StatusServiceUnavailable},
	}
	secondary := &policyStubProvider{
		statuses: []int{http.StatusOK, http.StatusOK},
	}

	p := New(Config{
		RequestTimeout:       time.Second,
		StreamSetupTimeout:   time.Second,
		BreakerFailureThresh: 1,
		BreakerOpenFor:       time.Minute,
	}, map[string]providerapi.ProviderAdapter{
		"primary":   primary,
		"secondary": secondary,
	})

	route := routing.Route{
		Targets: []routing.Target{
			{ProviderName: "primary", BackendModel: "a"},
			{ProviderName: "secondary", BackendModel: "b"},
		},
	}
	meta := protocols.RequestMeta{
		Protocol:      protocols.ProtocolOpenAI,
		APIType:       protocols.APIChatCompletions,
		UpstreamPath:  "/chat/completions",
		ExternalModel: "proxy",
		Body:          []byte(`{"model":"proxy"}`),
		Headers:       http.Header{},
	}

	resp, handled, err := p.ExecutePassthrough(context.Background(), route, meta)
	if err != nil || !handled {
		t.Fatalf("first ExecutePassthrough() handled=%v err=%v", handled, err)
	}
	body, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		t.Fatalf("read body: %v", readErr)
	}
	if string(body) != "ok" {
		t.Fatalf("body = %q", body)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("close body: %v", err)
	}

	_, handled, err = p.ExecutePassthrough(context.Background(), route, meta)
	if err != nil || !handled {
		t.Fatalf("second ExecutePassthrough() handled=%v err=%v", handled, err)
	}
	if primary.forwardCalls != 1 {
		t.Fatalf("primary forward calls = %d", primary.forwardCalls)
	}
	if secondary.forwardCalls != 2 {
		t.Fatalf("secondary forward calls = %d", secondary.forwardCalls)
	}
}

func TestExecutePassthroughRecordsUsage(t *testing.T) {
	recorder := &usageSpy{}
	p := NewWithRecorder(DefaultConfig(), map[string]providerapi.ProviderAdapter{
		"primary": &policyStubProvider{
			forwardResponse: &http.Response{
				StatusCode: http.StatusOK,
				Body: io.NopCloser(bytes.NewBufferString(`{
					"id":"resp_1",
					"object":"response",
					"model":"backend-model",
					"status":"completed",
					"output":[],
					"usage":{"input_tokens":9,"output_tokens":4,"total_tokens":13}
				}`)),
				Header: http.Header{"Content-Type": []string{"application/json"}},
			},
		},
	}, recorder)

	resp, handled, err := p.ExecutePassthrough(context.Background(), routing.Route{
		Targets: []routing.Target{{ProviderName: "primary", BackendModel: "backend-model"}},
	}, protocols.RequestMeta{
		Protocol:      protocols.ProtocolOpenAI,
		APIType:       protocols.APIResponses,
		UpstreamPath:  "/responses",
		ExternalModel: "proxy-model",
		Body:          []byte(`{"model":"proxy-model"}`),
		Headers:       http.Header{},
	})
	if err != nil || !handled {
		t.Fatalf("ExecutePassthrough() handled=%v err=%v", handled, err)
	}
	if resp == nil {
		t.Fatalf("response is nil")
	}
	if len(recorder.events) != 1 {
		t.Fatalf("usage events = %#v", recorder.events)
	}
	event := recorder.events[0]
	if event.ProviderName != "primary" || event.BackendModel != "backend-model" || event.ExternalModel != "proxy-model" {
		t.Fatalf("usage event = %#v", event)
	}
	if event.Protocol != protocols.ProtocolOpenAI || event.APIType != protocols.APIResponses || !event.Passthrough {
		t.Fatalf("usage event = %#v", event)
	}
	if event.InputTokens != 9 || event.OutputTokens != 4 || !event.UsageKnown {
		t.Fatalf("usage event = %#v", event)
	}
}

func TestExecutePassthroughKeepsNonStreamingBodyReadable(t *testing.T) {
	p := New(DefaultConfig(), map[string]providerapi.ProviderAdapter{
		"primary": &policyStubProvider{
			forwardFunc: func(ctx context.Context, _, _ string, _ []byte, _ http.Header) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Body: &ctxBoundReadCloser{
						ctx:  ctx,
						data: []byte(`{"id":"1","object":"chat.completion","model":"backend-model","choices":[]}`),
					},
					Header: http.Header{"Content-Type": []string{"application/json"}},
				}, nil
			},
		},
	})

	resp, handled, err := p.ExecutePassthrough(context.Background(), routing.Route{
		Targets: []routing.Target{{ProviderName: "primary", BackendModel: "backend-model"}},
	}, protocols.RequestMeta{
		Protocol:      protocols.ProtocolOpenAI,
		APIType:       protocols.APIChatCompletions,
		UpstreamPath:  "/chat/completions",
		ExternalModel: "proxy-model",
		Body:          []byte(`{"model":"proxy-model"}`),
		Headers:       http.Header{},
	})
	if err != nil || !handled {
		t.Fatalf("ExecutePassthrough() handled=%v err=%v", handled, err)
	}
	body, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		t.Fatalf("read body: %v", readErr)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("close body: %v", err)
	}
	if !bytes.Contains(body, []byte(`"model":"backend-model"`)) {
		t.Fatalf("body = %s", string(body))
	}
}

func TestExecutePassthroughStripsIgnoredRequestFields(t *testing.T) {
	primary := &policyStubProvider{
		forwardFunc: func(_ context.Context, _, _ string, body []byte, _ http.Header) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       io.NopCloser(bytes.NewReader(body)),
				Header:     http.Header{"Content-Type": []string{"application/json"}},
			}, nil
		},
	}
	p := New(DefaultConfig(), map[string]providerapi.ProviderAdapter{
		"primary": primary,
	})

	resp, handled, err := p.ExecutePassthrough(context.Background(), routing.Route{
		Targets: []routing.Target{{
			ProviderName:        "primary",
			BackendModel:        "kimi-k2.5",
			IgnoreRequestFields: []string{"temperature", "max_tokens"},
		}},
	}, protocols.RequestMeta{
		Protocol:      protocols.ProtocolOpenAI,
		APIType:       protocols.APIChatCompletions,
		UpstreamPath:  "/chat/completions",
		ExternalModel: "proxy-model",
		Body:          []byte(`{"model":"proxy-model","temperature":0.7,"max_tokens":256,"messages":[{"role":"user","content":"hello"}]}`),
		Headers:       http.Header{},
	})
	if err != nil || !handled {
		t.Fatalf("ExecutePassthrough() handled=%v err=%v", handled, err)
	}
	body, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		t.Fatalf("read body: %v", readErr)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("close body: %v", err)
	}
	if bytes.Contains(body, []byte(`"temperature"`)) || bytes.Contains(body, []byte(`"max_tokens"`)) {
		t.Fatalf("body = %s", string(body))
	}
	if !bytes.Contains(body, []byte(`"model":"kimi-k2.5"`)) {
		t.Fatalf("body = %s", string(body))
	}
}

func TestExecutePassthroughFallsBackOnUnsupportedStatus(t *testing.T) {
	recorder := &usageSpy{}
	primary := &policyStubProvider{
		forwardResponse: &http.Response{
			StatusCode: http.StatusNotFound,
			Body:       io.NopCloser(bytes.NewBufferString("")),
			Header:     http.Header{"Content-Type": []string{"application/json"}},
		},
	}
	secondary := &policyStubProvider{
		forwardResponse: &http.Response{
			StatusCode: http.StatusOK,
			Body: io.NopCloser(bytes.NewBufferString(`{
				"id":"resp_1",
				"object":"response",
				"model":"backend-model-2",
				"status":"completed",
				"output":[],
				"usage":{"input_tokens":9,"output_tokens":4,"total_tokens":13}
			}`)),
			Header: http.Header{"Content-Type": []string{"application/json"}},
		},
	}
	p := NewWithRecorder(DefaultConfig(), map[string]providerapi.ProviderAdapter{
		"primary":   primary,
		"secondary": secondary,
	}, recorder)

	resp, handled, err := p.ExecutePassthrough(context.Background(), routing.Route{
		Targets: []routing.Target{
			{ProviderName: "primary", BackendModel: "backend-model-1"},
			{ProviderName: "secondary", BackendModel: "backend-model-2"},
		},
	}, protocols.RequestMeta{
		Protocol:      protocols.ProtocolOpenAI,
		APIType:       protocols.APIResponses,
		UpstreamPath:  "/responses",
		ExternalModel: "proxy-model",
		Body:          []byte(`{"model":"proxy-model"}`),
		Headers:       http.Header{},
	})
	if err != nil || !handled {
		t.Fatalf("ExecutePassthrough() handled=%v err=%v", handled, err)
	}
	if resp == nil {
		t.Fatalf("response is nil")
	}
	if primary.forwardCalls != 1 || secondary.forwardCalls != 1 {
		t.Fatalf("forward calls: primary=%d secondary=%d", primary.forwardCalls, secondary.forwardCalls)
	}
	if len(recorder.events) != 1 || recorder.events[0].ProviderName != "secondary" {
		t.Fatalf("usage events = %#v", recorder.events)
	}
}

func TestExecutePassthroughRecordsUsageForResponsesSSE(t *testing.T) {
	recorder := &usageSpy{}
	p := NewWithRecorder(DefaultConfig(), map[string]providerapi.ProviderAdapter{
		"primary": &policyStubProvider{
			forwardResponse: &http.Response{
				StatusCode: http.StatusOK,
				Body: io.NopCloser(bytes.NewBufferString("" +
					"event: response.created\n" +
					"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n" +
					"event: response.completed\n" +
					"data: {\"id\":\"resp_1\",\"object\":\"response\",\"model\":\"backend-model\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":9,\"output_tokens\":4,\"total_tokens\":13}}\n\n")),
				Header: http.Header{"Content-Type": []string{"text/event-stream"}},
			},
		},
	}, recorder)

	resp, handled, err := p.ExecutePassthrough(context.Background(), routing.Route{
		Targets: []routing.Target{{ProviderName: "primary", BackendModel: "backend-model"}},
	}, protocols.RequestMeta{
		Protocol:      protocols.ProtocolOpenAI,
		APIType:       protocols.APIResponses,
		UpstreamPath:  "/responses",
		ExternalModel: "proxy-model",
		Body:          []byte(`{"model":"proxy-model","stream":true}`),
		Headers:       http.Header{},
		Stream:        true,
	})
	if err != nil || !handled {
		t.Fatalf("ExecutePassthrough() handled=%v err=%v", handled, err)
	}
	if _, err := io.ReadAll(resp.Body); err != nil {
		t.Fatalf("read passthrough body: %v", err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("close passthrough body: %v", err)
	}
	if len(recorder.events) != 1 {
		t.Fatalf("usage events = %#v", recorder.events)
	}
	event := recorder.events[0]
	if !event.Stream || !event.Passthrough || !event.UsageKnown {
		t.Fatalf("usage event = %#v", event)
	}
	if event.InputTokens != 9 || event.OutputTokens != 4 {
		t.Fatalf("usage event = %#v", event)
	}
}

func TestExecutePassthroughKeepsSSEBodyReadable(t *testing.T) {
	recorder := &usageSpy{}
	p := NewWithRecorder(DefaultConfig(), map[string]providerapi.ProviderAdapter{
		"primary": &policyStubProvider{
			forwardFunc: func(ctx context.Context, _, _ string, _ []byte, _ http.Header) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Body: &ctxBoundReadCloser{
						ctx: ctx,
						data: []byte("" +
							"event: response.created\n" +
							"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n" +
							"event: response.completed\n" +
							"data: {\"id\":\"resp_1\",\"object\":\"response\",\"model\":\"backend-model\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":9,\"output_tokens\":4,\"total_tokens\":13}}\n\n"),
					},
					Header: http.Header{"Content-Type": []string{"text/event-stream"}},
				}, nil
			},
		},
	}, recorder)

	resp, handled, err := p.ExecutePassthrough(context.Background(), routing.Route{
		Targets: []routing.Target{{ProviderName: "primary", BackendModel: "backend-model"}},
	}, protocols.RequestMeta{
		Protocol:      protocols.ProtocolOpenAI,
		APIType:       protocols.APIResponses,
		UpstreamPath:  "/responses",
		ExternalModel: "proxy-model",
		Body:          []byte(`{"model":"proxy-model","stream":true}`),
		Headers:       http.Header{},
		Stream:        true,
	})
	if err != nil || !handled {
		t.Fatalf("ExecutePassthrough() handled=%v err=%v", handled, err)
	}
	body, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		t.Fatalf("read passthrough sse body: %v", readErr)
	}
	if err := resp.Body.Close(); err != nil {
		t.Fatalf("close passthrough sse body: %v", err)
	}
	if !bytes.Contains(body, []byte("event: response.completed")) {
		t.Fatalf("body = %s", string(body))
	}
	if len(recorder.events) != 1 {
		t.Fatalf("usage events = %#v", recorder.events)
	}
	event := recorder.events[0]
	if !event.Stream || !event.Passthrough || !event.UsageKnown {
		t.Fatalf("usage event = %#v", event)
	}
	if event.InputTokens != 9 || event.OutputTokens != 4 {
		t.Fatalf("usage event = %#v", event)
	}
}

func TestExecuteStreamRecordsUsageOnGracefulEOFWithoutStopEvent(t *testing.T) {
	recorder := &usageSpy{}
	events := make(chan llm.StreamEvent, 1)
	events <- llm.StreamEvent{Type: llm.StreamEventUsage, InputTokens: 12, OutputTokens: 5}
	close(events)
	errs := make(chan error)
	close(errs)

	p := NewWithRecorder(DefaultConfig(), map[string]providerapi.ProviderAdapter{
		"primary": &policyStubProvider{
			stream: &providerapi.StreamReader{
				Events: events,
				Err:    errs,
				Close: func() error {
					return nil
				},
			},
		},
	}, recorder)

	stream, err := p.ExecuteStream(context.Background(), routing.Route{
		Targets: []routing.Target{{ProviderName: "primary", BackendModel: "backend-model"}},
	}, llm.ChatRequest{Model: "proxy-model", Stream: true})
	if err != nil {
		t.Fatalf("ExecuteStream() error = %v", err)
	}
	for range stream.Events {
	}
	if len(recorder.events) != 1 {
		t.Fatalf("usage events = %#v", recorder.events)
	}
	event := recorder.events[0]
	if !event.Stream || event.Passthrough || !event.UsageKnown {
		t.Fatalf("usage event = %#v", event)
	}
	if event.InputTokens != 12 || event.OutputTokens != 5 {
		t.Fatalf("usage event = %#v", event)
	}
}

type policyStubProvider struct {
	chatResp       *llm.ChatResponse
	chatErr        error
	lastChatReq    llm.ChatRequest
	nativeProtocol string
	stream         *providerapi.StreamReader
	streamErr      error
	streamFactory  func(context.Context, llm.ChatRequest) (*providerapi.StreamReader, error)
	forwardFunc    func(context.Context, string, string, []byte, http.Header) (*http.Response, error)

	statuses        []int
	forwardCalls    int
	forwardResponse *http.Response
}

func (p policyStubProvider) Name() string { return "stub" }

func (p *policyStubProvider) NativeProtocol() string { return p.nativeProtocol }

func (p *policyStubProvider) Chat(_ context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	p.lastChatReq = req
	return p.chatResp, p.chatErr
}

func (p policyStubProvider) ChatStream(ctx context.Context, req llm.ChatRequest) (*providerapi.StreamReader, error) {
	if p.streamFactory != nil {
		return p.streamFactory(ctx, req)
	}
	if p.streamErr != nil {
		return nil, p.streamErr
	}
	if p.stream != nil {
		return p.stream, nil
	}
	return nil, errors.New("not implemented")
}

func (p *policyStubProvider) SupportsPassthrough(protocol, apiType string) bool {
	return protocol == protocols.ProtocolOpenAI && (apiType == protocols.APIChatCompletions || apiType == protocols.APIResponses)
}

func (p *policyStubProvider) Forward(ctx context.Context, protocol, path string, body []byte, headers http.Header) (*http.Response, error) {
	p.forwardCalls++
	if p.forwardFunc != nil {
		return p.forwardFunc(ctx, protocol, path, body, headers)
	}
	if p.forwardResponse != nil {
		return p.forwardResponse, nil
	}
	status := http.StatusOK
	if len(p.statuses) >= p.forwardCalls {
		status = p.statuses[p.forwardCalls-1]
	}
	respBody := io.NopCloser(bytes.NewBufferString("ok"))
	if status != http.StatusOK {
		respBody = io.NopCloser(bytes.NewBufferString("retry"))
	}
	return &http.Response{
		StatusCode: status,
		Body:       respBody,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
	}, nil
}

type usageSpy struct {
	mu     sync.Mutex
	events []usage.Event
}

func (s *usageSpy) Record(_ context.Context, event usage.Event) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.events = append(s.events, event)
}

type ctxBoundReadCloser struct {
	ctx  context.Context
	data []byte
}

func (r *ctxBoundReadCloser) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	if len(r.data) == 0 {
		return 0, io.EOF
	}
	n := copy(p, r.data)
	r.data = r.data[n:]
	return n, nil
}

func (r *ctxBoundReadCloser) Close() error {
	return nil
}
