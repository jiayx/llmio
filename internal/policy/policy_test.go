package policy

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/jiayx/llmio/internal/llm"
	"github.com/jiayx/llmio/internal/protocols"
	providerapi "github.com/jiayx/llmio/internal/providers/api"
	"github.com/jiayx/llmio/internal/routing"
)

func TestExecuteChatFallsBackOnRetryableError(t *testing.T) {
	p := New(Config{
		RequestTimeout:       time.Second,
		StreamSetupTimeout:   time.Second,
		BreakerFailureThresh: 3,
		BreakerOpenFor:       time.Second,
	}, map[string]providerapi.ProviderAdapter{
		"primary":   policyStubProvider{chatErr: errors.New("provider primary returned status 503: busy")},
		"secondary": policyStubProvider{chatResp: &llm.ChatResponse{ID: "ok"}},
	})

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

type policyStubProvider struct {
	chatResp *llm.ChatResponse
	chatErr  error

	statuses     []int
	forwardCalls int
}

func (p policyStubProvider) Name() string { return "stub" }

func (p policyStubProvider) Chat(context.Context, llm.ChatRequest) (*llm.ChatResponse, error) {
	return p.chatResp, p.chatErr
}

func (p policyStubProvider) ChatStream(context.Context, llm.ChatRequest) (*providerapi.StreamReader, error) {
	return nil, errors.New("not implemented")
}

func (p *policyStubProvider) SupportsPassthrough(protocol, apiType string) bool {
	return protocol == protocols.ProtocolOpenAI && apiType == protocols.APIChatCompletions
}

func (p *policyStubProvider) Forward(_ context.Context, _, _ string, _ []byte, _ http.Header) (*http.Response, error) {
	p.forwardCalls++
	status := http.StatusOK
	if len(p.statuses) >= p.forwardCalls {
		status = p.statuses[p.forwardCalls-1]
	}
	body := io.NopCloser(bytes.NewBufferString("ok"))
	if status != http.StatusOK {
		body = io.NopCloser(bytes.NewBufferString("retry"))
	}
	return &http.Response{
		StatusCode: status,
		Body:       body,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
	}, nil
}
