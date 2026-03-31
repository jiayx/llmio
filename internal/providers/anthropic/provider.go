package anthropic

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/jiayx/llmio/internal/config"
	"github.com/jiayx/llmio/internal/llm"
	anthropicproto "github.com/jiayx/llmio/internal/protocols/anthropic"
	providerapi "github.com/jiayx/llmio/internal/providers/api"
	providershared "github.com/jiayx/llmio/internal/providers/shared"
)

// AnthropicNative implements a normalized provider backed by Anthropic's native API.
type AnthropicNative struct {
	name       string
	httpClient *providershared.HTTPClient
	supported  map[string]struct{}
}

// NewAnthropicNative constructs an Anthropic-native provider from config.
func NewAnthropicNative(cfg config.ProviderConfig) *AnthropicNative {
	return &AnthropicNative{
		name: cfg.Name,
		httpClient: providershared.NewHTTPClient(cfg.BaseURL, cfg.Headers, nil, func(req *http.Request) {
			if cfg.APIKey != "" {
				req.Header.Set("x-api-key", cfg.APIKey)
			}
			req.Header.Set("anthropic-version", "2023-06-01")
		}),
		supported: providershared.NewAPISupportSet(cfg.SupportedAPITypes),
	}
}

// Name returns the configured provider name.
func (p *AnthropicNative) Name() string {
	return p.name
}

// NativeProtocol reports the provider's native protocol family.
func (p *AnthropicNative) NativeProtocol() string {
	return "anthropic"
}

// SupportsAnthropicAPI reports whether a native Anthropic API type should use passthrough.
func (p *AnthropicNative) SupportsAnthropicAPI(apiType string) bool {
	return providershared.SupportsAPIType(p.supported, apiType)
}

// SupportsPassthrough reports whether this adapter can natively forward the given request shape.
func (p *AnthropicNative) SupportsPassthrough(protocol, apiType string) bool {
	return protocol == "anthropic" && p.SupportsAnthropicAPI(apiType)
}

// ForwardAnthropic forwards a native Anthropic request without normalization.
func (p *AnthropicNative) ForwardAnthropic(ctx context.Context, path string, body []byte, headers http.Header) (*http.Response, error) {
	return p.httpClient.Do(ctx, http.MethodPost, path, body, headers)
}

// Forward forwards a native request using the adapter's protocol-aware passthrough path.
func (p *AnthropicNative) Forward(ctx context.Context, protocol, path string, body []byte, headers http.Header) (*http.Response, error) {
	if protocol != "anthropic" {
		return nil, fmt.Errorf("provider %s does not support %s passthrough", p.name, protocol)
	}
	return p.ForwardAnthropic(ctx, path, body, headers)
}

// Chat executes a normalized non-streaming chat request.
func (p *AnthropicNative) Chat(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	payload := req.RawRequestBody
	if !canUseNativeAnthropicMessagesRequest(req) {
		var err error
		payload, err = json.Marshal(llmToAnthropic(req))
		if err != nil {
			return nil, fmt.Errorf("marshal anthropic request: %w", err)
		}
	}

	resp, err := p.httpClient.Do(ctx, http.MethodPost, "/messages", payload, nil)
	if err != nil {
		return nil, err
	}
	defer providershared.CloseResponseBody("anthropic chat", resp.Body)

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read provider response: %w", err)
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("provider %s returned status %d: %s", p.name, resp.StatusCode, strings.TrimSpace(string(data)))
	}

	var out anthropicproto.MessagesResponse
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("decode provider response: %w", err)
	}

	return anthropicToLLMResponse(out, data), nil
}

// ChatStream executes a normalized streaming chat request.
func (p *AnthropicNative) ChatStream(ctx context.Context, req llm.ChatRequest) (*providerapi.StreamReader, error) {
	body := req.RawRequestBody
	if !canUseNativeAnthropicMessagesRequest(req) {
		payload := llmToAnthropic(req)
		payload.Stream = true

		var err error
		body, err = json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("marshal anthropic request: %w", err)
		}
	}

	resp, err := p.httpClient.Do(ctx, http.MethodPost, "/messages", body, nil)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		defer providershared.CloseResponseBody("anthropic chat stream error", resp.Body)
		data, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return nil, fmt.Errorf("provider %s returned status %d", p.name, resp.StatusCode)
		}
		return nil, fmt.Errorf("provider %s returned status %d: %s", p.name, resp.StatusCode, strings.TrimSpace(string(data)))
	}

	events := make(chan llm.StreamEvent, 16)
	errs := make(chan error, 1)

	go func() {
		defer close(events)
		defer close(errs)
		defer providershared.CloseResponseBody("anthropic chat stream", resp.Body)

		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

		var eventName string
		blockState := make(map[int]anthropicproto.ContentBlock)
		for scanner.Scan() {
			line := scanner.Text()
			trimmed := strings.TrimSpace(line)
			if trimmed == "" || strings.HasPrefix(trimmed, ":") {
				continue
			}
			if strings.HasPrefix(trimmed, "event:") {
				eventName = strings.TrimSpace(strings.TrimPrefix(trimmed, "event:"))
				continue
			}
			if !strings.HasPrefix(trimmed, "data:") {
				continue
			}

			payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
			if err := anthropicSSEToLLMEvents(eventName, payload, blockState, events); err != nil {
				errs <- err
				return
			}
			if eventName == "message_stop" {
				return
			}
		}

		if err := scanner.Err(); err != nil {
			errs <- fmt.Errorf("read anthropic stream: %w", err)
		}
	}()

	return &providerapi.StreamReader{
		Events: events,
		Err:    errs,
		Close:  resp.Body.Close,
	}, nil
}

func canUseNativeAnthropicMessagesRequest(req llm.ChatRequest) bool {
	return req.SourceProtocol == "anthropic" && req.SourceAPIType == "messages" && len(req.RawRequestBody) > 0
}
