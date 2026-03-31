package openai

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/jiayx/llmio/internal/config"
	"github.com/jiayx/llmio/internal/llm"
	"github.com/jiayx/llmio/internal/observability"
	openaiproto "github.com/jiayx/llmio/internal/protocols/openai"
	providerapi "github.com/jiayx/llmio/internal/providers/api"
	providershared "github.com/jiayx/llmio/internal/providers/shared"
)

// OpenAICompatible implements a normalized provider backed by an OpenAI-compatible API.
type OpenAICompatible struct {
	name       string
	modelsPath string
	httpClient *providershared.HTTPClient
	supported  map[string]struct{}
}

// NewOpenAICompatible constructs an OpenAI-compatible provider from config.
func NewOpenAICompatible(cfg config.ProviderConfig) *OpenAICompatible {
	return &OpenAICompatible{
		name:       cfg.Name,
		modelsPath: cfg.ModelsPath,
		httpClient: providershared.NewHTTPClient(cfg.BaseURL, cfg.Headers, nil, func(req *http.Request) {
			if cfg.APIKey != "" {
				req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
			}
		}),
		supported: providershared.NewAPISupportSet(cfg.SupportedAPITypes),
	}
}

// Name returns the configured provider name.
func (p *OpenAICompatible) Name() string {
	return p.name
}

// NativeProtocol reports the provider's native protocol family.
func (p *OpenAICompatible) NativeProtocol() string {
	return "openai"
}

func (p *OpenAICompatible) NativeChatPath() string {
	return "/chat/completions"
}

// SupportsOpenAIAPI reports whether a native OpenAI API type should use passthrough.
func (p *OpenAICompatible) SupportsOpenAIAPI(apiType string) bool {
	return providershared.SupportsAPIType(p.supported, apiType)
}

// SupportsPassthrough reports whether this adapter can natively forward the given request shape.
func (p *OpenAICompatible) SupportsPassthrough(protocol, apiType string) bool {
	return protocol == "openai" && p.SupportsOpenAIAPI(apiType)
}

// ForwardOpenAI forwards a native OpenAI-compatible request without normalization.
func (p *OpenAICompatible) ForwardOpenAI(ctx context.Context, path string, body []byte, headers http.Header) (*http.Response, error) {
	return p.httpClient.Do(ctx, http.MethodPost, path, body, headers)
}

// Forward forwards a native request using the adapter's protocol-aware passthrough path.
func (p *OpenAICompatible) Forward(ctx context.Context, protocol, path string, body []byte, headers http.Header) (*http.Response, error) {
	if protocol != "openai" {
		return nil, fmt.Errorf("provider %s does not support %s passthrough", p.name, protocol)
	}
	return p.ForwardOpenAI(ctx, path, body, headers)
}

// Chat executes a normalized non-streaming chat request.
func (p *OpenAICompatible) Chat(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error) {
	payload := req.RawRequestBody
	if !canUseNativeOpenAIChatRequest(req) {
		var err error
		payload, err = json.Marshal(llmToOpenAI(req))
		if err != nil {
			return nil, fmt.Errorf("marshal openai request: %w", err)
		}
	}

	resp, err := p.httpClient.Do(ctx, http.MethodPost, "/chat/completions", payload, nil)
	if err != nil {
		return nil, err
	}
	defer providershared.CloseResponseBody("openai chat", resp.Body)

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read provider response: %w", err)
	}
	if observability.Enabled() {
		slog.Debug("openai provider response body trace",
			"provider", p.name,
			"path", "/chat/completions",
			"body", observability.Bytes(data),
		)
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("provider %s returned status %d: %s", p.name, resp.StatusCode, strings.TrimSpace(string(data)))
	}

	var out openaiproto.ChatCompletionResponse
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("decode provider response: %w", err)
	}

	return openAIToLLMResponse(out, data), nil
}

// ChatStream executes a normalized streaming chat request.
func (p *OpenAICompatible) ChatStream(ctx context.Context, req llm.ChatRequest) (*providerapi.StreamReader, error) {
	body := req.RawRequestBody
	if !canUseNativeOpenAIChatRequest(req) {
		payload := llmToOpenAI(req)
		payload.Stream = true

		var err error
		body, err = json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("marshal openai request: %w", err)
		}
	}

	resp, err := p.httpClient.Do(ctx, http.MethodPost, "/chat/completions", body, nil)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		defer providershared.CloseResponseBody("openai chat stream error", resp.Body)
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
		defer providershared.CloseResponseBody("openai chat stream", resp.Body)

		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		var pendingStop *llm.StreamEvent
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" || strings.HasPrefix(line, ":") || !strings.HasPrefix(line, "data:") {
				continue
			}

			payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if observability.Enabled() {
				slog.Debug("openai provider stream chunk trace",
					"provider", p.name,
					"path", "/chat/completions",
					"payload", observability.String(payload),
				)
			}
			if payload == "[DONE]" {
				if pendingStop != nil {
					events <- *pendingStop
				} else {
					events <- llm.StreamEvent{Type: llm.StreamEventStop}
				}
				return
			}

			streamEvents, err := openAIStreamChunkToLLMEvents(payload)
			if err != nil {
				errs <- err
				return
			}
			for _, event := range streamEvents {
				if observability.Enabled() {
					slog.Debug("openai provider stream event trace",
						"provider", p.name,
						"type", event.Type,
						"text_delta", observability.String(event.TextDelta),
						"input_tokens", event.InputTokens,
						"output_tokens", event.OutputTokens,
						"finish_reason", event.FinishReason,
					)
				}
				if event.Type == llm.StreamEventStop {
					stopEvent := event
					pendingStop = &stopEvent
					continue
				}
				events <- event
			}
		}

		if err := scanner.Err(); err != nil {
			errs <- fmt.Errorf("read openai stream: %w", err)
			return
		}
		if pendingStop != nil {
			events <- *pendingStop
		}
	}()

	return &providerapi.StreamReader{
		Events: events,
		Err:    errs,
		Close:  resp.Body.Close,
	}, nil
}

func canUseNativeOpenAIChatRequest(req llm.ChatRequest) bool {
	return req.SourceProtocol == "openai" && req.SourceAPIType == "chat_completions" && len(req.RawRequestBody) > 0
}
