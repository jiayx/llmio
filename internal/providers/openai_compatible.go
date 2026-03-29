package providers

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/jiayx/llmio/internal/config"
	"github.com/jiayx/llmio/internal/core"
	openaiproto "github.com/jiayx/llmio/internal/protocols/openai"
)

// OpenAICompatible implements a normalized provider backed by an OpenAI-compatible API.
type OpenAICompatible struct {
	name       string
	modelsPath string
	httpClient *providerHTTPClient
	supported  map[string]struct{}
}

// NewOpenAICompatible constructs an OpenAI-compatible provider from config.
func NewOpenAICompatible(cfg config.ProviderConfig) *OpenAICompatible {
	return &OpenAICompatible{
		name:       cfg.Name,
		modelsPath: cfg.ModelsPath,
		httpClient: newProviderHTTPClient(cfg.BaseURL, cfg.Headers, nil, func(req *http.Request) {
			if cfg.APIKey != "" {
				req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
			}
		}),
		supported: newAPISupportSet(cfg.SupportedAPITypes),
	}
}

// Name returns the configured provider name.
func (p *OpenAICompatible) Name() string {
	return p.name
}

// SupportsOpenAIAPI reports whether a native OpenAI API type should use passthrough.
func (p *OpenAICompatible) SupportsOpenAIAPI(apiType string) bool {
	return supportsAPIType(p.supported, apiType)
}

// ForwardOpenAI forwards a native OpenAI-compatible request without normalization.
func (p *OpenAICompatible) ForwardOpenAI(ctx context.Context, path string, body []byte, headers http.Header) (*http.Response, error) {
	return p.httpClient.Do(ctx, http.MethodPost, path, body, headers)
}

// Chat executes a normalized non-streaming chat request.
func (p *OpenAICompatible) Chat(ctx context.Context, req core.ChatRequest) (*core.ChatResponse, error) {
	payload, err := json.Marshal(coreToOpenAI(req))
	if err != nil {
		return nil, fmt.Errorf("marshal openai request: %w", err)
	}

	resp, err := p.httpClient.Do(ctx, http.MethodPost, "/chat/completions", payload, nil)
	if err != nil {
		return nil, err
	}
	defer closeResponseBody("openai chat", resp.Body)

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read provider response: %w", err)
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("provider %s returned status %d: %s", p.name, resp.StatusCode, strings.TrimSpace(string(data)))
	}

	var out openaiproto.ChatCompletionResponse
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, fmt.Errorf("decode provider response: %w", err)
	}

	return openAIToCoreResponse(out, data), nil
}

// ChatStream executes a normalized streaming chat request.
func (p *OpenAICompatible) ChatStream(ctx context.Context, req core.ChatRequest) (*StreamReader, error) {
	payload := coreToOpenAI(req)
	payload.Stream = true

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal openai request: %w", err)
	}

	resp, err := p.httpClient.Do(ctx, http.MethodPost, "/chat/completions", body, nil)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		defer closeResponseBody("openai chat stream error", resp.Body)
		data, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return nil, fmt.Errorf("provider %s returned status %d", p.name, resp.StatusCode)
		}
		return nil, fmt.Errorf("provider %s returned status %d: %s", p.name, resp.StatusCode, strings.TrimSpace(string(data)))
	}

	events := make(chan core.StreamEvent, 16)
	errs := make(chan error, 1)

	go func() {
		defer close(events)
		defer close(errs)
		defer closeResponseBody("openai chat stream", resp.Body)

		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" || strings.HasPrefix(line, ":") || !strings.HasPrefix(line, "data:") {
				continue
			}

			payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if payload == "[DONE]" {
				events <- core.StreamEvent{Type: core.StreamEventStop}
				return
			}

			var chunk openaiproto.StreamChunk
			if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
				errs <- fmt.Errorf("decode openai stream chunk: %w", err)
				return
			}
			if len(chunk.Choices) > 0 {
				choice := chunk.Choices[0]
				if choice.Delta.Content != "" {
					events <- core.StreamEvent{
						Type:      core.StreamEventDelta,
						Part:      core.ContentPart{Type: core.ContentTypeText},
						TextDelta: choice.Delta.Content,
						Raw:       json.RawMessage(payload),
					}
				}
				if choice.Delta.Reasoning != "" {
					events <- core.StreamEvent{
						Type:      core.StreamEventDelta,
						Part:      core.ContentPart{Type: core.ContentTypeReasoning},
						TextDelta: choice.Delta.Reasoning,
						Raw:       json.RawMessage(payload),
					}
				}
				for _, toolCall := range choice.Delta.ToolCalls {
					if toolCall.ID == "" && toolCall.Function.Name == "" && toolCall.Function.Arguments == "" {
						continue
					}
					events <- core.StreamEvent{
						Type:       core.StreamEventTool,
						ToolIndex:  toolCall.Index,
						ToolCallID: toolCall.ID,
						ToolName:   toolCall.Function.Name,
						ToolInput:  toolCall.Function.Arguments,
						Raw:        json.RawMessage(payload),
					}
				}
				if choice.FinishReason != "" {
					events <- core.StreamEvent{
						Type:         core.StreamEventStop,
						FinishReason: choice.FinishReason,
						Raw:          json.RawMessage(payload),
					}
					return
				}
			}
			if chunk.Usage != nil {
				events <- core.StreamEvent{
					Type:         core.StreamEventUsage,
					InputTokens:  chunk.Usage.PromptTokens,
					OutputTokens: chunk.Usage.CompletionTokens,
					Raw:          json.RawMessage(payload),
				}
			}
		}

		if err := scanner.Err(); err != nil {
			errs <- fmt.Errorf("read openai stream: %w", err)
		}
	}()

	return &StreamReader{
		Events: events,
		Err:    errs,
		Close:  resp.Body.Close,
	}, nil
}
