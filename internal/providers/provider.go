package providers

import (
	"context"
	"net/http"

	"github.com/jiayx/llmio/internal/core"
)

// ChatProvider handles normalized chat requests for a backend model provider.
type ChatProvider interface {
	Name() string
	Chat(ctx context.Context, req core.ChatRequest) (*core.ChatResponse, error)
	ChatStream(ctx context.Context, req core.ChatRequest) (*StreamReader, error)
}

// OpenAIPassthroughProvider forwards native OpenAI-compatible requests.
type OpenAIPassthroughProvider interface {
	ForwardOpenAI(ctx context.Context, path string, body []byte, headers http.Header) (*http.Response, error)
}

// AnthropicPassthroughProvider forwards native Anthropic-compatible requests.
type AnthropicPassthroughProvider interface {
	ForwardAnthropic(ctx context.Context, path string, body []byte, headers http.Header) (*http.Response, error)
}

const (
	OpenAIAPIChatCompletions = "chat_completions"
	OpenAIAPIResponses       = "responses"
	OpenAIAPIModels          = "models"
	AnthropicAPIMessages     = "messages"
)

// OpenAPITypeSupporter reports which native API types a provider can passthrough.
type OpenAPITypeSupporter interface {
	SupportsOpenAIAPI(apiType string) bool
}

// AnthropicAPITypeSupporter reports which Anthropic API types a provider can passthrough.
type AnthropicAPITypeSupporter interface {
	SupportsAnthropicAPI(apiType string) bool
}

// StreamReader exposes a normalized provider event stream and its lifecycle hooks.
type StreamReader struct {
	Events <-chan core.StreamEvent
	Err    <-chan error
	Close  func() error
}
