package providerapi

import (
	"context"
	"net/http"

	"github.com/jiayx/llmio/internal/llm"
)

// ChatProvider handles normalized chat requests for a backend model provider.
type ChatProvider interface {
	Name() string
	Chat(ctx context.Context, req llm.ChatRequest) (*llm.ChatResponse, error)
	ChatStream(ctx context.Context, req llm.ChatRequest) (*StreamReader, error)
}

// ProviderAdapter is the backend-facing adapter contract used by the gateway.
type ProviderAdapter = ChatProvider

// NativeProtocolReporter reports the provider's native request protocol family.
type NativeProtocolReporter interface {
	NativeProtocol() string
}

// OpenAIPassthroughProvider forwards native OpenAI-compatible requests.
type OpenAIPassthroughProvider interface {
	ForwardOpenAI(ctx context.Context, upstreamPath string, body []byte, headers http.Header) (*http.Response, error)
}

// AnthropicPassthroughProvider forwards native Anthropic-compatible requests.
type AnthropicPassthroughProvider interface {
	ForwardAnthropic(ctx context.Context, upstreamPath string, body []byte, headers http.Header) (*http.Response, error)
}

// PassthroughSupporter reports whether a provider can natively forward a request without normalization.
type PassthroughSupporter interface {
	SupportsPassthrough(protocol, apiType string) bool
}

// PassthroughForwarder forwards a native request using protocol-aware dispatch.
type PassthroughForwarder interface {
	Forward(ctx context.Context, protocol, upstreamPath string, body []byte, headers http.Header) (*http.Response, error)
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
	Events <-chan llm.StreamEvent
	Err    <-chan error
	Close  func() error
}
