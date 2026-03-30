package protocols

import (
	"net/http"

	providerapi "github.com/jiayx/llmio/internal/providers/api"
)

const (
	ProtocolOpenAI    = "openai"
	ProtocolAnthropic = "anthropic"

	APIChatCompletions = providerapi.OpenAIAPIChatCompletions
	APIResponses       = providerapi.OpenAIAPIResponses
	APIMessages        = providerapi.AnthropicAPIMessages
	APIModels          = providerapi.OpenAIAPIModels
)

// RequestMeta captures the protocol identity of an inbound request.
type RequestMeta struct {
	Protocol      string
	APIType       string
	UpstreamPath  string
	ExternalModel string
	APIKeyID      string
	APIKeyName    string
	Body          []byte
	Headers       http.Header
	Stream        bool
}

// Endpoint describes one inbound protocol endpoint and its upstream passthrough path.
type Endpoint struct {
	APIType      string
	InboundPath  string
	UpstreamPath string
}

func (e Endpoint) upstreamPath() string {
	if e.UpstreamPath != "" {
		return e.UpstreamPath
	}
	return e.InboundPath
}

func (e Endpoint) match(path string) bool {
	return e.InboundPath == path
}

// ProtocolAdapter describes one inbound protocol surface and its bound endpoints.
type ProtocolAdapter interface {
	Protocol() string
	Endpoints() []Endpoint
	Match(path string) (Endpoint, bool)
	Request(endpoint Endpoint, externalModel string, body []byte, headers http.Header, stream bool) RequestMeta
}

// OpenAIAdapter describes the OpenAI-compatible client surface.
type OpenAIAdapter struct{}

func (OpenAIAdapter) Protocol() string {
	return ProtocolOpenAI
}

func (OpenAIAdapter) Endpoints() []Endpoint {
	return []Endpoint{
		{APIType: APIModels, InboundPath: "/v1/models", UpstreamPath: "/models"},
		{APIType: APIModels, InboundPath: "/models", UpstreamPath: "/models"},
		{APIType: APIChatCompletions, InboundPath: "/v1/chat/completions", UpstreamPath: "/chat/completions"},
		{APIType: APIChatCompletions, InboundPath: "/chat/completions", UpstreamPath: "/chat/completions"},
		{APIType: APIResponses, InboundPath: "/v1/responses", UpstreamPath: "/responses"},
		{APIType: APIResponses, InboundPath: "/responses", UpstreamPath: "/responses"},
	}
}

func (a OpenAIAdapter) Match(path string) (Endpoint, bool) {
	for _, endpoint := range a.Endpoints() {
		if endpoint.match(path) {
			return endpoint, true
		}
	}
	return Endpoint{}, false
}

func (OpenAIAdapter) Request(endpoint Endpoint, externalModel string, body []byte, headers http.Header, stream bool) RequestMeta {
	return RequestMeta{
		Protocol:      ProtocolOpenAI,
		APIType:       endpoint.APIType,
		UpstreamPath:  endpoint.upstreamPath(),
		ExternalModel: externalModel,
		Body:          body,
		Headers:       headers,
		Stream:        stream,
	}
}

// AnthropicAdapter describes the Anthropic-compatible client surface.
type AnthropicAdapter struct{}

func (AnthropicAdapter) Protocol() string {
	return ProtocolAnthropic
}

func (AnthropicAdapter) Endpoints() []Endpoint {
	return []Endpoint{
		{APIType: APIMessages, InboundPath: "/anthropic/v1/messages", UpstreamPath: "/messages"},
	}
}

func (a AnthropicAdapter) Match(path string) (Endpoint, bool) {
	for _, endpoint := range a.Endpoints() {
		if endpoint.match(path) {
			return endpoint, true
		}
	}
	return Endpoint{}, false
}

func (AnthropicAdapter) Request(endpoint Endpoint, externalModel string, body []byte, headers http.Header, stream bool) RequestMeta {
	return RequestMeta{
		Protocol:      ProtocolAnthropic,
		APIType:       endpoint.APIType,
		UpstreamPath:  endpoint.upstreamPath(),
		ExternalModel: externalModel,
		Body:          body,
		Headers:       headers,
		Stream:        stream,
	}
}

// DefaultAdapters returns the built-in client protocol adapters.
func DefaultAdapters() []ProtocolAdapter {
	return []ProtocolAdapter{
		OpenAIAdapter{},
		AnthropicAdapter{},
	}
}

// LookupEndpoint resolves an inbound path to the owning adapter and endpoint metadata.
func LookupEndpoint(adapters []ProtocolAdapter, path string) (ProtocolAdapter, Endpoint, bool) {
	if len(adapters) == 0 {
		adapters = DefaultAdapters()
	}
	for _, adapter := range adapters {
		if endpoint, ok := adapter.Match(path); ok {
			return adapter, endpoint, true
		}
	}
	return nil, Endpoint{}, false
}
