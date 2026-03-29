package clients

import (
	"net/http"
	"strings"
)

const (
	ProtocolOpenAI    = "openai"
	ProtocolAnthropic = "anthropic"

	APIChatCompletions = "chat_completions"
	APIResponses       = "responses"
	APIMessages        = "messages"
	APIModels          = "models"
)

// RequestMeta captures the client-side protocol identity of a request.
type RequestMeta struct {
	Protocol      string
	APIType       string
	Path          string
	ExternalModel string
	Body          []byte
	Headers       http.Header
	Stream        bool
}

// ClientAdapter describes a client-side protocol surface.
type ClientAdapter interface {
	Protocol() string
	SupportsPath(path string) bool
}

// OpenAIAdapter describes the OpenAI-compatible client surface.
type OpenAIAdapter struct{}

func (OpenAIAdapter) Protocol() string {
	return ProtocolOpenAI
}

func (OpenAIAdapter) SupportsPath(path string) bool {
	switch path {
	case "/v1/models", "/models", "/v1/chat/completions", "/chat/completions", "/v1/responses", "/responses":
		return true
	default:
		return false
	}
}

func (OpenAIAdapter) Request(apiType, path, externalModel string, body []byte, headers http.Header, stream bool) RequestMeta {
	return RequestMeta{
		Protocol:      ProtocolOpenAI,
		APIType:       apiType,
		Path:          path,
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

func (AnthropicAdapter) SupportsPath(path string) bool {
	return path == "/anthropic/v1/messages"
}

func (AnthropicAdapter) Request(apiType, path, externalModel string, body []byte, headers http.Header, stream bool) RequestMeta {
	return RequestMeta{
		Protocol:      ProtocolAnthropic,
		APIType:       apiType,
		Path:          path,
		ExternalModel: externalModel,
		Body:          body,
		Headers:       headers,
		Stream:        stream,
	}
}

// ProtocolForPath infers the client protocol from the inbound URL path.
func ProtocolForPath(path string) string {
	if strings.HasPrefix(path, "/anthropic/") {
		return ProtocolAnthropic
	}
	return ProtocolOpenAI
}
