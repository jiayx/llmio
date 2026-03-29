package providers

import (
	"fmt"

	"github.com/jiayx/llmio/internal/config"
	anthropicprovider "github.com/jiayx/llmio/internal/providers/anthropic"
	providerapi "github.com/jiayx/llmio/internal/providers/api"
	openaiprovider "github.com/jiayx/llmio/internal/providers/openai"
)

// NewAdapter constructs a provider adapter from config.
func NewAdapter(cfg config.ProviderConfig) (providerapi.ProviderAdapter, error) {
	switch cfg.Type {
	case "openai-compatible":
		return openaiprovider.NewOpenAICompatible(cfg), nil
	case "anthropic-native":
		return anthropicprovider.NewAnthropicNative(cfg), nil
	default:
		return nil, fmt.Errorf("unsupported provider type %q", cfg.Type)
	}
}
