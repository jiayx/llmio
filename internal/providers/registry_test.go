package providers

import (
	"testing"

	"github.com/jiayx/llmio/internal/config"
	anthropicprovider "github.com/jiayx/llmio/internal/providers/anthropic"
	openaiprovider "github.com/jiayx/llmio/internal/providers/openai"
)

func TestNewAdapter(t *testing.T) {
	tests := []struct {
		name    string
		cfg     config.ProviderConfig
		wantErr bool
		assert  func(t *testing.T, adapter any)
	}{
		{
			name: "openai-compatible",
			cfg: config.ProviderConfig{
				Name:    "openai",
				Type:    "openai-compatible",
				BaseURL: "https://example.com/v1",
			},
			assert: func(t *testing.T, adapter any) {
				t.Helper()
				if _, ok := adapter.(*openaiprovider.OpenAICompatible); !ok {
					t.Fatalf("adapter type = %T", adapter)
				}
			},
		},
		{
			name: "anthropic-native",
			cfg: config.ProviderConfig{
				Name:    "anthropic",
				Type:    "anthropic-native",
				BaseURL: "https://example.com/v1",
			},
			assert: func(t *testing.T, adapter any) {
				t.Helper()
				if _, ok := adapter.(*anthropicprovider.AnthropicNative); !ok {
					t.Fatalf("adapter type = %T", adapter)
				}
			},
		},
		{
			name: "unsupported",
			cfg: config.ProviderConfig{
				Name:    "unknown",
				Type:    "custom-provider",
				BaseURL: "https://example.com/v1",
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			adapter, err := NewAdapter(tc.cfg)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("NewAdapter() error = nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("NewAdapter() error = %v", err)
			}
			tc.assert(t, adapter)
		})
	}
}

func TestNewAdapterResolvesProviderAPIKeyEnvReference(t *testing.T) {
	t.Setenv("TEST_PROVIDER_KEY", "from-env")

	adapter, err := NewAdapter(config.ProviderConfig{
		Name:    "openai",
		Type:    "openai-compatible",
		BaseURL: "https://example.com/v1",
		APIKey:  "${TEST_PROVIDER_KEY}",
	})
	if err != nil {
		t.Fatalf("NewAdapter() error = %v", err)
	}
	if _, ok := adapter.(*openaiprovider.OpenAICompatible); !ok {
		t.Fatalf("adapter type = %T", adapter)
	}
}

func TestNewAdapterRejectsInvalidProviderAPIKeyReference(t *testing.T) {
	_, err := NewAdapter(config.ProviderConfig{
		Name:    "openai",
		Type:    "openai-compatible",
		BaseURL: "https://example.com/v1",
		APIKey:  "prefix-${TEST_PROVIDER_KEY}",
	})
	if err == nil || err.Error() != `provider "openai" api_key must be plaintext or ${ENV_NAME}` {
		t.Fatalf("error = %v", err)
	}
}
