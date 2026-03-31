package billing

import (
	"testing"

	"github.com/jiayx/llmio/internal/config"
	"github.com/jiayx/llmio/internal/usage"
)

func TestEstimateOpenAICostWithCachedTokens(t *testing.T) {
	catalog := NewCatalog([]config.PricingRule{{
		Provider:               "openai",
		BackendModel:           "gpt-5.4",
		Scheme:                 "openai",
		InputPer1MTokens:       2.5,
		CachedInputPer1MTokens: 0.25,
		OutputPer1MTokens:      15,
	}})

	cost, ok := catalog.Estimate(usage.Event{
		ProviderName:      "openai",
		BackendModel:      "gpt-5.4",
		InputTokens:       100000,
		CachedInputTokens: 60000,
		OutputTokens:      20000,
	})
	if !ok {
		t.Fatal("expected cost to be known")
	}
	want := 0.415
	if diff := cost - want; diff < -1e-9 || diff > 1e-9 {
		t.Fatalf("cost = %v want %v", cost, want)
	}
}

func TestEstimateAnthropicCostWithCacheTokens(t *testing.T) {
	catalog := NewCatalog([]config.PricingRule{{
		Provider:                      "anthropic",
		BackendModel:                  "claude-sonnet-4",
		Scheme:                        "anthropic",
		InputPer1MTokens:              3,
		CacheReadInputPer1MTokens:     0.3,
		CacheCreationInputPer1MTokens: 3.75,
		OutputPer1MTokens:             15,
	}})

	cost, ok := catalog.Estimate(usage.Event{
		ProviderName:             "anthropic",
		BackendModel:             "claude-sonnet-4",
		InputTokens:              100000,
		CacheReadInputTokens:     60000,
		CacheCreationInputTokens: 40000,
		OutputTokens:             20000,
	})
	if !ok {
		t.Fatal("expected cost to be known")
	}
	want := 0.768
	if diff := cost - want; diff < -1e-9 || diff > 1e-9 {
		t.Fatalf("cost = %v want %v", cost, want)
	}
}
