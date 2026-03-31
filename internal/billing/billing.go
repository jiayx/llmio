package billing

import (
	"context"
	"strings"

	"github.com/jiayx/llmio/internal/config"
	"github.com/jiayx/llmio/internal/usage"
)

type Scheme string

const (
	SchemeGeneric   Scheme = "generic"
	SchemeOpenAI    Scheme = "openai"
	SchemeAnthropic Scheme = "anthropic"
)

type RateCard struct {
	Provider                      string
	BackendModel                  string
	Scheme                        Scheme
	InputPer1MTokens              float64
	CachedInputPer1MTokens        float64
	CacheReadInputPer1MTokens     float64
	CacheCreationInputPer1MTokens float64
	OutputPer1MTokens             float64
}

type Catalog struct {
	cards map[string]RateCard
}

func NewCatalog(rules []config.PricingRule) *Catalog {
	if len(rules) == 0 {
		return &Catalog{}
	}
	cards := make(map[string]RateCard, len(rules))
	for _, rule := range rules {
		card := RateCard{
			Provider:                      rule.Provider,
			BackendModel:                  rule.BackendModel,
			Scheme:                        Scheme(strings.ToLower(strings.TrimSpace(rule.Scheme))),
			InputPer1MTokens:              rule.InputPer1MTokens,
			CachedInputPer1MTokens:        rule.CachedInputPer1MTokens,
			CacheReadInputPer1MTokens:     rule.CacheReadInputPer1MTokens,
			CacheCreationInputPer1MTokens: rule.CacheCreationInputPer1MTokens,
			OutputPer1MTokens:             rule.OutputPer1MTokens,
		}
		if card.Scheme == "" {
			card.Scheme = SchemeGeneric
		}
		cards[key(rule.Provider, rule.BackendModel)] = card
	}
	return &Catalog{cards: cards}
}

func (c *Catalog) Estimate(event usage.Event) (float64, bool) {
	if c == nil || len(c.cards) == 0 {
		return 0, false
	}
	card, ok := c.cards[key(event.ProviderName, event.BackendModel)]
	if !ok {
		return 0, false
	}

	const perMillion = 1_000_000.0
	cost := 0.0
	switch card.Scheme {
	case SchemeOpenAI:
		uncachedInput := event.InputTokens - event.CachedInputTokens
		if uncachedInput < 0 {
			uncachedInput = 0
		}
		cost += float64(uncachedInput) / perMillion * card.InputPer1MTokens
		cost += float64(event.CachedInputTokens) / perMillion * card.CachedInputPer1MTokens
	case SchemeAnthropic:
		cost += float64(event.InputTokens) / perMillion * card.InputPer1MTokens
		cost += float64(event.CacheReadInputTokens) / perMillion * card.CacheReadInputPer1MTokens
		cost += float64(event.CacheCreationInputTokens) / perMillion * card.CacheCreationInputPer1MTokens
	default:
		if event.CachedInputTokens > 0 && card.CachedInputPer1MTokens > 0 {
			uncachedInput := event.InputTokens - event.CachedInputTokens
			if uncachedInput < 0 {
				uncachedInput = 0
			}
			cost += float64(uncachedInput) / perMillion * card.InputPer1MTokens
			cost += float64(event.CachedInputTokens) / perMillion * card.CachedInputPer1MTokens
		} else {
			cost += float64(event.InputTokens) / perMillion * card.InputPer1MTokens
		}
		cost += float64(event.CacheReadInputTokens) / perMillion * card.CacheReadInputPer1MTokens
		cost += float64(event.CacheCreationInputTokens) / perMillion * card.CacheCreationInputPer1MTokens
	}
	cost += float64(event.OutputTokens) / perMillion * card.OutputPer1MTokens
	return cost, true
}

type PricingRecorder struct {
	Catalog *Catalog
	Next    usage.Recorder
}

func (r PricingRecorder) Record(ctx context.Context, event usage.Event) {
	if r.Catalog != nil {
		if cost, ok := r.Catalog.Estimate(event); ok {
			event.EstimatedCostUSD = cost
			event.CostKnown = true
		}
	}
	if r.Next != nil {
		r.Next.Record(ctx, event)
	}
}

func key(provider, backendModel string) string {
	return strings.TrimSpace(provider) + "\x00" + strings.TrimSpace(backendModel)
}
