package sqlite

import "time"

// UsageTotals aggregates token consumption for one API key.
type UsageTotals struct {
	RequestCount             int        `json:"request_count"`
	PricedRequestCount       int        `json:"priced_request_count"`
	InputTokens              int        `json:"input_tokens"`
	CachedInputTokens        int        `json:"cached_input_tokens"`
	CacheReadInputTokens     int        `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int        `json:"cache_creation_input_tokens"`
	OutputTokens             int        `json:"output_tokens"`
	TotalTokens              int        `json:"total_tokens"`
	EstimatedCostUSD         float64    `json:"estimated_cost_usd"`
	LastUsedAt               *time.Time `json:"last_used_at,omitempty"`
}

// KeySummary is the externally visible API key metadata.
type KeySummary struct {
	ID                 string      `json:"id"`
	Name               string      `json:"name"`
	Prefix             string      `json:"prefix"`
	BudgetUSD          *float64    `json:"budget_usd,omitempty"`
	RemainingBudgetUSD *float64    `json:"remaining_budget_usd,omitempty"`
	CreatedAt          time.Time   `json:"created_at"`
	DisabledAt         *time.Time  `json:"disabled_at,omitempty"`
	Usage              UsageTotals `json:"usage"`
}

// UsageReport returns aggregated usage with key identity.
type UsageReport struct {
	KeyID              string      `json:"key_id"`
	KeyName            string      `json:"key_name"`
	KeyPrefix          string      `json:"key_prefix"`
	BudgetUSD          *float64    `json:"budget_usd,omitempty"`
	RemainingBudgetUSD *float64    `json:"remaining_budget_usd,omitempty"`
	DisabledAt         *time.Time  `json:"disabled_at,omitempty"`
	Usage              UsageTotals `json:"usage"`
}

// CreateResult returns the generated secret exactly once.
type CreateResult struct {
	Key    KeySummary `json:"key"`
	Secret string     `json:"secret"`
}
