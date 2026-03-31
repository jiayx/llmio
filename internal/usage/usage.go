package usage

import (
	"context"
	"log/slog"
	"time"
)

// Event captures one successful model usage record.
type Event struct {
	Timestamp                time.Time
	APIKeyID                 string
	APIKeyName               string
	ProviderName             string
	ExternalModel            string
	BackendModel             string
	Protocol                 string
	APIType                  string
	InputTokens              int
	CachedInputTokens        int
	CacheReadInputTokens     int
	CacheCreationInputTokens int
	OutputTokens             int
	TotalTokens              int
	EstimatedCostUSD         float64
	CostKnown                bool
	Stream                   bool
	Passthrough              bool
	UsageKnown               bool
}

// Recorder stores or emits usage events.
type Recorder interface {
	Record(context.Context, Event)
}

// SlogRecorder writes usage events to structured logs.
type SlogRecorder struct{}

// Record emits one usage event.
func (SlogRecorder) Record(_ context.Context, event Event) {
	slog.Info("usage recorded",
		"timestamp", event.Timestamp.Format(time.RFC3339),
		"api_key_id", event.APIKeyID,
		"api_key_name", event.APIKeyName,
		"provider", event.ProviderName,
		"external_model", event.ExternalModel,
		"backend_model", event.BackendModel,
		"protocol", event.Protocol,
		"api_type", event.APIType,
		"input_tokens", event.InputTokens,
		"cached_input_tokens", event.CachedInputTokens,
		"cache_read_input_tokens", event.CacheReadInputTokens,
		"cache_creation_input_tokens", event.CacheCreationInputTokens,
		"output_tokens", event.OutputTokens,
		"total_tokens", event.TotalTokens,
		"estimated_cost_usd", event.EstimatedCostUSD,
		"cost_known", event.CostKnown,
		"stream", event.Stream,
		"passthrough", event.Passthrough,
		"usage_known", event.UsageKnown,
	)
}

// NopRecorder drops all usage events.
type NopRecorder struct{}

// Record drops the event.
func (NopRecorder) Record(context.Context, Event) {}

// MultiRecorder fans out one event to multiple recorders.
type MultiRecorder []Recorder

// Record forwards the event to every non-nil recorder.
func (m MultiRecorder) Record(ctx context.Context, event Event) {
	for _, recorder := range m {
		if recorder == nil {
			continue
		}
		recorder.Record(ctx, event)
	}
}
