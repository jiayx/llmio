package sqlite

import (
	"context"
	"database/sql"
	"time"

	"github.com/jiayx/llmio/internal/usage"
)

// UsageStore persists aggregated usage in SQLite.
type UsageStore struct {
	db *sql.DB
}

// NewUsageStore constructs a usage store on top of the shared SQLite database.
func NewUsageStore(db *sql.DB) *UsageStore {
	return &UsageStore{db: db}
}

// Record stores usage under the authenticated managed API key.
func (s *UsageStore) Record(_ context.Context, event usage.Event) {
	if event.APIKeyID == "" {
		return
	}

	lastUsedAt := event.Timestamp.UTC().Format(time.RFC3339Nano)
	_, _ = s.db.Exec(`
		UPDATE api_key_usage
		SET
			request_count = request_count + 1,
			priced_request_count = priced_request_count + ?,
			input_tokens = input_tokens + ?,
			cached_input_tokens = cached_input_tokens + ?,
			cache_read_input_tokens = cache_read_input_tokens + ?,
			cache_creation_input_tokens = cache_creation_input_tokens + ?,
			output_tokens = output_tokens + ?,
			total_tokens = total_tokens + ?,
			estimated_cost_usd = estimated_cost_usd + ?,
			last_used_at = ?
		WHERE api_key_id = ?
	`, boolToInt(event.CostKnown), event.InputTokens, event.CachedInputTokens, event.CacheReadInputTokens, event.CacheCreationInputTokens, event.OutputTokens, event.TotalTokens, event.EstimatedCostUSD, lastUsedAt, event.APIKeyID)
}

func boolToInt(v bool) int {
	if v {
		return 1
	}
	return 0
}
