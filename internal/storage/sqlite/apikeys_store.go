package sqlite

import (
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/jiayx/llmio/internal/authctx"
)

// APIKeyStore persists managed API keys and usage in SQLite.
type APIKeyStore struct {
	db *sql.DB
	mu sync.Mutex
}

// NewAPIKeyStore constructs an API key store on top of the shared SQLite database.
func NewAPIKeyStore(db *sql.DB) *APIKeyStore {
	return &APIKeyStore{db: db}
}

func (s *APIKeyStore) BudgetExceeded(id string) bool {
	row := s.db.QueryRow(`
		SELECT k.budget_usd, u.estimated_cost_usd
		FROM api_keys k
		LEFT JOIN api_key_usage u ON u.api_key_id = k.id
		WHERE k.id = ? AND k.disabled_at IS NULL
	`, id)

	var (
		budget sql.NullFloat64
		cost   sql.NullFloat64
	)
	if err := row.Scan(&budget, &cost); err != nil {
		return false
	}
	if !budget.Valid {
		return false
	}
	return cost.Float64 >= budget.Float64
}

// Authenticate validates an API key secret.
func (s *APIKeyStore) Authenticate(secret string) (authctx.Principal, bool) {
	secret = strings.TrimSpace(secret)
	if secret == "" {
		return authctx.Principal{}, false
	}

	row := s.db.QueryRow(`
		SELECT id, name, budget_usd
		FROM api_keys
		WHERE secret_hash = ? AND disabled_at IS NULL
	`, hashSecret(secret))

	var (
		principal authctx.Principal
		budgetUSD sql.NullFloat64
	)
	if err := row.Scan(&principal.ID, &principal.Name, &budgetUSD); err != nil {
		return authctx.Principal{}, false
	}
	principal.Managed = true
	if budgetUSD.Valid {
		value := budgetUSD.Float64
		principal.BudgetUSD = &value
	}
	return principal, true
}

// Create inserts a new API key and returns the secret once.
func (s *APIKeyStore) Create(name string, budgetUSD *float64) (CreateResult, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return CreateResult{}, fmt.Errorf("name is required")
	}
	if budgetUSD != nil && *budgetUSD < 0 {
		return CreateResult{}, fmt.Errorf("budget_usd must be >= 0")
	}

	secret, prefix, err := generateSecret()
	if err != nil {
		return CreateResult{}, err
	}
	now := time.Now().UTC()
	key := KeySummary{
		ID:        newID(now),
		Name:      name,
		Prefix:    prefix,
		CreatedAt: now,
	}
	if budgetUSD != nil {
		value := *budgetUSD
		key.BudgetUSD = &value
		key.RemainingBudgetUSD = &value
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	tx, err := s.db.Begin()
	if err != nil {
		return CreateResult{}, fmt.Errorf("begin create api key: %w", err)
	}
	defer tx.Rollback()

	if _, err := tx.Exec(`
		INSERT INTO api_keys (id, name, prefix, budget_usd, created_at, secret_hash)
		VALUES (?, ?, ?, ?, ?, ?)
	`, key.ID, key.Name, key.Prefix, nullableFloat64(budgetUSD), key.CreatedAt.Format(time.RFC3339Nano), hashSecret(secret)); err != nil {
		return CreateResult{}, fmt.Errorf("insert api key: %w", err)
	}
	if _, err := tx.Exec(`
		INSERT INTO api_key_usage (api_key_id)
		VALUES (?)
	`, key.ID); err != nil {
		return CreateResult{}, fmt.Errorf("insert api key usage: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return CreateResult{}, fmt.Errorf("commit create api key: %w", err)
	}

	return CreateResult{
		Key:    key,
		Secret: secret,
	}, nil
}

// Disable marks an API key as disabled.
func (s *APIKeyStore) Disable(id string) (KeySummary, error) {
	now := time.Now().UTC()
	result, err := s.db.Exec(`
		UPDATE api_keys
		SET disabled_at = COALESCE(disabled_at, ?)
		WHERE id = ?
	`, now.Format(time.RFC3339Nano), id)
	if err != nil {
		return KeySummary{}, fmt.Errorf("disable api key %s: %w", id, err)
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return KeySummary{}, fmt.Errorf("disable api key %s: %w", id, err)
	}
	if rows == 0 {
		return KeySummary{}, os.ErrNotExist
	}

	key, ok := s.Get(id)
	if !ok {
		return KeySummary{}, os.ErrNotExist
	}
	return key, nil
}

// Get returns API key metadata and usage summary.
func (s *APIKeyStore) Get(id string) (KeySummary, bool) {
	row := s.db.QueryRow(`
		SELECT
			k.id,
			k.name,
			k.prefix,
			k.budget_usd,
			k.created_at,
			k.disabled_at,
			u.request_count,
			u.priced_request_count,
			u.input_tokens,
			u.cached_input_tokens,
			u.cache_read_input_tokens,
			u.cache_creation_input_tokens,
			u.output_tokens,
			u.total_tokens,
			u.estimated_cost_usd,
			u.last_used_at
		FROM api_keys k
		LEFT JOIN api_key_usage u ON u.api_key_id = k.id
		WHERE k.id = ?
	`, id)

	key, err := scanKeySummary(row)
	if err != nil {
		return KeySummary{}, false
	}
	return key, true
}

// List returns all managed API keys.
func (s *APIKeyStore) List() []KeySummary {
	rows, err := s.db.Query(`
		SELECT
			k.id,
			k.name,
			k.prefix,
			k.budget_usd,
			k.created_at,
			k.disabled_at,
			u.request_count,
			u.priced_request_count,
			u.input_tokens,
			u.cached_input_tokens,
			u.cache_read_input_tokens,
			u.cache_creation_input_tokens,
			u.output_tokens,
			u.total_tokens,
			u.estimated_cost_usd,
			u.last_used_at
		FROM api_keys k
		LEFT JOIN api_key_usage u ON u.api_key_id = k.id
		ORDER BY k.created_at DESC, k.id DESC
	`)
	if err != nil {
		return nil
	}
	defer rows.Close()

	out := make([]KeySummary, 0)
	for rows.Next() {
		key, err := scanKeySummary(rows)
		if err != nil {
			continue
		}
		out = append(out, key)
	}
	return out
}

// UsageByID returns usage for one managed API key.
func (s *APIKeyStore) UsageByID(id string) (UsageReport, bool) {
	row := s.db.QueryRow(`
		SELECT
			k.id,
			k.name,
			k.prefix,
			k.budget_usd,
			k.disabled_at,
			u.request_count,
			u.priced_request_count,
			u.input_tokens,
			u.cached_input_tokens,
			u.cache_read_input_tokens,
			u.cache_creation_input_tokens,
			u.output_tokens,
			u.total_tokens,
			u.estimated_cost_usd,
			u.last_used_at
		FROM api_keys k
		LEFT JOIN api_key_usage u ON u.api_key_id = k.id
		WHERE k.id = ?
	`, id)

	report, err := scanUsageReport(row)
	if err != nil {
		return UsageReport{}, false
	}
	return report, true
}

// UsageReports returns usage for all managed API keys and the grand total.
func (s *APIKeyStore) UsageReports() ([]UsageReport, UsageTotals) {
	rows, err := s.db.Query(`
		SELECT
			k.id,
			k.name,
			k.prefix,
			k.budget_usd,
			k.disabled_at,
			u.request_count,
			u.priced_request_count,
			u.input_tokens,
			u.cached_input_tokens,
			u.cache_read_input_tokens,
			u.cache_creation_input_tokens,
			u.output_tokens,
			u.total_tokens,
			u.estimated_cost_usd,
			u.last_used_at
		FROM api_keys k
		LEFT JOIN api_key_usage u ON u.api_key_id = k.id
		ORDER BY u.total_tokens DESC, k.created_at DESC, k.id DESC
	`)
	if err != nil {
		return nil, UsageTotals{}
	}
	defer rows.Close()

	out := make([]UsageReport, 0)
	total := UsageTotals{}
	for rows.Next() {
		report, err := scanUsageReport(rows)
		if err != nil {
			continue
		}
		out = append(out, report)
		total.RequestCount += report.Usage.RequestCount
		total.PricedRequestCount += report.Usage.PricedRequestCount
		total.InputTokens += report.Usage.InputTokens
		total.CachedInputTokens += report.Usage.CachedInputTokens
		total.CacheReadInputTokens += report.Usage.CacheReadInputTokens
		total.CacheCreationInputTokens += report.Usage.CacheCreationInputTokens
		total.OutputTokens += report.Usage.OutputTokens
		total.TotalTokens += report.Usage.TotalTokens
		total.EstimatedCostUSD += report.Usage.EstimatedCostUSD
		if report.Usage.LastUsedAt != nil && (total.LastUsedAt == nil || report.Usage.LastUsedAt.After(*total.LastUsedAt)) {
			last := *report.Usage.LastUsedAt
			total.LastUsedAt = &last
		}
	}
	return out, total
}

func scanKeySummary(scanner interface {
	Scan(dest ...any) error
}) (KeySummary, error) {
	var (
		key       KeySummary
		budget    sql.NullFloat64
		disabled  sql.NullString
		lastUsed  sql.NullString
		createdAt string
	)
	if err := scanner.Scan(
		&key.ID,
		&key.Name,
		&key.Prefix,
		&budget,
		&createdAt,
		&disabled,
		&key.Usage.RequestCount,
		&key.Usage.PricedRequestCount,
		&key.Usage.InputTokens,
		&key.Usage.CachedInputTokens,
		&key.Usage.CacheReadInputTokens,
		&key.Usage.CacheCreationInputTokens,
		&key.Usage.OutputTokens,
		&key.Usage.TotalTokens,
		&key.Usage.EstimatedCostUSD,
		&lastUsed,
	); err != nil {
		return KeySummary{}, err
	}
	created, err := time.Parse(time.RFC3339Nano, createdAt)
	if err != nil {
		return KeySummary{}, err
	}
	key.CreatedAt = created
	if budget.Valid {
		value := budget.Float64
		key.BudgetUSD = &value
		remaining := budget.Float64 - key.Usage.EstimatedCostUSD
		key.RemainingBudgetUSD = &remaining
	}
	if disabled.Valid {
		t, err := time.Parse(time.RFC3339Nano, disabled.String)
		if err != nil {
			return KeySummary{}, err
		}
		key.DisabledAt = &t
	}
	if lastUsed.Valid {
		t, err := time.Parse(time.RFC3339Nano, lastUsed.String)
		if err != nil {
			return KeySummary{}, err
		}
		key.Usage.LastUsedAt = &t
	}
	return key, nil
}

func scanUsageReport(scanner interface {
	Scan(dest ...any) error
}) (UsageReport, error) {
	var (
		report   UsageReport
		budget   sql.NullFloat64
		disabled sql.NullString
		lastUsed sql.NullString
	)
	if err := scanner.Scan(
		&report.KeyID,
		&report.KeyName,
		&report.KeyPrefix,
		&budget,
		&disabled,
		&report.Usage.RequestCount,
		&report.Usage.PricedRequestCount,
		&report.Usage.InputTokens,
		&report.Usage.CachedInputTokens,
		&report.Usage.CacheReadInputTokens,
		&report.Usage.CacheCreationInputTokens,
		&report.Usage.OutputTokens,
		&report.Usage.TotalTokens,
		&report.Usage.EstimatedCostUSD,
		&lastUsed,
	); err != nil {
		return UsageReport{}, err
	}
	if budget.Valid {
		value := budget.Float64
		report.BudgetUSD = &value
		remaining := budget.Float64 - report.Usage.EstimatedCostUSD
		report.RemainingBudgetUSD = &remaining
	}
	if disabled.Valid {
		t, err := time.Parse(time.RFC3339Nano, disabled.String)
		if err != nil {
			return UsageReport{}, err
		}
		report.DisabledAt = &t
	}
	if lastUsed.Valid {
		t, err := time.Parse(time.RFC3339Nano, lastUsed.String)
		if err != nil {
			return UsageReport{}, err
		}
		report.Usage.LastUsedAt = &t
	}
	return report, nil
}

func hashSecret(secret string) string {
	sum := sha256.Sum256([]byte(secret))
	return hex.EncodeToString(sum[:])
}

func newID(now time.Time) string {
	return fmt.Sprintf("key_%d", now.UnixNano())
}

func generateSecret() (string, string, error) {
	buf := make([]byte, 18)
	if _, err := rand.Read(buf); err != nil {
		return "", "", fmt.Errorf("generate api key: %w", err)
	}
	encoded := hex.EncodeToString(buf)
	secret := "llmio_" + encoded
	prefix := secret
	if len(prefix) > 14 {
		prefix = prefix[:14]
	}
	return secret, prefix, nil
}

func nullableFloat64(v *float64) any {
	if v == nil {
		return nil
	}
	return *v
}
