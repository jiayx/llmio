package apikeys

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/jiayx/llmio/internal/usage"
	_ "modernc.org/sqlite"
)

type contextKey string

const principalContextKey contextKey = "apikey-principal"

// Principal identifies the authenticated caller.
type Principal struct {
	ID        string
	Name      string
	Managed   bool
	BudgetUSD *float64
}

// WithPrincipal stores the authenticated caller in context.
func WithPrincipal(ctx context.Context, principal Principal) context.Context {
	return context.WithValue(ctx, principalContextKey, principal)
}

// PrincipalFromContext returns the authenticated caller from context.
func PrincipalFromContext(ctx context.Context) (Principal, bool) {
	principal, ok := ctx.Value(principalContextKey).(Principal)
	return principal, ok
}

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

// Store persists managed API keys and their usage summary in SQLite.
type Store struct {
	db *sql.DB
	mu sync.Mutex
}

func (s *Store) BudgetExceeded(id string) bool {
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

// Open loads or initializes the API key store database.
func Open(path string) (*Store, error) {
	dsn := strings.TrimSpace(path)
	if dsn == "" {
		dsn = "file:llmio-apikeys?mode=memory&cache=shared"
	} else {
		if !filepath.IsAbs(dsn) && !strings.HasPrefix(dsn, "file:") {
			dsn = filepath.Clean(dsn)
		}
		if !strings.HasPrefix(dsn, "file:") {
			if err := os.MkdirAll(filepath.Dir(dsn), 0o755); err != nil {
				return nil, fmt.Errorf("mkdir sqlite dir %s: %w", filepath.Dir(dsn), err)
			}
		}
	}

	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open sqlite store %s: %w", path, err)
	}
	db.SetMaxOpenConns(1)

	s := &Store{db: db}
	if err := s.init(); err != nil {
		_ = db.Close()
		return nil, err
	}
	return s, nil
}

func (s *Store) init() error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS api_keys (
			id TEXT PRIMARY KEY,
			name TEXT NOT NULL,
			prefix TEXT NOT NULL,
			budget_usd REAL,
			created_at TEXT NOT NULL,
			disabled_at TEXT,
			secret_hash TEXT NOT NULL UNIQUE
		)`,
		`CREATE TABLE IF NOT EXISTS api_key_usage (
			api_key_id TEXT PRIMARY KEY,
			request_count INTEGER NOT NULL DEFAULT 0,
			priced_request_count INTEGER NOT NULL DEFAULT 0,
			input_tokens INTEGER NOT NULL DEFAULT 0,
			cached_input_tokens INTEGER NOT NULL DEFAULT 0,
			cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
			cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
			output_tokens INTEGER NOT NULL DEFAULT 0,
			total_tokens INTEGER NOT NULL DEFAULT 0,
			estimated_cost_usd REAL NOT NULL DEFAULT 0,
			last_used_at TEXT,
			FOREIGN KEY(api_key_id) REFERENCES api_keys(id) ON DELETE CASCADE
		)`,
	}
	for _, stmt := range stmts {
		if _, err := s.db.Exec(stmt); err != nil {
			return fmt.Errorf("init sqlite api key store: %w", err)
		}
	}
	if err := ensureColumn(s.db, "api_keys", "budget_usd", "REAL"); err != nil {
		return err
	}
	for _, column := range []struct {
		name string
		def  string
	}{
		{name: "priced_request_count", def: "INTEGER NOT NULL DEFAULT 0"},
		{name: "cached_input_tokens", def: "INTEGER NOT NULL DEFAULT 0"},
		{name: "cache_read_input_tokens", def: "INTEGER NOT NULL DEFAULT 0"},
		{name: "cache_creation_input_tokens", def: "INTEGER NOT NULL DEFAULT 0"},
		{name: "estimated_cost_usd", def: "REAL NOT NULL DEFAULT 0"},
	} {
		if err := ensureColumn(s.db, "api_key_usage", column.name, column.def); err != nil {
			return err
		}
	}
	return nil
}

// Authenticate validates an API key secret.
func (s *Store) Authenticate(secret string) (Principal, bool) {
	secret = strings.TrimSpace(secret)
	if secret == "" {
		return Principal{}, false
	}

	row := s.db.QueryRow(`
		SELECT id, name, budget_usd
		FROM api_keys
		WHERE secret_hash = ? AND disabled_at IS NULL
	`, hashSecret(secret))

	var (
		principal Principal
		budgetUSD sql.NullFloat64
	)
	if err := row.Scan(&principal.ID, &principal.Name, &budgetUSD); err != nil {
		return Principal{}, false
	}
	principal.Managed = true
	if budgetUSD.Valid {
		value := budgetUSD.Float64
		principal.BudgetUSD = &value
	}
	return principal, true
}

// Create inserts a new API key and returns the secret once.
func (s *Store) Create(name string, budgetUSD *float64) (CreateResult, error) {
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
func (s *Store) Disable(id string) (KeySummary, error) {
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
func (s *Store) Get(id string) (KeySummary, bool) {
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
func (s *Store) List() []KeySummary {
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
func (s *Store) UsageByID(id string) (UsageReport, bool) {
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
func (s *Store) UsageReports() ([]UsageReport, UsageTotals) {
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

// Record stores usage under the authenticated managed API key.
func (s *Store) Record(_ context.Context, event usage.Event) {
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

func scanKeySummary(scanner interface {
	Scan(dest ...any) error
}) (KeySummary, error) {
	var (
		key                      KeySummary
		createdAt                string
		budgetUSD                sql.NullFloat64
		disabledAt               sql.NullString
		lastUsedAt               sql.NullString
		requestCount             sql.NullInt64
		pricedRequestCount       sql.NullInt64
		inputTokens              sql.NullInt64
		cachedInputTokens        sql.NullInt64
		cacheReadInputTokens     sql.NullInt64
		cacheCreationInputTokens sql.NullInt64
		outputTokens             sql.NullInt64
		totalTokens              sql.NullInt64
		estimatedCostUSD         sql.NullFloat64
	)
	if err := scanner.Scan(
		&key.ID,
		&key.Name,
		&key.Prefix,
		&budgetUSD,
		&createdAt,
		&disabledAt,
		&requestCount,
		&pricedRequestCount,
		&inputTokens,
		&cachedInputTokens,
		&cacheReadInputTokens,
		&cacheCreationInputTokens,
		&outputTokens,
		&totalTokens,
		&estimatedCostUSD,
		&lastUsedAt,
	); err != nil {
		return KeySummary{}, err
	}

	parsedCreatedAt, err := time.Parse(time.RFC3339Nano, createdAt)
	if err != nil {
		return KeySummary{}, err
	}
	key.CreatedAt = parsedCreatedAt
	if disabledAt.Valid {
		parsedDisabledAt, err := time.Parse(time.RFC3339Nano, disabledAt.String)
		if err != nil {
			return KeySummary{}, err
		}
		key.DisabledAt = &parsedDisabledAt
	}
	key.Usage = UsageTotals{
		RequestCount:             int(requestCount.Int64),
		PricedRequestCount:       int(pricedRequestCount.Int64),
		InputTokens:              int(inputTokens.Int64),
		CachedInputTokens:        int(cachedInputTokens.Int64),
		CacheReadInputTokens:     int(cacheReadInputTokens.Int64),
		CacheCreationInputTokens: int(cacheCreationInputTokens.Int64),
		OutputTokens:             int(outputTokens.Int64),
		TotalTokens:              int(totalTokens.Int64),
		EstimatedCostUSD:         estimatedCostUSD.Float64,
	}
	if budgetUSD.Valid {
		value := budgetUSD.Float64
		key.BudgetUSD = &value
		remaining := value - key.Usage.EstimatedCostUSD
		key.RemainingBudgetUSD = &remaining
	}
	if lastUsedAt.Valid {
		parsedLastUsedAt, err := time.Parse(time.RFC3339Nano, lastUsedAt.String)
		if err != nil {
			return KeySummary{}, err
		}
		key.Usage.LastUsedAt = &parsedLastUsedAt
	}
	return key, nil
}

func scanUsageReport(scanner interface {
	Scan(dest ...any) error
}) (UsageReport, error) {
	var (
		report                   UsageReport
		budgetUSD                sql.NullFloat64
		disabledAt               sql.NullString
		lastUsedAt               sql.NullString
		requestCount             sql.NullInt64
		pricedRequestCount       sql.NullInt64
		inputTokens              sql.NullInt64
		cachedInputTokens        sql.NullInt64
		cacheReadInputTokens     sql.NullInt64
		cacheCreationInputTokens sql.NullInt64
		outputTokens             sql.NullInt64
		totalTokens              sql.NullInt64
		estimatedCostUSD         sql.NullFloat64
	)
	if err := scanner.Scan(
		&report.KeyID,
		&report.KeyName,
		&report.KeyPrefix,
		&budgetUSD,
		&disabledAt,
		&requestCount,
		&pricedRequestCount,
		&inputTokens,
		&cachedInputTokens,
		&cacheReadInputTokens,
		&cacheCreationInputTokens,
		&outputTokens,
		&totalTokens,
		&estimatedCostUSD,
		&lastUsedAt,
	); err != nil {
		return UsageReport{}, err
	}
	if disabledAt.Valid {
		parsedDisabledAt, err := time.Parse(time.RFC3339Nano, disabledAt.String)
		if err != nil {
			return UsageReport{}, err
		}
		report.DisabledAt = &parsedDisabledAt
	}
	report.Usage = UsageTotals{
		RequestCount:             int(requestCount.Int64),
		PricedRequestCount:       int(pricedRequestCount.Int64),
		InputTokens:              int(inputTokens.Int64),
		CachedInputTokens:        int(cachedInputTokens.Int64),
		CacheReadInputTokens:     int(cacheReadInputTokens.Int64),
		CacheCreationInputTokens: int(cacheCreationInputTokens.Int64),
		OutputTokens:             int(outputTokens.Int64),
		TotalTokens:              int(totalTokens.Int64),
		EstimatedCostUSD:         estimatedCostUSD.Float64,
	}
	if budgetUSD.Valid {
		value := budgetUSD.Float64
		report.BudgetUSD = &value
		remaining := value - report.Usage.EstimatedCostUSD
		report.RemainingBudgetUSD = &remaining
	}
	if lastUsedAt.Valid {
		parsedLastUsedAt, err := time.Parse(time.RFC3339Nano, lastUsedAt.String)
		if err != nil {
			return UsageReport{}, err
		}
		report.Usage.LastUsedAt = &parsedLastUsedAt
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

func ensureColumn(db *sql.DB, table, column, definition string) error {
	rows, err := db.Query(`PRAGMA table_info(` + table + `)`)
	if err != nil {
		return fmt.Errorf("inspect sqlite table %s: %w", table, err)
	}
	defer rows.Close()

	for rows.Next() {
		var (
			cid        int
			name       string
			typ        string
			notNull    int
			defaultVal sql.NullString
			primaryKey int
		)
		if err := rows.Scan(&cid, &name, &typ, &notNull, &defaultVal, &primaryKey); err != nil {
			return fmt.Errorf("scan sqlite table info %s: %w", table, err)
		}
		if name == column {
			return nil
		}
	}

	if _, err := db.Exec(`ALTER TABLE ` + table + ` ADD COLUMN ` + column + ` ` + definition); err != nil {
		return fmt.Errorf("add sqlite column %s.%s: %w", table, column, err)
	}
	return nil
}

func boolToInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

func nullableFloat64(v *float64) any {
	if v == nil {
		return nil
	}
	return *v
}
