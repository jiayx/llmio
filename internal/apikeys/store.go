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
	ID      string
	Name    string
	Managed bool
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
	RequestCount int        `json:"request_count"`
	InputTokens  int        `json:"input_tokens"`
	OutputTokens int        `json:"output_tokens"`
	TotalTokens  int        `json:"total_tokens"`
	LastUsedAt   *time.Time `json:"last_used_at,omitempty"`
}

// KeySummary is the externally visible API key metadata.
type KeySummary struct {
	ID         string      `json:"id"`
	Name       string      `json:"name"`
	Prefix     string      `json:"prefix"`
	CreatedAt  time.Time   `json:"created_at"`
	DisabledAt *time.Time  `json:"disabled_at,omitempty"`
	Usage      UsageTotals `json:"usage"`
}

// UsageReport returns aggregated usage with key identity.
type UsageReport struct {
	KeyID      string      `json:"key_id"`
	KeyName    string      `json:"key_name"`
	KeyPrefix  string      `json:"key_prefix"`
	DisabledAt *time.Time  `json:"disabled_at,omitempty"`
	Usage      UsageTotals `json:"usage"`
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
			created_at TEXT NOT NULL,
			disabled_at TEXT,
			secret_hash TEXT NOT NULL UNIQUE
		)`,
		`CREATE TABLE IF NOT EXISTS api_key_usage (
			api_key_id TEXT PRIMARY KEY,
			request_count INTEGER NOT NULL DEFAULT 0,
			input_tokens INTEGER NOT NULL DEFAULT 0,
			output_tokens INTEGER NOT NULL DEFAULT 0,
			total_tokens INTEGER NOT NULL DEFAULT 0,
			last_used_at TEXT,
			FOREIGN KEY(api_key_id) REFERENCES api_keys(id) ON DELETE CASCADE
		)`,
	}
	for _, stmt := range stmts {
		if _, err := s.db.Exec(stmt); err != nil {
			return fmt.Errorf("init sqlite api key store: %w", err)
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
		SELECT id, name
		FROM api_keys
		WHERE secret_hash = ? AND disabled_at IS NULL
	`, hashSecret(secret))

	var principal Principal
	if err := row.Scan(&principal.ID, &principal.Name); err != nil {
		return Principal{}, false
	}
	principal.Managed = true
	return principal, true
}

// Create inserts a new API key and returns the secret once.
func (s *Store) Create(name string) (CreateResult, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return CreateResult{}, fmt.Errorf("name is required")
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

	s.mu.Lock()
	defer s.mu.Unlock()

	tx, err := s.db.Begin()
	if err != nil {
		return CreateResult{}, fmt.Errorf("begin create api key: %w", err)
	}
	defer tx.Rollback()

	if _, err := tx.Exec(`
		INSERT INTO api_keys (id, name, prefix, created_at, secret_hash)
		VALUES (?, ?, ?, ?, ?)
	`, key.ID, key.Name, key.Prefix, key.CreatedAt.Format(time.RFC3339Nano), hashSecret(secret)); err != nil {
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
			k.created_at,
			k.disabled_at,
			u.request_count,
			u.input_tokens,
			u.output_tokens,
			u.total_tokens,
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
			k.created_at,
			k.disabled_at,
			u.request_count,
			u.input_tokens,
			u.output_tokens,
			u.total_tokens,
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
			k.disabled_at,
			u.request_count,
			u.input_tokens,
			u.output_tokens,
			u.total_tokens,
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
			k.disabled_at,
			u.request_count,
			u.input_tokens,
			u.output_tokens,
			u.total_tokens,
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
		total.InputTokens += report.Usage.InputTokens
		total.OutputTokens += report.Usage.OutputTokens
		total.TotalTokens += report.Usage.TotalTokens
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
			input_tokens = input_tokens + ?,
			output_tokens = output_tokens + ?,
			total_tokens = total_tokens + ?,
			last_used_at = ?
		WHERE api_key_id = ?
	`, event.InputTokens, event.OutputTokens, event.TotalTokens, lastUsedAt, event.APIKeyID)
}

func scanKeySummary(scanner interface {
	Scan(dest ...any) error
}) (KeySummary, error) {
	var (
		key          KeySummary
		createdAt    string
		disabledAt   sql.NullString
		lastUsedAt   sql.NullString
		requestCount sql.NullInt64
		inputTokens  sql.NullInt64
		outputTokens sql.NullInt64
		totalTokens  sql.NullInt64
	)
	if err := scanner.Scan(
		&key.ID,
		&key.Name,
		&key.Prefix,
		&createdAt,
		&disabledAt,
		&requestCount,
		&inputTokens,
		&outputTokens,
		&totalTokens,
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
		RequestCount: int(requestCount.Int64),
		InputTokens:  int(inputTokens.Int64),
		OutputTokens: int(outputTokens.Int64),
		TotalTokens:  int(totalTokens.Int64),
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
		report       UsageReport
		disabledAt   sql.NullString
		lastUsedAt   sql.NullString
		requestCount sql.NullInt64
		inputTokens  sql.NullInt64
		outputTokens sql.NullInt64
		totalTokens  sql.NullInt64
	)
	if err := scanner.Scan(
		&report.KeyID,
		&report.KeyName,
		&report.KeyPrefix,
		&disabledAt,
		&requestCount,
		&inputTokens,
		&outputTokens,
		&totalTokens,
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
		RequestCount: int(requestCount.Int64),
		InputTokens:  int(inputTokens.Int64),
		OutputTokens: int(outputTokens.Int64),
		TotalTokens:  int(totalTokens.Int64),
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
