package sqlite

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	_ "modernc.org/sqlite"
)

// Open opens the shared SQLite database handle used by the application.
func Open(path string) (*sql.DB, error) {
	dsn := strings.TrimSpace(path)
	if dsn == "" {
		dsn = "file:llmio?mode=memory&cache=shared"
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
		return nil, fmt.Errorf("open sqlite db %s: %w", path, err)
	}
	db.SetMaxOpenConns(1)
	return db, nil
}

// Init initializes and migrates the shared SQLite schema.
func Init(db *sql.DB) error {
	if db == nil {
		return fmt.Errorf("sqlite db is required")
	}

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
		`CREATE TABLE IF NOT EXISTS runtime_config (
			id INTEGER PRIMARY KEY CHECK (id = 1),
			payload_json TEXT NOT NULL,
			updated_at TEXT NOT NULL
		)`,
	}
	for _, stmt := range stmts {
		if _, err := db.Exec(stmt); err != nil {
			return fmt.Errorf("init sqlite schema: %w", err)
		}
	}
	return nil
}
