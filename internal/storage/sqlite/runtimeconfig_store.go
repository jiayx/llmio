package sqlite

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/jiayx/llmio/internal/config"
)

// RuntimeConfigStore persists runtime config in SQLite.
type RuntimeConfigStore struct {
	db *sql.DB
}

// InvalidRuntimeConfigError reports that persisted runtime config exists but cannot be decoded.
type InvalidRuntimeConfigError struct {
	Err error
}

func (e *InvalidRuntimeConfigError) Error() string {
	return e.Err.Error()
}

func (e *InvalidRuntimeConfigError) Unwrap() error {
	return e.Err
}

// NewRuntimeConfigStore constructs a runtime config store on top of the shared SQLite database.
func NewRuntimeConfigStore(db *sql.DB) *RuntimeConfigStore {
	return &RuntimeConfigStore{db: db}
}

func (s *RuntimeConfigStore) Load() (config.RuntimeConfig, time.Time, error) {
	row := s.db.QueryRow(`SELECT payload_json, updated_at FROM runtime_config WHERE id = 1`)
	var (
		payload   string
		updatedAt string
	)
	if err := row.Scan(&payload, &updatedAt); err != nil {
		if err == sql.ErrNoRows {
			return config.RuntimeConfig{}, time.Time{}, os.ErrNotExist
		}
		return config.RuntimeConfig{}, time.Time{}, fmt.Errorf("load runtime config: %w", err)
	}
	var out config.RuntimeConfig
	if err := json.Unmarshal([]byte(payload), &out); err != nil {
		return config.RuntimeConfig{}, time.Time{}, &InvalidRuntimeConfigError{Err: fmt.Errorf("decode runtime config: %w", err)}
	}
	t, err := time.Parse(time.RFC3339Nano, updatedAt)
	if err != nil {
		return config.RuntimeConfig{}, time.Time{}, &InvalidRuntimeConfigError{Err: fmt.Errorf("parse runtime config updated_at: %w", err)}
	}
	return out, t, nil
}

func (s *RuntimeConfigStore) Save(doc config.RuntimeConfig) error {
	payload, err := json.Marshal(doc)
	if err != nil {
		return fmt.Errorf("encode runtime config: %w", err)
	}
	now := time.Now().UTC().Format(time.RFC3339Nano)
	_, err = s.db.Exec(`
		INSERT INTO runtime_config (id, payload_json, updated_at)
		VALUES (1, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
			payload_json = excluded.payload_json,
			updated_at = excluded.updated_at
	`, string(payload), now)
	if err != nil {
		return fmt.Errorf("save runtime config: %w", err)
	}
	return nil
}
