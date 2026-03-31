package runtimeconfig

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/jiayx/llmio/internal/config"
	_ "modernc.org/sqlite"
)

type Store struct {
	db *sql.DB
}

func Open(path string) (*Store, error) {
	dsn := strings.TrimSpace(path)
	if dsn == "" {
		dsn = "file:llmio-runtime-config?mode=memory&cache=shared"
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
		return nil, fmt.Errorf("open sqlite runtime config store %s: %w", path, err)
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
	_, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS runtime_config (
			id INTEGER PRIMARY KEY CHECK (id = 1),
			payload_json TEXT NOT NULL,
			updated_at TEXT NOT NULL
		)
	`)
	if err != nil {
		return fmt.Errorf("init sqlite runtime config store: %w", err)
	}
	return nil
}

func (s *Store) Load() (config.RuntimeConfig, time.Time, error) {
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
		return config.RuntimeConfig{}, time.Time{}, fmt.Errorf("decode runtime config: %w", err)
	}
	t, err := time.Parse(time.RFC3339Nano, updatedAt)
	if err != nil {
		return config.RuntimeConfig{}, time.Time{}, fmt.Errorf("parse runtime config updated_at: %w", err)
	}
	return out, t, nil
}

func (s *Store) Save(doc config.RuntimeConfig) error {
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
