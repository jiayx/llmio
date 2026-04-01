package app

import (
	"path/filepath"
	"testing"

	"github.com/jiayx/llmio/internal/config"
	storagesqlite "github.com/jiayx/llmio/internal/storage/sqlite"
)

func TestBootstrapBuildsApplication(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "llmio.db")
	db, err := storagesqlite.Open(dbPath)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	if err := storagesqlite.Init(db); err != nil {
		t.Fatalf("Init() error = %v", err)
	}
	store := storagesqlite.NewRuntimeConfigStore(db)
	if err := store.Save(config.RuntimeConfig{
		Providers: []config.ProviderConfig{{
			Name:    "openai",
			Type:    "openai-compatible",
			BaseURL: "https://example.com/v1",
		}},
		Targets: []config.TargetConfig{{
			Name:         "openai-gpt-4-1",
			Provider:     "openai",
			BackendModel: "gpt-4.1",
		}},
		Models: []config.ModelConfig{{
			Name:    "gpt-proxy",
			Targets: []string{"openai-gpt-4-1"},
		}},
	}); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	t.Setenv("LLMIO_LISTEN", ":9090")
	t.Setenv("LLMIO_ADMIN_API_KEYS", "secret")

	app, err := Bootstrap(dbPath)
	if err != nil {
		t.Fatalf("Bootstrap() error = %v", err)
	}
	if app.Config == nil || app.Gateway == nil {
		t.Fatalf("application = %#v", app)
	}
	if got := app.ListenAddr(); got != ":9090" {
		t.Fatalf("ListenAddr() = %q", got)
	}
	if app.Handler() == nil {
		t.Fatalf("Handler() returned nil")
	}
}

func TestBootstrapDefaultsListenAddr(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "llmio.db")
	db, err := storagesqlite.Open(dbPath)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	if err := storagesqlite.Init(db); err != nil {
		t.Fatalf("Init() error = %v", err)
	}
	store := storagesqlite.NewRuntimeConfigStore(db)
	if err := store.Save(config.RuntimeConfig{
		Providers: []config.ProviderConfig{{
			Name:    "openai",
			Type:    "openai-compatible",
			BaseURL: "https://example.com/v1",
		}},
		Targets: []config.TargetConfig{{
			Name:         "openai-gpt-4-1",
			Provider:     "openai",
			BackendModel: "gpt-4.1",
		}},
		Models: []config.ModelConfig{{
			Name:    "gpt-proxy",
			Targets: []string{"openai-gpt-4-1"},
		}},
	}); err != nil {
		t.Fatalf("Save() error = %v", err)
	}

	app, err := Bootstrap(dbPath)
	if err != nil {
		t.Fatalf("Bootstrap() error = %v", err)
	}
	if got := app.ListenAddr(); got != ":18080" {
		t.Fatalf("ListenAddr() = %q", got)
	}
}

func TestBootstrapStartsWithoutRuntimeConfig(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "llmio.db")
	db, err := storagesqlite.Open(dbPath)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	if err := storagesqlite.Init(db); err != nil {
		t.Fatalf("Init() error = %v", err)
	}

	app, err := Bootstrap(dbPath)
	if err != nil {
		t.Fatalf("Bootstrap() error = %v", err)
	}
	if app == nil || app.Gateway == nil {
		t.Fatalf("application = %#v", app)
	}
}

func TestBootstrapStartsWithInvalidRuntimeConfig(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "llmio.db")
	db, err := storagesqlite.Open(dbPath)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	if err := storagesqlite.Init(db); err != nil {
		t.Fatalf("Init() error = %v", err)
	}
	if _, err := db.Exec(`
		INSERT INTO runtime_config (id, payload_json, updated_at)
		VALUES (1, ?, ?)
	`, `{"providers":[{"name":"p","type":"openai-compatible","base_url":"https://example.com/v1"}]}`, "2026-04-01T00:00:00Z"); err != nil {
		t.Fatalf("insert runtime_config: %v", err)
	}

	app, err := Bootstrap(dbPath)
	if err != nil {
		t.Fatalf("Bootstrap() error = %v", err)
	}
	if app == nil || app.Gateway == nil {
		t.Fatalf("application = %#v", app)
	}
}

func TestBootstrapStartsWithUnreadableRuntimeConfig(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "llmio.db")
	db, err := storagesqlite.Open(dbPath)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	if err := storagesqlite.Init(db); err != nil {
		t.Fatalf("Init() error = %v", err)
	}
	if _, err := db.Exec(`
		INSERT INTO runtime_config (id, payload_json, updated_at)
		VALUES (1, ?, ?)
	`, `{"providers":`, "2026-04-01T00:00:00Z"); err != nil {
		t.Fatalf("insert runtime_config: %v", err)
	}

	app, err := Bootstrap(dbPath)
	if err != nil {
		t.Fatalf("Bootstrap() error = %v", err)
	}
	if app == nil || app.Gateway == nil {
		t.Fatalf("application = %#v", app)
	}
}
