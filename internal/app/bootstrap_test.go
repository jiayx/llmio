package app

import (
	"os"
	"path/filepath"
	"testing"
)

func TestBootstrapBuildsApplication(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "llmio.json")
	if err := os.WriteFile(configPath, []byte(`{
		"listen": ":9090",
		"admin_api_keys": ["secret"],
		"database_path": "./llmio.db",
		"providers": [
			{
				"name": "openai",
				"type": "openai-compatible",
				"base_url": "https://example.com/v1"
			}
		],
		"model_routes": [
			{
				"external_model": "gpt-proxy",
				"targets": [
					{"provider": "openai", "backend_model": "gpt-4.1"}
				]
			}
		]
	}`), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	app, err := Bootstrap(configPath)
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

func TestApplicationListenAddrDefaults(t *testing.T) {
	if got := (*Application)(nil).ListenAddr(); got != ":18080" {
		t.Fatalf("nil ListenAddr() = %q", got)
	}

	app := &Application{}
	if got := app.ListenAddr(); got != ":18080" {
		t.Fatalf("empty ListenAddr() = %q", got)
	}
}
