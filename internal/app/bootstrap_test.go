package app

import (
	"path/filepath"
	"testing"

	"github.com/jiayx/llmio/internal/config"
	"github.com/jiayx/llmio/internal/runtimeconfig"
)

func TestBootstrapBuildsApplication(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "llmio.db")
	store, err := runtimeconfig.Open(dbPath)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	if err := store.Save(config.RuntimeConfig{
		Providers: []config.ProviderConfig{{
			Name:    "openai",
			Type:    "openai-compatible",
			BaseURL: "https://example.com/v1",
		}},
		ModelRoutes: []config.ModelRoute{{
			ExternalModel: "gpt-proxy",
			Targets: []config.Target{{
				Provider:     "openai",
				BackendModel: "gpt-4.1",
			}},
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

func TestApplicationListenAddrDefaults(t *testing.T) {
	if got := (*Application)(nil).ListenAddr(); got != ":18080" {
		t.Fatalf("nil ListenAddr() = %q", got)
	}

	app := &Application{}
	if got := app.ListenAddr(); got != ":18080" {
		t.Fatalf("empty ListenAddr() = %q", got)
	}
}
