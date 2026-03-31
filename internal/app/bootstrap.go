package app

import (
	"net/http"
	"os"
	"strings"

	"github.com/jiayx/llmio/internal/config"
	"github.com/jiayx/llmio/internal/gateway"
	"github.com/jiayx/llmio/internal/runtimeconfig"
)

// Application wires config and gateway into a runnable server surface.
type Application struct {
	Config  *config.Config
	Gateway *gateway.Server
}

// Bootstrap loads runtime config from SQLite and assembles the gateway service graph.
func Bootstrap(databasePath string) (*Application, error) {
	if strings.TrimSpace(databasePath) == "" {
		databasePath = os.Getenv("LLMIO_DATABASE_PATH")
	}
	if strings.TrimSpace(databasePath) == "" {
		databasePath = "llmio.db"
	}

	store, err := runtimeconfig.Open(databasePath)
	if err != nil {
		return nil, err
	}
	runtimeCfg, _, err := store.Load()
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	cfg := &config.Config{
		Listen:       strings.TrimSpace(os.Getenv("LLMIO_LISTEN")),
		AdminAPIKeys: splitCSV(os.Getenv("LLMIO_ADMIN_API_KEYS")),
		DatabasePath: databasePath,
		Providers:    runtimeCfg.Providers,
		ModelRoutes:  runtimeCfg.ModelRoutes,
		Pricing:      runtimeCfg.Pricing,
	}

	server, err := gateway.NewServer(cfg)
	if err != nil {
		return nil, err
	}

	return &Application{
		Config:  cfg,
		Gateway: server,
	}, nil
}

func splitCSV(raw string) []string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		out = append(out, part)
	}
	return out
}

// Handler returns the root HTTP handler for the application.
func (a *Application) Handler() http.Handler {
	return a.Gateway.Handler()
}

// ListenAddr returns the configured listen address with a sensible default.
func (a *Application) ListenAddr() string {
	if a == nil || a.Config == nil || a.Config.Listen == "" {
		return ":18080"
	}
	return a.Config.Listen
}
