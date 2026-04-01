package app

import (
	"errors"
	"net/http"
	"os"
	"strings"

	"github.com/jiayx/llmio/internal/config"
	"github.com/jiayx/llmio/internal/gateway"
	storagesqlite "github.com/jiayx/llmio/internal/storage/sqlite"
)

// Application wires config and gateway into a runnable server surface.
type Application struct {
	Config  *config.AppConfig
	Gateway *gateway.Server
}

// Bootstrap loads runtime config from SQLite and assembles the gateway service graph.
func Bootstrap(databasePath string) (*Application, error) {
	databasePath = strings.TrimSpace(databasePath)
	if databasePath == "" {
		databasePath = os.Getenv("LLMIO_DATABASE_PATH")
	}
	databasePath = strings.TrimSpace(databasePath)
	if databasePath == "" {
		databasePath = "llmio.db"
	}

	db, err := storagesqlite.Open(databasePath)
	if err != nil {
		return nil, err
	}
	if err := storagesqlite.Init(db); err != nil {
		_ = db.Close()
		return nil, err
	}
	store := storagesqlite.NewRuntimeConfigStore(db)
	runtimeCfg, _, err := store.Load()
	runtimeState := gateway.MissingRuntimeConfigState()
	var invalidErr *storagesqlite.InvalidRuntimeConfigError
	switch {
	case err == nil:
		if prepareErr := config.PrepareRuntimeConfig(&runtimeCfg); prepareErr != nil {
			runtimeState = gateway.InvalidRuntimeConfigState(runtimeCfg, prepareErr)
		} else {
			runtimeState = gateway.ReadyRuntimeConfigState(runtimeCfg)
		}
	case os.IsNotExist(err):
		runtimeState = gateway.MissingRuntimeConfigState()
	case errors.As(err, &invalidErr):
		runtimeState = gateway.InvalidRuntimeConfigState(config.RuntimeConfig{}, err)
	default:
		_ = db.Close()
		return nil, err
	}
	appCfg := &config.AppConfig{
		Listen:       strings.TrimSpace(os.Getenv("LLMIO_LISTEN")),
		AdminAPIKeys: splitCSV(os.Getenv("LLMIO_ADMIN_API_KEYS")),
		DatabasePath: databasePath,
	}
	if appCfg.Listen == "" {
		appCfg.Listen = ":18080"
	}
	server, err := gateway.NewServer(appCfg, runtimeState, db)
	if err != nil {
		_ = db.Close()
		return nil, err
	}

	return &Application{
		Config:  appCfg,
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

// ListenAddr returns the configured listen address.
func (a *Application) ListenAddr() string {
	return a.Config.Listen
}
