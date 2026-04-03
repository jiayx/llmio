package app

import (
	"errors"
	"net/http"
	"os"

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
func Bootstrap(appCfg *config.AppConfig) (*Application, error) {
	if appCfg == nil {
		return nil, errors.New("app config is required")
	}
	if appCfg.DatabasePath == "" {
		return nil, errors.New("database path is required")
	}

	db, err := storagesqlite.Open(appCfg.DatabasePath)
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

// Handler returns the root HTTP handler for the application.
func (a *Application) Handler() http.Handler {
	return a.Gateway.Handler()
}

// ListenAddr returns the configured listen address.
func (a *Application) ListenAddr() string {
	return a.Config.Listen
}
