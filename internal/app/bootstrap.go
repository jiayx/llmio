package app

import (
	"net/http"

	"github.com/jiayx/llmio/internal/config"
	"github.com/jiayx/llmio/internal/gateway"
)

// Application wires config and gateway into a runnable server surface.
type Application struct {
	Config  *config.Config
	Gateway *gateway.Server
}

// Bootstrap loads config and assembles the gateway service graph.
func Bootstrap(configPath string) (*Application, error) {
	cfg, err := config.Load(configPath)
	if err != nil {
		return nil, err
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

// Handler returns the root HTTP handler for the application.
func (a *Application) Handler() http.Handler {
	return a.Gateway.Handler()
}

// ListenAddr returns the configured listen address with a sensible default.
func (a *Application) ListenAddr() string {
	if a == nil || a.Config == nil || a.Config.Listen == "" {
		return ":8080"
	}
	return a.Config.Listen
}
