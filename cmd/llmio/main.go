package main

import (
	"log/slog"
	"net/http"
	"os"
	"strings"

	"github.com/jiayx/llmio/internal/config"
	"github.com/jiayx/llmio/internal/gateway"
)

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: parseLogLevel(os.Getenv("LLMIO_LOG_LEVEL")),
	}))
	slog.SetDefault(logger)

	configPath := os.Getenv("LLMIO_CONFIG")
	if configPath == "" {
		configPath = "llmio.json"
	}

	cfg, err := config.Load(configPath)
	if err != nil {
		slog.Error("load config", "path", configPath, "err", err)
		os.Exit(1)
	}

	server, err := gateway.NewServer(cfg)
	if err != nil {
		slog.Error("build server", "err", err)
		os.Exit(1)
	}

	addr := cfg.Listen
	if addr == "" {
		addr = ":8080"
	}

	slog.Info("server listening", "addr", addr)
	if err := http.ListenAndServe(addr, server.Handler()); err != nil {
		slog.Error("serve", "addr", addr, "err", err)
		os.Exit(1)
	}
}

func parseLogLevel(v string) slog.Leveler {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "debug":
		return slog.LevelDebug
	case "warn", "warning":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}
