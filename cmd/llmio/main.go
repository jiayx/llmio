package main

import (
	"log/slog"
	"net/http"
	"os"
	"strings"

	"github.com/jiayx/llmio/internal/app"
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

	application, err := app.Bootstrap(configPath)
	if err != nil {
		slog.Error("bootstrap app", "path", configPath, "err", err)
		os.Exit(1)
	}

	addr := application.ListenAddr()
	slog.Info("server listening", "addr", addr)
	if err := http.ListenAndServe(addr, application.Handler()); err != nil {
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
