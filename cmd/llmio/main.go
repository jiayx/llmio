package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/jiayx/llmio/internal/app"
	"github.com/jiayx/llmio/internal/config"
)

func main() {
	if err := config.LoadEnv(); err != nil {
		_, _ = os.Stderr.WriteString("load .env: " + err.Error() + "\n")
		os.Exit(1)
	}

	configPath := os.Getenv("LLMIO_CONFIG")
	if configPath == "" {
		configPath = "llmio.json"
	}

	logger := slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{
		Level: parseLogLevel(os.Getenv("LLMIO_LOG_LEVEL")),
	}))
	slog.SetDefault(logger)

	application, err := app.Bootstrap(configPath)
	if err != nil {
		slog.Error("bootstrap app", "path", configPath, "err", err)
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	addr := application.ListenAddr()
	server := &http.Server{
		Addr:    addr,
		Handler: application.Handler(),
	}

	slog.Info("server listening", "addr", addr)
	if err := serve(ctx, server, 10*time.Second); err != nil {
		slog.Error("serve", "addr", addr, "err", err)
		os.Exit(1)
	}
}

func serve(ctx context.Context, server *http.Server, shutdownTimeout time.Duration) error {
	return serveServer(ctx, server, shutdownTimeout, server.ListenAndServe)
}

func serveServer(ctx context.Context, server *http.Server, shutdownTimeout time.Duration, serveFn func() error) error {
	done := make(chan struct{})
	shutdownErr := make(chan error, 1)

	go func() {
		select {
		case <-ctx.Done():
			slog.Info("shutdown signal received", "addr", server.Addr)
			shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
			defer cancel()
			shutdownErr <- server.Shutdown(shutdownCtx)
		case <-done:
		}
	}()

	err := serveFn()
	close(done)
	if err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	if ctx.Err() == nil {
		return nil
	}

	err = <-shutdownErr
	if err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	slog.Info("server stopped", "addr", server.Addr)
	return nil
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
