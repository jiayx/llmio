package main

import (
	"context"
	"net"
	"net/http"
	"testing"
	"time"
)

func TestServeServerGracefulShutdown(t *testing.T) {
	t.Helper()

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen() error = %v", err)
	}

	server := &http.Server{
		Addr: listener.Addr().String(),
		Handler: http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusNoContent)
		}),
	}

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	if err := serveServer(ctx, server, time.Second, func() error {
		return server.Serve(listener)
	}); err != nil {
		t.Fatalf("serveServer() error = %v", err)
	}
}
