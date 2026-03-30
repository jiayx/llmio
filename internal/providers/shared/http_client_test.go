package providershared

import (
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestTraceReadCloserPreservesBody(t *testing.T) {
	rc := newTraceReadCloser(io.NopCloser(strings.NewReader(`{"ok":true}`)), http.MethodPost, "https://example.com/v1/chat/completions", http.StatusOK, "application/json")

	data, err := io.ReadAll(rc)
	if err != nil {
		t.Fatalf("ReadAll() error = %v", err)
	}
	if string(data) != `{"ok":true}` {
		t.Fatalf("body = %s", string(data))
	}
}

func TestRedactHeaders(t *testing.T) {
	headers := http.Header{
		"Authorization": []string{"Bearer secret"},
		"X-API-Key":     []string{"secret"},
		"Content-Type":  []string{"application/json"},
	}

	got := redactHeaders(headers)
	if got["Authorization"][0] != "[REDACTED]" || got["X-API-Key"][0] != "[REDACTED]" {
		t.Fatalf("headers = %#v", got)
	}
	if got["Content-Type"][0] != "application/json" {
		t.Fatalf("headers = %#v", got)
	}
}
