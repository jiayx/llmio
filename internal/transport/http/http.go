package transporthttp

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
)

// WriteJSON writes a JSON response with the given status.
func WriteJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// WriteSSE writes one SSE frame.
func WriteSSE(w io.Writer, event, data string) {
	if event != "" {
		_, _ = fmt.Fprintf(w, "event: %s\n", event)
	}
	_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
}

// WriteSSEJSON marshals one SSE frame payload as JSON.
func WriteSSEJSON(w io.Writer, event string, v any) {
	data, _ := json.Marshal(v)
	WriteSSE(w, event, string(data))
}

// Close closes a response body or stream hook and logs failures.
func Close(scope string, body io.Closer) {
	if body == nil {
		return
	}
	if err := body.Close(); err != nil {
		slog.Warn("close response body", "scope", scope, "err", err)
	}
}

// BearerToken extracts the bearer token from an authorization header.
func BearerToken(v string) string {
	if !strings.HasPrefix(strings.ToLower(v), "bearer ") {
		return ""
	}
	return strings.TrimSpace(v[7:])
}

// StatusRecorder captures the final HTTP status written through the response writer.
type StatusRecorder struct {
	http.ResponseWriter
	Status int
}

// WriteHeader records the status code before forwarding it downstream.
func (w *StatusRecorder) WriteHeader(status int) {
	w.Status = status
	w.ResponseWriter.WriteHeader(status)
}

// Flush forwards flushes when the wrapped writer supports them.
func (w *StatusRecorder) Flush() {
	if flusher, ok := w.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}
