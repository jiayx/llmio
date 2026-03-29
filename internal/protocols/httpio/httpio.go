package httpio

import (
	"bufio"
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

// WritePassthroughResponse proxies an upstream response, allowing protocol-specific body rewriting.
func WritePassthroughResponse(w http.ResponseWriter, resp *http.Response, rewrite func([]byte) ([]byte, error)) {
	defer closeResponse("passthrough response", resp.Body)
	copyPassthroughHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)

	if flusher, ok := w.(http.Flusher); ok && strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") {
		rewritePassthroughSSE(w, flusher, resp.Body, rewrite)
		return
	}

	rewritePassthroughJSON(w, resp.Body, rewrite)
}

func closeResponse(scope string, body io.Closer) {
	if body == nil {
		return
	}
	if err := body.Close(); err != nil {
		slog.Warn("close response body", "scope", scope, "err", err)
	}
}

func copyPassthroughHeaders(dst, src http.Header) {
	for k, vals := range src {
		if !allowPassthroughHeader(k) {
			continue
		}
		for _, v := range vals {
			dst.Set(k, v)
		}
	}
}

func allowPassthroughHeader(k string) bool {
	switch strings.ToLower(k) {
	case "content-type", "content-encoding", "cache-control", "retry-after":
		return true
	default:
		return false
	}
}

func rewritePassthroughJSON(w io.Writer, body io.Reader, rewrite func([]byte) ([]byte, error)) {
	data, err := io.ReadAll(body)
	if err != nil {
		return
	}
	if rewrite == nil {
		_, _ = w.Write(data)
		return
	}
	rewritten, err := rewrite(data)
	if err != nil {
		_, _ = w.Write(data)
		return
	}
	_, _ = w.Write(rewritten)
}

func rewritePassthroughSSE(w io.Writer, flusher http.Flusher, body io.Reader, rewrite func([]byte) ([]byte, error)) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "data:") && rewrite != nil {
			payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
			if payload != "" && payload != "[DONE]" {
				if rewritten, err := rewrite([]byte(payload)); err == nil {
					line = "data: " + string(rewritten)
				}
			}
		}
		_, _ = io.WriteString(w, line+"\n")
		flusher.Flush()
	}
}
