package httpio

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
	data, err := json.Marshal(v)
	if err != nil {
		slog.Error("marshal json response", "status", status, "err", err)
		data = []byte(`{"error":"internal server error"}`)
		status = http.StatusInternalServerError
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(append(data, '\n'))
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
	lines, err := readSSELines(body)
	if err != nil {
		slog.Warn("read passthrough sse", "err", err)
	}

	block := make([]string, 0, 8)
	flushBlock := func() {
		if len(block) == 0 {
			return
		}
		for _, line := range rewriteSSEBlock(block, rewrite) {
			_, _ = io.WriteString(w, line+"\n")
		}
		_, _ = io.WriteString(w, "\n")
		flusher.Flush()
		block = block[:0]
	}

	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			flushBlock()
			continue
		}
		block = append(block, line)
	}
	if len(block) > 0 {
		flushBlock()
	}
}

func readSSELines(body io.Reader) ([]string, error) {
	data, err := io.ReadAll(body)
	if err != nil {
		return nil, err
	}
	text := strings.ReplaceAll(string(data), "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	return strings.Split(text, "\n"), nil
}

func rewriteSSEBlock(lines []string, rewrite func([]byte) ([]byte, error)) []string {
	if rewrite == nil {
		return lines
	}

	dataLines := make([]string, 0, len(lines))
	for _, line := range lines {
		if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	if len(dataLines) == 0 {
		return lines
	}

	payload := strings.Join(dataLines, "\n")
	if payload == "" || payload == "[DONE]" {
		return lines
	}

	rewritten, err := rewrite([]byte(payload))
	if err != nil {
		return lines
	}

	out := make([]string, 0, len(lines)-len(dataLines)+1)
	emittedData := false
	for _, line := range lines {
		if strings.HasPrefix(line, "data:") {
			if !emittedData {
				for _, rewrittenLine := range strings.Split(string(rewritten), "\n") {
					out = append(out, "data: "+rewrittenLine)
				}
				emittedData = true
			}
			continue
		}
		out = append(out, line)
	}
	return out
}
