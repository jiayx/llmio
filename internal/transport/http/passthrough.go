package transporthttp

import (
	"bufio"
	"io"
	"net/http"
	"strings"
)

// WritePassthroughResponse proxies an upstream response, allowing protocol-specific body rewriting.
func WritePassthroughResponse(w http.ResponseWriter, resp *http.Response, rewrite func([]byte) ([]byte, error)) {
	defer Close("passthrough response", resp.Body)
	copyPassthroughHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)

	if flusher, ok := w.(http.Flusher); ok && strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") {
		rewritePassthroughSSE(w, flusher, resp.Body, rewrite)
		return
	}

	rewritePassthroughJSON(w, resp.Body, rewrite)
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
