package gateway

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"

	anthropicproto "github.com/jiayx/llmio/internal/protocols/anthropic"
	openaiproto "github.com/jiayx/llmio/internal/protocols/openai"
	"github.com/jiayx/llmio/internal/providers"
)

func retryableError(err error) bool {
	msg := err.Error()
	for _, code := range []string{" status 429", " status 500", " status 502", " status 503", " status 504"} {
		if strings.Contains(msg, code) {
			return true
		}
	}
	return false
}

func retryableStatus(status int) bool {
	switch status {
	case http.StatusTooManyRequests, http.StatusInternalServerError, http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout:
		return true
	default:
		return false
	}
}

func writeOpenAIError(w http.ResponseWriter, status int, message string) {
	logProtocolError("openai", status, "invalid_request_error", message)
	writeJSON(w, status, openaiproto.ChatCompletionResponse{
		Error: &openaiproto.CompletionError{
			Message: message,
			Type:    "invalid_request_error",
		},
	})
}

func writeAnthropicError(w http.ResponseWriter, status int, typ, message string) {
	logProtocolError("anthropic", status, typ, message)
	writeJSON(w, status, map[string]any{
		"type": "error",
		"error": anthropicproto.Error{
			Type:    typ,
			Message: message,
		},
	})
}

func logProtocolError(protocol string, status int, errorType, message string) {
	slog.Error("request failed",
		"protocol", protocol,
		"status", status,
		"error_type", errorType,
		"message", message,
	)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writePassthroughResponse(w http.ResponseWriter, resp *http.Response, protocol, externalModel string) {
	defer closeResponseBody("passthrough response", resp.Body)
	copyPassthroughHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)

	if flusher, ok := w.(http.Flusher); ok && strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") {
		rewritePassthroughSSE(w, flusher, resp.Body, protocol, externalModel)
		return
	}

	rewritePassthroughJSON(w, resp.Body, protocol, externalModel)
}

func closeResponseBody(scope string, body io.Closer) {
	if body == nil {
		return
	}
	if err := body.Close(); err != nil {
		slog.Warn("close response body", "scope", scope, "err", err)
	}
}

func writeSSE(w io.Writer, event, data string) {
	if event != "" {
		_, _ = fmt.Fprintf(w, "event: %s\n", event)
	}
	_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
}

func writeSSEJSON(w io.Writer, event string, v any) {
	data, _ := json.Marshal(v)
	writeSSE(w, event, string(data))
}

func bearerToken(v string) string {
	if !strings.HasPrefix(strings.ToLower(v), "bearer ") {
		return ""
	}
	return strings.TrimSpace(v[7:])
}

func closeStream(stream *providers.StreamReader) {
	if stream != nil && stream.Close != nil {
		_ = stream.Close()
	}
}

func rewriteRequestModel(body []byte, backendModel string) ([]byte, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, fmt.Errorf("invalid json: %w", err)
	}
	payload["model"] = backendModel
	out, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	return out, nil
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

func rewritePassthroughJSON(w io.Writer, body io.Reader, protocol, externalModel string) {
	data, err := io.ReadAll(body)
	if err != nil {
		return
	}
	rewritten, err := rewritePassthroughJSONPayload(data, protocol, externalModel)
	if err != nil {
		_, _ = w.Write(data)
		return
	}
	_, _ = w.Write(rewritten)
}

func rewritePassthroughSSE(w io.Writer, flusher http.Flusher, body io.Reader, protocol, externalModel string) {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "data:") {
			payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
			if payload != "" && payload != "[DONE]" {
				if rewritten, err := rewritePassthroughJSONPayload([]byte(payload), protocol, externalModel); err == nil {
					line = "data: " + string(rewritten)
				}
			}
		}
		_, _ = io.WriteString(w, line+"\n")
		flusher.Flush()
	}
}

func rewritePassthroughJSONPayload(data []byte, protocol, externalModel string) ([]byte, error) {
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil, err
	}
	switch protocol {
	case "openai":
		rewriteMapModel(payload, externalModel)
	case "anthropic":
		rewriteMapModel(payload, externalModel)
		if nested, ok := payload["message"].(map[string]any); ok {
			rewriteMapModel(nested, externalModel)
		}
	default:
		rewriteMapModel(payload, externalModel)
	}
	return json.Marshal(payload)
}

func rewriteMapModel(m map[string]any, model string) {
	if _, ok := m["model"]; ok {
		m["model"] = model
	}
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (w *statusRecorder) WriteHeader(status int) {
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}

func (w *statusRecorder) Flush() {
	if flusher, ok := w.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}
