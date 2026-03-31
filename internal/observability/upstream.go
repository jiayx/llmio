package observability

import (
	"io"
	"log/slog"
	"net/http"
	"strings"
)

type LoggingRoundTripper struct {
	Base http.RoundTripper
}

func NewLoggingRoundTripper(base http.RoundTripper) http.RoundTripper {
	if base == nil {
		base = http.DefaultTransport
	}
	if _, ok := base.(*LoggingRoundTripper); ok {
		return base
	}
	return &LoggingRoundTripper{Base: base}
}

func (rt *LoggingRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if rt == nil {
		rt = &LoggingRoundTripper{Base: http.DefaultTransport}
	}
	if rt.Base == nil {
		rt.Base = http.DefaultTransport
	}
	requestID := RequestIDFromContext(req.Context())
	slog.Debug("provider http request",
		"request_id", requestID,
		"method", req.Method,
		"url", req.URL.String(),
		"headers", RedactHeaders(req.Header),
		"body", Bytes(readRequestBody(req)),
	)

	resp, err := rt.Base.RoundTrip(req)
	if err != nil {
		slog.Error("provider http request failed",
			"request_id", requestID,
			"method", req.Method,
			"url", req.URL.String(),
			"err", err,
		)
		return nil, err
	}

	slog.Debug("provider http response",
		"request_id", requestID,
		"method", req.Method,
		"url", req.URL.String(),
		"status", resp.StatusCode,
		"headers", RedactHeaders(resp.Header),
		"content_type", resp.Header.Get("Content-Type"),
	)
	resp.Body = newTraceReadCloser(resp.Body, requestID, req.Method, req.URL.String(), resp.StatusCode, resp.Header.Get("Content-Type"))
	return resp, nil
}

func readRequestBody(req *http.Request) []byte {
	if req == nil || req.Body == nil {
		return nil
	}
	if req.GetBody != nil {
		clone, err := req.GetBody()
		if err == nil {
			defer clone.Close()
			data, readErr := io.ReadAll(clone)
			if readErr == nil {
				return data
			}
		}
	}
	return nil
}

type traceReadCloser struct {
	requestID   string
	body        io.ReadCloser
	method      string
	url         string
	status      int
	contentType string
	buf         []byte
}

func newTraceReadCloser(body io.ReadCloser, requestID, method, target string, status int, contentType string) io.ReadCloser {
	if body == nil {
		return nil
	}
	return &traceReadCloser{
		requestID:   requestID,
		body:        body,
		method:      method,
		url:         target,
		status:      status,
		contentType: contentType,
	}
}

func (t *traceReadCloser) Read(p []byte) (int, error) {
	n, err := t.body.Read(p)
	if n > 0 {
		t.capture(p[:n])
		if isStreamingContentType(t.contentType) {
			slog.Debug("provider http response body chunk trace",
				"request_id", t.requestID,
				"method", t.method,
				"url", t.url,
				"status", t.status,
				"chunk", Bytes(p[:n]),
			)
		}
	}
	if err != nil && err == io.EOF && !isStreamingContentType(t.contentType) {
		slog.Debug("provider http response body trace",
			"request_id", t.requestID,
			"method", t.method,
			"url", t.url,
			"status", t.status,
			"content_type", t.contentType,
			"body", Bytes(t.buf),
		)
	}
	return n, err
}

func (t *traceReadCloser) Close() error {
	return t.body.Close()
}

func (t *traceReadCloser) capture(chunk []byte) {
	limit := Limit()
	if limit <= 0 {
		t.buf = append(t.buf, chunk...)
		return
	}
	if len(t.buf) >= limit {
		return
	}
	remaining := limit - len(t.buf)
	if len(chunk) > remaining {
		chunk = chunk[:remaining]
	}
	t.buf = append(t.buf, chunk...)
}

func isStreamingContentType(contentType string) bool {
	return strings.Contains(strings.ToLower(contentType), "text/event-stream")
}
