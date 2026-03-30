package providershared

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strings"

	"github.com/jiayx/llmio/internal/debugtrace"
)

type RequestMutator func(*http.Request)

type HTTPClient struct {
	baseURL   string
	headers   map[string]string
	client    *http.Client
	customize RequestMutator
}

func NewHTTPClient(baseURL string, headers map[string]string, client *http.Client, customize RequestMutator) *HTTPClient {
	if client == nil {
		client = &http.Client{}
	}
	return &HTTPClient{
		baseURL:   baseURL,
		headers:   headers,
		client:    client,
		customize: customize,
	}
}

func (c *HTTPClient) Do(ctx context.Context, method, path string, body []byte, inboundHeaders http.Header) (*http.Response, error) {
	target, err := url.JoinPath(c.baseURL, strings.TrimPrefix(path, "/"))
	if err != nil {
		return nil, fmt.Errorf("join path: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, method, target, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("new request: %w", err)
	}

	copyForwardHeaders(req.Header, inboundHeaders)
	req.Header.Del("Host")
	req.Header.Del("Content-Length")
	if c.customize != nil {
		c.customize(req)
	}
	for k, v := range c.headers {
		req.Header.Set(k, v)
	}
	if req.Header.Get("Content-Type") == "" && len(body) > 0 {
		req.Header.Set("Content-Type", "application/json")
	}
	slog.Debug("provider http request",
		"method", method,
		"url", target,
		"body_bytes", len(body),
	)
	slog.Debug("provider http request trace",
		"method", method,
		"url", target,
		"headers", redactHeaders(req.Header),
		"body", debugtrace.Bytes(body),
	)

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("provider request: %w", err)
	}
	slog.Debug("provider http response",
		"method", method,
		"url", target,
		"status", resp.StatusCode,
	)
	slog.Debug("provider http response trace",
		"method", method,
		"url", target,
		"status", resp.StatusCode,
		"headers", redactHeaders(resp.Header),
		"content_type", resp.Header.Get("Content-Type"),
	)
	resp.Body = newTraceReadCloser(resp.Body, method, target, resp.StatusCode, resp.Header.Get("Content-Type"))
	return resp, nil
}

type traceReadCloser struct {
	body        io.ReadCloser
	method      string
	url         string
	status      int
	contentType string
	buf         []byte
}

func newTraceReadCloser(body io.ReadCloser, method, target string, status int, contentType string) io.ReadCloser {
	if body == nil {
		return nil
	}
	return &traceReadCloser{
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
				"method", t.method,
				"url", t.url,
				"status", t.status,
				"chunk", debugtrace.Bytes(p[:n]),
			)
		}
	}
	if err != nil {
		if err == io.EOF && !isStreamingContentType(t.contentType) {
			slog.Debug("provider http response body trace",
				"method", t.method,
				"url", t.url,
				"status", t.status,
				"body", debugtrace.Bytes(t.buf),
			)
		}
	}
	return n, err
}

func (t *traceReadCloser) Close() error {
	return t.body.Close()
}

func (t *traceReadCloser) capture(chunk []byte) {
	limit := debugtrace.Limit()
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

func redactHeaders(headers http.Header) map[string][]string {
	if len(headers) == 0 {
		return nil
	}
	out := make(map[string][]string, len(headers))
	for k, vals := range headers {
		if strings.EqualFold(k, "Authorization") || strings.EqualFold(k, "x-api-key") {
			out[k] = []string{"[REDACTED]"}
			continue
		}
		copied := make([]string, len(vals))
		copy(copied, vals)
		out[k] = copied
	}
	return out
}

func isStreamingContentType(contentType string) bool {
	return strings.Contains(strings.ToLower(contentType), "text/event-stream")
}

func copyForwardHeaders(dst, src http.Header) {
	for k, vals := range src {
		if shouldSkipForwardHeader(k) {
			continue
		}
		for _, v := range vals {
			dst.Add(k, v)
		}
	}
}

func shouldSkipForwardHeader(name string) bool {
	switch {
	case strings.EqualFold(name, "Authorization"),
		strings.EqualFold(name, "x-api-key"),
		strings.EqualFold(name, "Connection"),
		strings.EqualFold(name, "Proxy-Connection"),
		strings.EqualFold(name, "Keep-Alive"),
		strings.EqualFold(name, "Proxy-Authenticate"),
		strings.EqualFold(name, "Proxy-Authorization"),
		strings.EqualFold(name, "Te"),
		strings.EqualFold(name, "Trailer"),
		strings.EqualFold(name, "Transfer-Encoding"),
		strings.EqualFold(name, "Upgrade"):
		return true
	default:
		return false
	}
}

func CloseResponseBody(scope string, body io.Closer) {
	if body == nil {
		return
	}
	if err := body.Close(); err != nil {
		slog.Warn("close response body", "scope", scope, "err", err)
	}
}
