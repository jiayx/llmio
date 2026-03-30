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
	if debugtrace.Enabled() {
		slog.Debug("provider http request trace",
			"method", method,
			"url", target,
			"body", debugtrace.Bytes(body),
		)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("provider request: %w", err)
	}
	slog.Debug("provider http response",
		"method", method,
		"url", target,
		"status", resp.StatusCode,
	)
	if debugtrace.Enabled() {
		slog.Debug("provider http response trace",
			"method", method,
			"url", target,
			"status", resp.StatusCode,
			"content_type", resp.Header.Get("Content-Type"),
		)
	}
	return resp, nil
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
