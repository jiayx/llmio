package providers

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
)

type requestMutator func(*http.Request)

type providerHTTPClient struct {
	baseURL   string
	headers   map[string]string
	client    *http.Client
	customize requestMutator
}

func newProviderHTTPClient(baseURL string, headers map[string]string, client *http.Client, customize requestMutator) *providerHTTPClient {
	if client == nil {
		client = &http.Client{}
	}
	return &providerHTTPClient{
		baseURL:   baseURL,
		headers:   headers,
		client:    client,
		customize: customize,
	}
}

func (c *providerHTTPClient) Do(ctx context.Context, method, path string, body []byte, inboundHeaders http.Header) (*http.Response, error) {
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

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("provider request: %w", err)
	}
	slog.Debug("provider http response",
		"method", method,
		"url", target,
		"status", resp.StatusCode,
	)
	return resp, nil
}

func copyForwardHeaders(dst, src http.Header) {
	for k, vals := range src {
		if strings.EqualFold(k, "Authorization") || strings.EqualFold(k, "x-api-key") {
			continue
		}
		for _, v := range vals {
			dst.Add(k, v)
		}
	}
}

func closeResponseBody(scope string, body io.Closer) {
	if body == nil {
		return
	}
	if err := body.Close(); err != nil {
		slog.Warn("close response body", "scope", scope, "err", err)
	}
}
