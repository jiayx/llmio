package providershared

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestHTTPClientDoPreservesResponseBody(t *testing.T) {
	client := NewHTTPClient("http://example.com", nil, &http.Client{
		Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body:       io.NopCloser(strings.NewReader(`{"ok":true}`)),
			}, nil
		}),
	}, nil)

	resp, err := client.Do(context.Background(), http.MethodPost, "/v1/chat/completions", []byte(`{"hello":"world"}`), nil)
	if err != nil {
		t.Fatalf("Do() error = %v", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("ReadAll() error = %v", err)
	}
	if string(data) != `{"ok":true}` {
		t.Fatalf("body = %s", string(data))
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}
