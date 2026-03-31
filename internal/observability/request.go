package observability

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"net/http"
	"strings"
)

type contextKey string

const (
	requestIDKey    contextKey = "request_id"
	requestPathKey  contextKey = "request_path"
	RequestIDHeader            = "X-Request-Id"
)

func WithRequestID(ctx context.Context, requestID string) context.Context {
	if strings.TrimSpace(requestID) == "" {
		return ctx
	}
	return context.WithValue(ctx, requestIDKey, strings.TrimSpace(requestID))
}

func RequestIDFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	requestID, _ := ctx.Value(requestIDKey).(string)
	return requestID
}

func WithRequestPath(ctx context.Context, path string) context.Context {
	if strings.TrimSpace(path) == "" {
		return ctx
	}
	return context.WithValue(ctx, requestPathKey, strings.TrimSpace(path))
}

func RequestPathFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	path, _ := ctx.Value(requestPathKey).(string)
	return path
}

func RequestIDFromHeaders(headers http.Header) string {
	if headers == nil {
		return ""
	}
	return strings.TrimSpace(headers.Get(RequestIDHeader))
}

func EnsureRequestID(headers http.Header) string {
	requestID := RequestIDFromHeaders(headers)
	if requestID != "" {
		return requestID
	}
	requestID = NewRequestID()
	if headers != nil {
		headers.Set(RequestIDHeader, requestID)
	}
	return requestID
}

func NewRequestID() string {
	var buf [12]byte
	if _, err := rand.Read(buf[:]); err != nil {
		return "req_fallback"
	}
	return "req_" + hex.EncodeToString(buf[:])
}
