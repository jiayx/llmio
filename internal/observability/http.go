package observability

import (
	"bytes"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

func Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestID := EnsureRequestID(r.Header)
		w.Header().Set(RequestIDHeader, requestID)
		r = r.WithContext(WithRequestID(r.Context(), requestID))
		r = r.WithContext(WithRequestPath(r.Context(), r.URL.Path))

		start := time.Now()
		rec := &responseRecorder{ResponseWriter: w, status: http.StatusOK}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			slog.Error("request read failed",
				"request_id", requestID,
				"method", r.Method,
				"path", r.URL.Path,
				"query", r.URL.RawQuery,
				"remote", r.RemoteAddr,
				"err", err,
			)
			http.Error(rec, "invalid request body", http.StatusBadRequest)
			return
		}
		r.Body = io.NopCloser(bytes.NewReader(body))

		slog.Info("request received",
			"request_id", requestID,
			"method", r.Method,
			"path", r.URL.Path,
			"query", r.URL.RawQuery,
			"remote", r.RemoteAddr,
			"headers", RedactHeaders(r.Header),
			"body", Bytes(body),
		)

		defer func() {
			slog.Info("request completed",
				"request_id", requestID,
				"method", r.Method,
				"path", r.URL.Path,
				"status", rec.status,
				"duration", time.Since(start).Truncate(time.Millisecond),
				"remote", r.RemoteAddr,
				"response_headers", RedactHeaders(rec.Header()),
				"response_body", Bytes(rec.body),
			)
		}()

		next.ServeHTTP(rec, r)
	})
}

type responseRecorder struct {
	http.ResponseWriter
	status int
	wrote  bool
	body   []byte
}

func (w *responseRecorder) WriteHeader(status int) {
	if w.wrote {
		return
	}
	w.wrote = true
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}

func (w *responseRecorder) Write(p []byte) (int, error) {
	if !w.wrote {
		w.WriteHeader(w.status)
	}
	w.capture(p)
	return w.ResponseWriter.Write(p)
}

func (w *responseRecorder) Flush() {
	if flusher, ok := w.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (w *responseRecorder) capture(chunk []byte) {
	limit := Limit()
	if limit <= 0 {
		w.body = append(w.body, chunk...)
		return
	}
	if len(w.body) >= limit {
		return
	}
	remaining := limit - len(w.body)
	if len(chunk) > remaining {
		chunk = chunk[:remaining]
	}
	w.body = append(w.body, chunk...)
}

func RedactHeaders(headers http.Header) map[string][]string {
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
