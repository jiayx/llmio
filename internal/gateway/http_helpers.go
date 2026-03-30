package gateway

import (
	"net/http"
	"strings"
)

func bearerToken(v string) string {
	if !strings.HasPrefix(strings.ToLower(v), "bearer ") {
		return ""
	}
	return strings.TrimSpace(v[7:])
}

type statusRecorder struct {
	http.ResponseWriter
	status int
	wrote  bool
}

func (w *statusRecorder) WriteHeader(status int) {
	if w.wrote {
		return
	}
	w.wrote = true
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}

func (w *statusRecorder) Write(p []byte) (int, error) {
	if !w.wrote {
		w.WriteHeader(w.status)
	}
	return w.ResponseWriter.Write(p)
}

func (w *statusRecorder) Flush() {
	if flusher, ok := w.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}
