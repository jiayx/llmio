package debugtrace

import (
	"os"
	"strconv"
	"strings"
	"sync"
)

var (
	loadOnce sync.Once
	enabled  bool
	limit    int
)

func Enabled() bool {
	load()
	return enabled
}

func Bytes(data []byte) string {
	return String(string(data))
}

func String(s string) string {
	load()
	if limit <= 0 || len(s) <= limit {
		return s
	}
	return s[:limit] + "...(truncated)"
}

func load() {
	loadOnce.Do(func() {
		switch strings.ToLower(strings.TrimSpace(os.Getenv("LLMIO_TRACE_UPSTREAM"))) {
		case "1", "true", "yes", "on":
			enabled = true
		}
		limit = 4000
		if raw := strings.TrimSpace(os.Getenv("LLMIO_TRACE_LIMIT")); raw != "" {
			if parsed, err := strconv.Atoi(raw); err == nil && parsed > 0 {
				limit = parsed
			}
		}
	})
}
