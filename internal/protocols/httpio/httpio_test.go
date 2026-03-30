package httpio

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRewritePassthroughSSERewritesMultiLineDataBlock(t *testing.T) {
	rec := httptest.NewRecorder()
	rewritePassthroughSSE(
		rec,
		rec,
		strings.NewReader("event: message\ndata: {\"model\":\"backend\"\ndata: }\n\ndata: [DONE]\n\n"),
		func(data []byte) ([]byte, error) {
			return []byte(strings.ReplaceAll(string(data), "backend", "proxy")), nil
		},
	)

	body := rec.Body.String()
	if !strings.Contains(body, "event: message\n") {
		t.Fatalf("body = %s", body)
	}
	if !strings.Contains(body, "data: {\"model\":\"proxy\"\n") || !strings.Contains(body, "data: }\n\n") {
		t.Fatalf("body = %s", body)
	}
	if !strings.Contains(body, "data: [DONE]\n\n") {
		t.Fatalf("body = %s", body)
	}
}

func TestRewritePassthroughSSEHandlesLargePayload(t *testing.T) {
	rec := httptest.NewRecorder()
	large := strings.Repeat("a", 2*1024*1024)
	rewritePassthroughSSE(
		rec,
		rec,
		strings.NewReader("data: "+large+"\n\n"),
		func(data []byte) ([]byte, error) {
			if len(data) != len(large) {
				t.Fatalf("payload len = %d", len(data))
			}
			return []byte("ok"), nil
		},
	)

	if got := rec.Body.String(); got != "data: ok\n\n" {
		t.Fatalf("body len=%d body prefix=%q", len(got), firstN(got, 32))
	}
}

func TestReadSSELinesNormalizesCRLF(t *testing.T) {
	lines, err := readSSELines(strings.NewReader("data: one\r\ndata: two\r\n\r\n"))
	if err != nil {
		t.Fatalf("readSSELines() error = %v", err)
	}
	if len(lines) < 3 || lines[0] != "data: one" || lines[1] != "data: two" {
		t.Fatalf("lines = %#v", lines)
	}
}

func firstN(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
