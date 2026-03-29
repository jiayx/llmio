package protocols

import (
	"net/http"
	"testing"
)

func TestLookupEndpointMatchesCanonicalOpenAIPath(t *testing.T) {
	adapter, endpoint, ok := LookupEndpoint(DefaultAdapters(), "/v1/responses")
	if !ok {
		t.Fatalf("lookup failed")
	}
	if adapter.Protocol() != ProtocolOpenAI {
		t.Fatalf("protocol = %q", adapter.Protocol())
	}
	if endpoint.InboundPath != "/v1/responses" {
		t.Fatalf("endpoint path = %q", endpoint.InboundPath)
	}
	if got := endpoint.upstreamPath(); got != "/responses" {
		t.Fatalf("forward path = %q", got)
	}
}

func TestLookupEndpointRejectsRootOpenAIAlias(t *testing.T) {
	if _, _, ok := LookupEndpoint(DefaultAdapters(), "/responses"); ok {
		t.Fatalf("lookup unexpectedly matched root alias")
	}
}

func TestOpenAIRequestUsesCanonicalPassthroughPath(t *testing.T) {
	adapter := OpenAIAdapter{}
	endpoint, ok := adapter.Match("/v1/chat/completions")
	if !ok {
		t.Fatalf("match failed")
	}

	meta := adapter.Request(endpoint, "gpt-proxy", []byte(`{"model":"gpt-proxy"}`), http.Header{"X-Test": []string{"1"}}, true)
	if meta.UpstreamPath != "/chat/completions" {
		t.Fatalf("meta path = %q", meta.UpstreamPath)
	}
	if meta.APIType != APIChatCompletions {
		t.Fatalf("api type = %q", meta.APIType)
	}
}
