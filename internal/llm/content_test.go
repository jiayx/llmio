package llm

import "testing"

func TestTextParts(t *testing.T) {
	if got := TextParts("   "); got != nil {
		t.Fatalf("TextParts(blank) = %#v", got)
	}
	got := TextParts("hello")
	if len(got) != 1 || got[0].Type != ContentTypeText || got[0].Text != "hello" {
		t.Fatalf("TextParts(hello) = %#v", got)
	}
}

func TestContentTextAndExtractText(t *testing.T) {
	parts := []ContentPart{
		{Type: ContentTypeText, Text: "hello"},
		{Type: ContentTypeToolResult, Output: "done"},
		{Type: ContentTypeReasoning, Text: "hidden"},
	}
	if got := ContentText(parts); got != "hello\ndone" {
		t.Fatalf("ContentText() = %q", got)
	}
	if got := ExtractText(parts); got != "hello" {
		t.Fatalf("ExtractText() = %q", got)
	}
}

func TestHasOnlyContentType(t *testing.T) {
	if HasOnlyContentType(nil, ContentTypeToolResult) {
		t.Fatalf("HasOnlyContentType(nil) should be false")
	}
	if !HasOnlyContentType([]ContentPart{{Type: ContentTypeToolResult}}, ContentTypeToolResult) {
		t.Fatalf("HasOnlyContentType(single tool result) should be true")
	}
	if HasOnlyContentType([]ContentPart{{Type: ContentTypeToolResult}, {Type: ContentTypeText}}, ContentTypeToolResult) {
		t.Fatalf("HasOnlyContentType(mixed) should be false")
	}
}

func TestChatResponseEffectiveOutput(t *testing.T) {
	resp := &ChatResponse{OutputText: "fallback"}
	parts := resp.EffectiveOutput()
	if len(parts) != 1 || parts[0].Text != "fallback" {
		t.Fatalf("EffectiveOutput() = %#v", parts)
	}

	resp = &ChatResponse{Output: []ContentPart{{Type: ContentTypeText, Text: "real"}}, OutputText: "fallback"}
	parts = resp.EffectiveOutput()
	if len(parts) != 1 || parts[0].Text != "real" {
		t.Fatalf("EffectiveOutput() preferred fallback unexpectedly = %#v", parts)
	}
}
