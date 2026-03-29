package llm

import "strings"

// TextParts converts a plain text value into normalized content parts.
func TextParts(s string) []ContentPart {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	return []ContentPart{{Type: ContentTypeText, Text: s}}
}

// ContentText joins textual content from text and tool-result parts.
func ContentText(parts []ContentPart) string {
	var texts []string
	for _, part := range parts {
		switch part.Type {
		case ContentTypeText:
			texts = append(texts, part.Text)
		case ContentTypeToolResult:
			texts = append(texts, part.Output)
		}
	}
	return strings.Join(texts, "\n")
}

// ExtractText joins only text parts from normalized content.
func ExtractText(parts []ContentPart) string {
	var texts []string
	for _, part := range parts {
		if part.Type == ContentTypeText && part.Text != "" {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, "\n")
}

// HasOnlyContentType reports whether every part has the same non-empty type.
func HasOnlyContentType(parts []ContentPart, typ string) bool {
	if len(parts) == 0 {
		return false
	}
	for _, part := range parts {
		if part.Type != typ {
			return false
		}
	}
	return true
}

// EffectiveOutput returns response output, falling back to OutputText when needed.
func (r *ChatResponse) EffectiveOutput() []ContentPart {
	if r == nil {
		return nil
	}
	if len(r.Output) > 0 || r.OutputText == "" {
		return r.Output
	}
	return TextParts(r.OutputText)
}
