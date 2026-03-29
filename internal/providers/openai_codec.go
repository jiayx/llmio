package providers

import (
	"strings"

	"github.com/jiayx/llmio/internal/core"
	openaiproto "github.com/jiayx/llmio/internal/protocols/openai"
)

func coreToOpenAI(req core.ChatRequest) openaiproto.ChatCompletionRequest {
	messages := make([]openaiproto.Message, 0, len(req.Messages)+1)
	if len(req.System) > 0 {
		systemContent, _ := corePartsToOpenAIMessage(req.System)
		messages = append(messages, openaiproto.Message{
			Role:    "system",
			Content: systemContent,
		})
	}
	for _, msg := range req.Messages {
		content, toolCalls := corePartsToOpenAIMessage(msg.Content)
		role := msg.Role
		if role == "user" && hasOnlyToolResults(msg.Content) {
			role = "tool"
			for _, part := range msg.Content {
				if part.Type != core.ContentTypeToolResult {
					continue
				}
				messages = append(messages, openaiproto.Message{
					Role:       role,
					Content:    part.Output,
					ToolCallID: part.ToolCallID,
				})
			}
			continue
		}
		messages = append(messages, openaiproto.Message{
			Role:      role,
			Content:   content,
			Name:      msg.Name,
			ToolCalls: toolCalls,
		})
	}

	var maxTokens *int
	if req.MaxTokens > 0 {
		maxTokens = &req.MaxTokens
	}

	return openaiproto.ChatCompletionRequest{
		Model:       req.Model,
		Messages:    messages,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		MaxTokens:   maxTokens,
		Stream:      req.Stream,
		User:        req.User,
		Tools:       coreToolsToOpenAI(req.Tools),
		ToolChoice:  coreToolChoiceToOpenAI(req.ToolChoice),
	}
}

func openAIToCoreResponse(resp openaiproto.ChatCompletionResponse, raw []byte) *core.ChatResponse {
	out := &core.ChatResponse{
		ID:    resp.ID,
		Model: resp.Model,
		Raw:   raw,
	}
	if resp.Usage != nil {
		out.InputTokens = resp.Usage.PromptTokens
		out.OutputTokens = resp.Usage.CompletionTokens
	}
	if len(resp.Choices) > 0 {
		out.FinishReason = resp.Choices[0].FinishReason
		out.Output = openAIMessageToCore(resp.Choices[0].Message)
		out.OutputText = extractCoreText(out.Output)
	}
	return out
}

func corePartsToOpenAIMessage(parts []core.ContentPart) (any, []openaiproto.ToolCall) {
	var content []map[string]any
	var toolCalls []openaiproto.ToolCall
	for _, part := range parts {
		switch part.Type {
		case core.ContentTypeText:
			content = append(content, map[string]any{"type": "text", "text": part.Text})
		case core.ContentTypeImage:
			if part.URL != "" {
				content = append(content, map[string]any{"type": "image_url", "image_url": map[string]any{"url": part.URL}})
			}
		case core.ContentTypeToolCall:
			toolCalls = append(toolCalls, openaiproto.ToolCall{
				ID:   part.ToolCallID,
				Type: "function",
				Function: openaiproto.FunctionCall{
					Name:      part.Name,
					Arguments: part.Input,
				},
			})
		}
	}
	if len(content) == 0 {
		return "", toolCalls
	}
	if len(content) == 1 && content[0]["type"] == "text" {
		return content[0]["text"], toolCalls
	}
	out := make([]any, 0, len(content))
	for _, item := range content {
		out = append(out, item)
	}
	return out, toolCalls
}

func coreToolsToOpenAI(tools []core.ToolDefinition) []openaiproto.Tool {
	out := make([]openaiproto.Tool, 0, len(tools))
	for _, tool := range tools {
		out = append(out, openaiproto.Tool{
			Type: "function",
			Function: openaiproto.FunctionDefinition{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  []byte(tool.InputSchema),
			},
		})
	}
	return out
}

func coreToolChoiceToOpenAI(choice *core.ToolChoice) any {
	if choice == nil {
		return nil
	}
	if choice.Name == "" {
		return choice.Type
	}
	return map[string]any{
		"type": "function",
		"function": map[string]any{
			"name": choice.Name,
		},
	}
}

func openAIMessageToCore(msg openaiproto.Message) []core.ContentPart {
	parts := make([]core.ContentPart, 0)
	switch v := msg.Content.(type) {
	case string:
		if v != "" {
			parts = append(parts, core.ContentPart{Type: core.ContentTypeText, Text: v})
		}
	case []any:
		for _, item := range v {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}
			switch m["type"] {
			case "text":
				parts = append(parts, core.ContentPart{Type: core.ContentTypeText, Text: stringValue(m["text"])})
			case "image_url":
				if image, ok := m["image_url"].(map[string]any); ok {
					parts = append(parts, core.ContentPart{Type: core.ContentTypeImage, URL: stringValue(image["url"])})
				}
			}
		}
	}
	for _, call := range msg.ToolCalls {
		parts = append(parts, core.ContentPart{
			Type:       core.ContentTypeToolCall,
			Name:       call.Function.Name,
			ToolCallID: call.ID,
			Input:      call.Function.Arguments,
		})
	}
	return parts
}

func extractCoreText(parts []core.ContentPart) string {
	var texts []string
	for _, part := range parts {
		if part.Type == core.ContentTypeText && part.Text != "" {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, "\n")
}

func hasOnlyToolResults(parts []core.ContentPart) bool {
	if len(parts) == 0 {
		return false
	}
	for _, part := range parts {
		if part.Type != core.ContentTypeToolResult {
			return false
		}
	}
	return true
}

func stringValue(v any) string {
	s, _ := v.(string)
	return s
}
