package anthropic

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/jiayx/llmio/internal/llm"
)

// MessagesRequestToLLM normalizes one Anthropic messages request into the internal LLM model.
func MessagesRequestToLLM(req MessagesRequest) (llm.ChatRequest, error) {
	out := llm.ChatRequest{
		Model:       req.Model,
		Messages:    make([]llm.Message, 0, len(req.Messages)),
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
		Tools:       anthropicToolsToLLM(req.Tools),
		ToolChoice:  anthropicToolChoiceToLLM(req.ToolChoice),
	}
	if req.System != nil {
		systemContent, err := normalizeAnthropicContent(req.System)
		if err != nil {
			return llm.ChatRequest{}, fmt.Errorf("invalid system content: %w", err)
		}
		out.System = systemContent
	}
	for _, msg := range req.Messages {
		content, err := normalizeAnthropicContent(msg.Content)
		if err != nil {
			return llm.ChatRequest{}, fmt.Errorf("invalid %s content: %w", msg.Role, err)
		}
		out.Messages = append(out.Messages, llm.Message{
			Role:    msg.Role,
			Content: content,
		})
	}
	if req.Metadata != nil {
		out.User = req.Metadata["user_id"]
	}
	return out, nil
}

// MessagesResponseFromLLM encodes one internal response as an Anthropic messages payload.
func MessagesResponseFromLLM(externalModel string, resp *llm.ChatResponse) MessagesResponse {
	parts := resp.EffectiveOutput()
	out := MessagesResponse{
		ID:    resp.ID,
		Type:  "message",
		Role:  "assistant",
		Model: externalModel,
		Usage: Usage{
			InputTokens:  resp.InputTokens,
			OutputTokens: resp.OutputTokens,
		},
	}
	out.Content = llmContentToAnthropic(parts)
	out.StopReason = mapFinishReason(resp.FinishReason)
	return out
}

func normalizeAnthropicContent(content any) ([]llm.ContentPart, error) {
	switch v := content.(type) {
	case string:
		return llm.TextParts(v), nil
	case []any:
		parts := make([]llm.ContentPart, 0, len(v))
		for _, item := range v {
			m, ok := item.(map[string]any)
			if !ok {
				return nil, errors.New("content item must be an object")
			}
			part, err := anthropicContentMapToLLM(m)
			if err != nil {
				return nil, err
			}
			parts = append(parts, part...)
		}
		return parts, nil
	default:
		return nil, fmt.Errorf("unsupported content shape %T", content)
	}
}

func anthropicToolsToLLM(tools []ToolDefinition) []llm.ToolDefinition {
	out := make([]llm.ToolDefinition, 0, len(tools))
	for _, tool := range tools {
		schema, _ := json.Marshal(tool.InputSchema)
		out = append(out, llm.ToolDefinition{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: string(schema),
		})
	}
	return out
}

func anthropicToolChoiceToLLM(choice any) *llm.ToolChoice {
	switch v := choice.(type) {
	case nil:
		return nil
	case string:
		return &llm.ToolChoice{Type: v}
	case map[string]any:
		typ, _ := v["type"].(string)
		name, _ := v["name"].(string)
		return &llm.ToolChoice{Type: typ, Name: name}
	default:
		return nil
	}
}

func anthropicContentMapToLLM(m map[string]any) ([]llm.ContentPart, error) {
	typ, _ := m["type"].(string)
	switch typ {
	case "text":
		text, _ := m["text"].(string)
		return llm.TextParts(text), nil
	case "image":
		if source, ok := m["source"].(map[string]any); ok {
			return []llm.ContentPart{{
				Type:      llm.ContentTypeImage,
				MediaType: stringValue(source["media_type"]),
				Data:      stringValue(source["data"]),
			}}, nil
		}
		return nil, errors.New("image block missing source")
	case "tool_use":
		input, _ := json.Marshal(m["input"])
		return []llm.ContentPart{{
			Type:       llm.ContentTypeToolCall,
			Name:       stringValue(m["name"]),
			ToolCallID: stringValue(m["id"]),
			Input:      string(input),
		}}, nil
	case "tool_result":
		output, err := normalizeToolResultOutput(m["content"])
		if err != nil {
			return nil, err
		}
		return []llm.ContentPart{{
			Type:       llm.ContentTypeToolResult,
			ToolCallID: stringValue(m["tool_use_id"]),
			Output:     output,
			IsError:    boolValue(m["is_error"]),
		}}, nil
	default:
		return nil, fmt.Errorf("unsupported content type %q", typ)
	}
}

func normalizeToolResultOutput(content any) (string, error) {
	switch v := content.(type) {
	case nil:
		return "", nil
	case string:
		return v, nil
	case []any:
		var parts []string
		for _, item := range v {
			m, ok := item.(map[string]any)
			if !ok {
				return "", errors.New("tool_result content item must be an object")
			}
			if typ, _ := m["type"].(string); typ != "text" {
				return "", fmt.Errorf("unsupported tool_result content type %q", typ)
			}
			parts = append(parts, stringValue(m["text"]))
		}
		return strings.Join(parts, "\n"), nil
	default:
		return "", fmt.Errorf("unsupported tool_result content shape %T", content)
	}
}

func stringValue(v any) string {
	s, _ := v.(string)
	return s
}

func boolValue(v any) bool {
	b, _ := v.(bool)
	return b
}
