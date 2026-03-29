package openai

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/jiayx/llmio/internal/llm"
)

// ChatCompletionRequestToLLM normalizes one OpenAI chat completions request into the internal LLM model.
func ChatCompletionRequestToLLM(req ChatCompletionRequest) (llm.ChatRequest, error) {
	out := llm.ChatRequest{
		Model:       req.Model,
		Messages:    make([]llm.Message, 0, len(req.Messages)),
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
		User:        req.User,
		Tools:       openAIToolsToLLM(req.Tools),
		ToolChoice:  openAIToolChoiceToLLM(req.ToolChoice),
	}
	if req.MaxTokens != nil {
		out.MaxTokens = *req.MaxTokens
	}
	for i, msg := range req.Messages {
		content, err := openAIContentToLLM(msg.Role, msg.Content)
		if err != nil {
			return llm.ChatRequest{}, fmt.Errorf("invalid messages[%d]: %w", i, err)
		}
		if len(msg.ToolCalls) > 0 {
			content = append(content, openAIToolCallsToLLM(msg.ToolCalls)...)
		}
		if msg.Role == "system" {
			out.System = append(out.System, content...)
			continue
		}
		role := msg.Role
		if role == "tool" {
			role = "user"
			content = []llm.ContentPart{{
				Type:       llm.ContentTypeToolResult,
				ToolCallID: msg.ToolCallID,
				Output:     llm.ContentText(content),
			}}
		}
		out.Messages = append(out.Messages, llm.Message{Role: role, Content: content, Name: msg.Name})
	}
	return out, nil
}

// ResponsesRequestToLLM normalizes one OpenAI responses request into the internal LLM model.
func ResponsesRequestToLLM(req ResponsesRequest) (llm.ChatRequest, error) {
	out := llm.ChatRequest{
		Model:       req.Model,
		Messages:    make([]llm.Message, 0, 4),
		System:      llm.TextParts(strings.TrimSpace(req.Instructions)),
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
		User:        req.User,
		Tools:       openAIToolsToLLM(req.Tools),
		ToolChoice:  openAIToolChoiceToLLM(req.ToolChoice),
	}
	if req.MaxOutputTokens != nil {
		out.MaxTokens = *req.MaxOutputTokens
	}

	inputMessages, err := normalizeResponsesInput(req.Input)
	if err != nil {
		return llm.ChatRequest{}, fmt.Errorf("invalid input: %w", err)
	}
	for _, msg := range inputMessages {
		if msg.Role == "system" {
			out.System = append(out.System, msg.Content...)
			continue
		}
		out.Messages = append(out.Messages, msg)
	}
	return out, nil
}

// ResponsesResponseFromLLM encodes one internal response as an OpenAI responses payload.
func ResponsesResponseFromLLM(externalModel string, resp *llm.ChatResponse) ResponsesResponse {
	parts := resp.EffectiveOutput()
	return ResponsesResponse{
		ID:         resp.ID,
		Object:     "response",
		CreatedAt:  time.Now().Unix(),
		Model:      externalModel,
		Status:     "completed",
		Output:     responseOutputItems(parts),
		OutputText: resp.OutputText,
		Usage: &ResponseUsage{
			InputTokens:  resp.InputTokens,
			OutputTokens: resp.OutputTokens,
			TotalTokens:  resp.InputTokens + resp.OutputTokens,
		},
	}
}

func normalizeResponsesInput(input any) ([]llm.Message, error) {
	switch v := input.(type) {
	case nil:
		return nil, nil
	case string:
		return []llm.Message{{Role: "user", Content: llm.TextParts(v)}}, nil
	case []any:
		messages := make([]llm.Message, 0, len(v))
		for _, item := range v {
			msg, ok, err := normalizeResponsesMessage(item)
			if err != nil {
				return nil, err
			}
			if ok {
				messages = append(messages, msg)
			}
		}
		return messages, nil
	default:
		return nil, fmt.Errorf("unsupported input shape %T", input)
	}
}

func normalizeResponsesMessage(item any) (llm.Message, bool, error) {
	m, ok := item.(map[string]any)
	if !ok {
		return llm.Message{}, false, errors.New("input item must be an object")
	}

	if itemType, _ := m["type"].(string); itemType != "" {
		switch itemType {
		case "function_call":
			return llm.Message{
				Role: "assistant",
				Content: []llm.ContentPart{{
					Type:       llm.ContentTypeToolCall,
					Name:       stringValue(m["name"]),
					ToolCallID: stringValue(m["call_id"]),
					Input:      stringValue(m["arguments"]),
				}},
			}, true, nil
		case "function_call_output":
			return llm.Message{
				Role: "user",
				Content: []llm.ContentPart{{
					Type:       llm.ContentTypeToolResult,
					ToolCallID: stringValue(m["call_id"]),
					Output:     stringValue(m["output"]),
				}},
			}, true, nil
		case "reasoning":
			return llm.Message{}, false, nil
		}
	}

	role, _ := m["role"].(string)
	if role == "" {
		role, _ = m["type"].(string)
	}
	switch role {
	case "message":
		role, _ = m["role"].(string)
	case "user", "assistant", "system", "developer":
	default:
		return llm.Message{}, false, nil
	}

	content, err := normalizeResponsesContent(m["content"])
	if err != nil {
		return llm.Message{}, false, err
	}

	if role == "system" || role == "developer" {
		return llm.Message{Role: "system", Content: content}, true, nil
	}
	return llm.Message{Role: role, Content: content}, true, nil
}

func normalizeResponsesContent(content any) ([]llm.ContentPart, error) {
	switch v := content.(type) {
	case nil:
		return nil, nil
	case string:
		return llm.TextParts(v), nil
	case []any:
		parts := make([]llm.ContentPart, 0, len(v))
		for _, item := range v {
			m, ok := item.(map[string]any)
			if !ok {
				return nil, errors.New("content item must be an object")
			}
			typ, _ := m["type"].(string)
			switch typ {
			case "input_text", "output_text", "text":
				text, _ := m["text"].(string)
				parts = append(parts, llm.ContentPart{Type: llm.ContentTypeText, Text: text})
			case "input_image":
				imageURL, _ := m["image_url"].(string)
				parts = append(parts, llm.ContentPart{Type: llm.ContentTypeImage, URL: imageURL})
			case "function_call":
				name, _ := m["name"].(string)
				callID, _ := m["call_id"].(string)
				args, _ := m["arguments"].(string)
				parts = append(parts, llm.ContentPart{Type: llm.ContentTypeToolCall, Name: name, ToolCallID: callID, Input: args})
			case "function_call_output":
				callID, _ := m["call_id"].(string)
				output, _ := m["output"].(string)
				parts = append(parts, llm.ContentPart{Type: llm.ContentTypeToolResult, ToolCallID: callID, Output: output})
			default:
				return nil, fmt.Errorf("unsupported content type %q", typ)
			}
		}
		return parts, nil
	default:
		return nil, fmt.Errorf("unsupported content shape %T", content)
	}
}

func openAIContentToLLM(role string, content any) ([]llm.ContentPart, error) {
	switch v := content.(type) {
	case nil:
		return nil, nil
	case string:
		return llm.TextParts(v), nil
	case []any:
		parts := make([]llm.ContentPart, 0, len(v))
		for _, item := range v {
			m, ok := item.(map[string]any)
			if !ok {
				return nil, errors.New("content item must be an object")
			}
			typ, _ := m["type"].(string)
			switch typ {
			case "text", "input_text", "output_text":
				text, _ := m["text"].(string)
				parts = append(parts, llm.ContentPart{Type: llm.ContentTypeText, Text: text})
			case "image_url":
				var imageURL string
				switch image := m["image_url"].(type) {
				case string:
					imageURL = image
				case map[string]any:
					imageURL, _ = image["url"].(string)
				}
				if imageURL == "" {
					return nil, errors.New("image_url content item requires a url")
				}
				parts = append(parts, llm.ContentPart{Type: llm.ContentTypeImage, URL: imageURL})
			default:
				if role == "tool" {
					text, _ := m["text"].(string)
					parts = append(parts, llm.ContentPart{Type: llm.ContentTypeToolResult, Output: text})
					continue
				}
				return nil, fmt.Errorf("unsupported content type %q", typ)
			}
		}
		return parts, nil
	default:
		return nil, fmt.Errorf("unsupported content shape %T", content)
	}
}

func openAIToolsToLLM(tools []Tool) []llm.ToolDefinition {
	out := make([]llm.ToolDefinition, 0, len(tools))
	for _, tool := range tools {
		if tool.Type != "" && tool.Type != "function" {
			continue
		}
		out = append(out, llm.ToolDefinition{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			InputSchema: string(tool.Function.Parameters),
		})
	}
	return out
}

func openAIToolChoiceToLLM(choice any) *llm.ToolChoice {
	switch v := choice.(type) {
	case nil:
		return nil
	case string:
		if v == "" || v == "auto" || v == "none" {
			return &llm.ToolChoice{Type: v}
		}
	case map[string]any:
		typ, _ := v["type"].(string)
		if fn, ok := v["function"].(map[string]any); ok {
			name, _ := fn["name"].(string)
			return &llm.ToolChoice{Type: typ, Name: name}
		}
	}
	return nil
}

func openAIToolCallsToLLM(calls []ToolCall) []llm.ContentPart {
	out := make([]llm.ContentPart, 0, len(calls))
	for _, call := range calls {
		out = append(out, llm.ContentPart{
			Type:       llm.ContentTypeToolCall,
			Name:       call.Function.Name,
			ToolCallID: call.ID,
			Input:      call.Function.Arguments,
		})
	}
	return out
}

func stringValue(v any) string {
	s, _ := v.(string)
	return s
}
