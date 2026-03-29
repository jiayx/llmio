package gateway

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/jiayx/llmio/internal/core"
	anthropicproto "github.com/jiayx/llmio/internal/protocols/anthropic"
	openaiproto "github.com/jiayx/llmio/internal/protocols/openai"
)

func openAIRequestToCore(req openaiproto.ChatCompletionRequest) core.ChatRequest {
	out := core.ChatRequest{
		Model:       req.Model,
		Messages:    make([]core.Message, 0, len(req.Messages)),
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
		User:        req.User,
		Tools:       openAIToolsToCore(req.Tools),
		ToolChoice:  openAIToolChoiceToCore(req.ToolChoice),
	}
	if req.MaxTokens != nil {
		out.MaxTokens = *req.MaxTokens
	}
	for _, msg := range req.Messages {
		content, err := openAIContentToCore(msg.Role, msg.Content)
		if err != nil {
			continue
		}
		if len(msg.ToolCalls) > 0 {
			content = append(content, openAIToolCallsToCore(msg.ToolCalls)...)
		}
		if msg.Role == "system" {
			out.System = append(out.System, content...)
			continue
		}
		role := msg.Role
		if role == "tool" {
			role = "user"
			content = []core.ContentPart{{
				Type:       core.ContentTypeToolResult,
				ToolCallID: msg.ToolCallID,
				Output:     contentText(content),
			}}
		}
		out.Messages = append(out.Messages, core.Message{Role: role, Content: content, Name: msg.Name})
	}
	return out
}

func openAIResponsesRequestToCore(req openaiproto.ResponsesRequest) (core.ChatRequest, error) {
	out := core.ChatRequest{
		Model:       req.Model,
		Messages:    make([]core.Message, 0, 4),
		System:      textParts(strings.TrimSpace(req.Instructions)),
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
		User:        req.User,
		Tools:       openAIToolsToCore(req.Tools),
		ToolChoice:  openAIToolChoiceToCore(req.ToolChoice),
	}
	if req.MaxOutputTokens != nil {
		out.MaxTokens = *req.MaxOutputTokens
	}

	inputMessages, err := normalizeResponsesInput(req.Input)
	if err != nil {
		return core.ChatRequest{}, fmt.Errorf("invalid input: %w", err)
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

func anthropicToCore(req anthropicproto.MessagesRequest) (core.ChatRequest, error) {
	out := core.ChatRequest{
		Model:       req.Model,
		Messages:    make([]core.Message, 0, len(req.Messages)),
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
		Tools:       anthropicToolsToCore(req.Tools),
		ToolChoice:  anthropicToolChoiceToCore(req.ToolChoice),
	}
	if req.System != nil {
		systemContent, err := normalizeAnthropicContent(req.System)
		if err != nil {
			return core.ChatRequest{}, fmt.Errorf("invalid system content: %w", err)
		}
		out.System = systemContent
	}
	for _, msg := range req.Messages {
		content, err := normalizeAnthropicContent(msg.Content)
		if err != nil {
			return core.ChatRequest{}, fmt.Errorf("invalid %s content: %w", msg.Role, err)
		}
		out.Messages = append(out.Messages, core.Message{
			Role:    msg.Role,
			Content: content,
		})
	}
	if req.Metadata != nil {
		out.User = req.Metadata["user_id"]
	}
	return out, nil
}

func coreResponseToAnthropic(externalModel string, resp *core.ChatResponse) anthropicproto.MessagesResponse {
	parts := resp.Output
	if len(parts) == 0 && resp.OutputText != "" {
		parts = textParts(resp.OutputText)
	}
	out := anthropicproto.MessagesResponse{
		ID:    resp.ID,
		Type:  "message",
		Role:  "assistant",
		Model: externalModel,
		Usage: anthropicproto.Usage{
			InputTokens:  resp.InputTokens,
			OutputTokens: resp.OutputTokens,
		},
	}
	out.Content = coreContentToAnthropic(parts)
	out.StopReason = mapFinishReason(resp.FinishReason)
	return out
}

func writeCoreResponseAsOpenAI(w http.ResponseWriter, externalModel string, resp *core.ChatResponse) {
	parts := resp.Output
	if len(parts) == 0 && resp.OutputText != "" {
		parts = textParts(resp.OutputText)
	}
	content, toolCalls := coreContentToOpenAI(parts)
	out := openaiproto.ChatCompletionResponse{
		ID:      resp.ID,
		Object:  "chat.completion",
		Model:   externalModel,
		Choices: []openaiproto.Choice{{Index: 0, Message: openaiproto.Message{Role: "assistant", Content: content, ToolCalls: toolCalls}, FinishReason: resp.FinishReason}},
		Usage: &openaiproto.CompletionUsage{
			PromptTokens:     resp.InputTokens,
			CompletionTokens: resp.OutputTokens,
			TotalTokens:      resp.InputTokens + resp.OutputTokens,
		},
	}
	writeJSON(w, http.StatusOK, out)
}

func coreResponseToOpenAIResponse(externalModel string, resp *core.ChatResponse) openaiproto.ResponsesResponse {
	parts := resp.Output
	if len(parts) == 0 && resp.OutputText != "" {
		parts = textParts(resp.OutputText)
	}
	return openaiproto.ResponsesResponse{
		ID:         resp.ID,
		Object:     "response",
		CreatedAt:  time.Now().Unix(),
		Model:      externalModel,
		Status:     "completed",
		Output:     responseOutputItems(parts),
		OutputText: resp.OutputText,
		Usage: &openaiproto.ResponseUsage{
			InputTokens:  resp.InputTokens,
			OutputTokens: resp.OutputTokens,
			TotalTokens:  resp.InputTokens + resp.OutputTokens,
		},
	}
}

func responseOutputItems(parts []core.ContentPart) []openaiproto.ResponseOutputItem {
	if len(parts) == 0 {
		return nil
	}
	var items []openaiproto.ResponseOutputItem
	var messageParts []openaiproto.ResponseContentPart
	for _, part := range parts {
		switch part.Type {
		case core.ContentTypeText:
			messageParts = append(messageParts, openaiproto.ResponseContentPart{Type: "output_text", Text: part.Text})
		case core.ContentTypeImage:
			messageParts = append(messageParts, openaiproto.ResponseContentPart{Type: "input_image", ImageURL: part.URL})
		case core.ContentTypeReasoning:
			items = append(items, openaiproto.ResponseOutputItem{
				Type:   "reasoning",
				Status: "completed",
				Content: []openaiproto.ResponseContentPart{{
					Type: "output_text",
					Text: part.Text,
				}},
			})
		case core.ContentTypeToolCall:
			items = append(items, openaiproto.ResponseOutputItem{
				Type:      "function_call",
				Name:      part.Name,
				CallID:    part.ToolCallID,
				Arguments: part.Input,
				Status:    "completed",
			})
		}
	}
	if len(messageParts) > 0 {
		items = append([]openaiproto.ResponseOutputItem{{
			Type:    "message",
			Role:    "assistant",
			Status:  "completed",
			Content: messageParts,
		}}, items...)
	}
	return items
}

func normalizeAnthropicContent(content any) ([]core.ContentPart, error) {
	switch v := content.(type) {
	case string:
		return textParts(v), nil
	case []any:
		parts := make([]core.ContentPart, 0, len(v))
		for _, item := range v {
			m, ok := item.(map[string]any)
			if !ok {
				return nil, errors.New("content item must be an object")
			}
			part, err := anthropicContentMapToCore(m)
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

func normalizeResponsesInput(input any) ([]core.Message, error) {
	switch v := input.(type) {
	case nil:
		return nil, nil
	case string:
		return []core.Message{{Role: "user", Content: textParts(v)}}, nil
	case []any:
		messages := make([]core.Message, 0, len(v))
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

func normalizeResponsesMessage(item any) (core.Message, bool, error) {
	m, ok := item.(map[string]any)
	if !ok {
		return core.Message{}, false, errors.New("input item must be an object")
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
		return core.Message{}, false, nil
	}

	content, err := normalizeResponsesContent(m["content"])
	if err != nil {
		return core.Message{}, false, err
	}

	if role == "system" || role == "developer" {
		return core.Message{Role: "system", Content: content}, true, nil
	}
	if role == "function_call_output" {
		role = "user"
	}
	return core.Message{Role: role, Content: content}, true, nil
}

func normalizeResponsesContent(content any) ([]core.ContentPart, error) {
	switch v := content.(type) {
	case nil:
		return nil, nil
	case string:
		return textParts(v), nil
	case []any:
		parts := make([]core.ContentPart, 0, len(v))
		for _, item := range v {
			m, ok := item.(map[string]any)
			if !ok {
				return nil, errors.New("content item must be an object")
			}
			typ, _ := m["type"].(string)
			switch typ {
			case "input_text", "output_text", "text":
				text, _ := m["text"].(string)
				parts = append(parts, core.ContentPart{Type: core.ContentTypeText, Text: text})
			case "input_image":
				imageURL, _ := m["image_url"].(string)
				parts = append(parts, core.ContentPart{Type: core.ContentTypeImage, URL: imageURL})
			case "function_call":
				name, _ := m["name"].(string)
				callID, _ := m["call_id"].(string)
				args, _ := m["arguments"].(string)
				parts = append(parts, core.ContentPart{Type: core.ContentTypeToolCall, Name: name, ToolCallID: callID, Input: args})
			case "function_call_output":
				callID, _ := m["call_id"].(string)
				output, _ := m["output"].(string)
				parts = append(parts, core.ContentPart{Type: core.ContentTypeToolResult, ToolCallID: callID, Output: output})
			default:
				return nil, fmt.Errorf("unsupported content type %q", typ)
			}
		}
		return parts, nil
	default:
		return nil, fmt.Errorf("unsupported content shape %T", content)
	}
}

func openAIContentToCore(role string, content any) ([]core.ContentPart, error) {
	switch v := content.(type) {
	case nil:
		return nil, nil
	case string:
		return textParts(v), nil
	case []any:
		parts := make([]core.ContentPart, 0, len(v))
		for _, item := range v {
			m, ok := item.(map[string]any)
			if !ok {
				return nil, errors.New("content item must be an object")
			}
			typ, _ := m["type"].(string)
			switch typ {
			case "text", "input_text", "output_text":
				text, _ := m["text"].(string)
				parts = append(parts, core.ContentPart{Type: core.ContentTypeText, Text: text})
			case "image_url":
				var imageURL string
				switch image := m["image_url"].(type) {
				case string:
					imageURL = image
				case map[string]any:
					imageURL, _ = image["url"].(string)
				}
				parts = append(parts, core.ContentPart{Type: core.ContentTypeImage, URL: imageURL})
			default:
				if role == "tool" {
					text, _ := m["text"].(string)
					parts = append(parts, core.ContentPart{Type: core.ContentTypeToolResult, Output: text})
					continue
				}
			}
		}
		return parts, nil
	default:
		return nil, fmt.Errorf("unsupported content shape %T", content)
	}
}

func openAIToolsToCore(tools []openaiproto.Tool) []core.ToolDefinition {
	out := make([]core.ToolDefinition, 0, len(tools))
	for _, tool := range tools {
		if tool.Type != "" && tool.Type != "function" {
			continue
		}
		out = append(out, core.ToolDefinition{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			InputSchema: string(tool.Function.Parameters),
		})
	}
	return out
}

func openAIToolChoiceToCore(choice any) *core.ToolChoice {
	switch v := choice.(type) {
	case nil:
		return nil
	case string:
		if v == "" || v == "auto" || v == "none" {
			return &core.ToolChoice{Type: v}
		}
	case map[string]any:
		typ, _ := v["type"].(string)
		if fn, ok := v["function"].(map[string]any); ok {
			name, _ := fn["name"].(string)
			return &core.ToolChoice{Type: typ, Name: name}
		}
	}
	return nil
}

func openAIToolCallsToCore(calls []openaiproto.ToolCall) []core.ContentPart {
	out := make([]core.ContentPart, 0, len(calls))
	for _, call := range calls {
		out = append(out, core.ContentPart{
			Type:       core.ContentTypeToolCall,
			Name:       call.Function.Name,
			ToolCallID: call.ID,
			Input:      call.Function.Arguments,
		})
	}
	return out
}

func anthropicToolsToCore(tools []anthropicproto.ToolDefinition) []core.ToolDefinition {
	out := make([]core.ToolDefinition, 0, len(tools))
	for _, tool := range tools {
		schema, _ := json.Marshal(tool.InputSchema)
		out = append(out, core.ToolDefinition{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: string(schema),
		})
	}
	return out
}

func anthropicToolChoiceToCore(choice any) *core.ToolChoice {
	switch v := choice.(type) {
	case nil:
		return nil
	case string:
		return &core.ToolChoice{Type: v}
	case map[string]any:
		typ, _ := v["type"].(string)
		name, _ := v["name"].(string)
		return &core.ToolChoice{Type: typ, Name: name}
	default:
		return nil
	}
}

func anthropicContentMapToCore(m map[string]any) ([]core.ContentPart, error) {
	typ, _ := m["type"].(string)
	switch typ {
	case "text":
		text, _ := m["text"].(string)
		return textParts(text), nil
	case "image":
		if source, ok := m["source"].(map[string]any); ok {
			return []core.ContentPart{{
				Type:      core.ContentTypeImage,
				MediaType: stringValue(source["media_type"]),
				Data:      stringValue(source["data"]),
			}}, nil
		}
		return nil, errors.New("image block missing source")
	case "tool_use":
		input, _ := json.Marshal(m["input"])
		return []core.ContentPart{{
			Type:       core.ContentTypeToolCall,
			Name:       stringValue(m["name"]),
			ToolCallID: stringValue(m["id"]),
			Input:      string(input),
		}}, nil
	case "tool_result":
		output, err := normalizeToolResultOutput(m["content"])
		if err != nil {
			return nil, err
		}
		return []core.ContentPart{{
			Type:       core.ContentTypeToolResult,
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

func coreContentToAnthropic(parts []core.ContentPart) []anthropicproto.ContentBlock {
	out := make([]anthropicproto.ContentBlock, 0, len(parts))
	for _, part := range parts {
		switch part.Type {
		case core.ContentTypeText:
			out = append(out, anthropicproto.ContentBlock{Type: "text", Text: part.Text})
		case core.ContentTypeReasoning:
			out = append(out, anthropicproto.ContentBlock{Type: "text", Text: part.Text})
		case core.ContentTypeImage:
			if part.URL != "" {
				continue
			}
			out = append(out, anthropicproto.ContentBlock{
				Type: "image",
				Source: &anthropicproto.ImageSource{
					Type:      "base64",
					MediaType: part.MediaType,
					Data:      part.Data,
				},
			})
		case core.ContentTypeToolCall:
			var input any
			if part.Input != "" {
				_ = json.Unmarshal([]byte(part.Input), &input)
			}
			out = append(out, anthropicproto.ContentBlock{
				Type:  "tool_use",
				ID:    part.ToolCallID,
				Name:  part.Name,
				Input: input,
			})
		case core.ContentTypeToolResult:
			out = append(out, anthropicproto.ContentBlock{
				Type:      "tool_result",
				ToolUseID: part.ToolCallID,
				Content: []anthropicproto.ContentBlock{{
					Type: "text",
					Text: part.Output,
				}},
				IsError: part.IsError,
			})
		}
	}
	return out
}

func coreContentToOpenAI(parts []core.ContentPart) (any, []openaiproto.ToolCall) {
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

func textParts(s string) []core.ContentPart {
	if strings.TrimSpace(s) == "" {
		return nil
	}
	return []core.ContentPart{{Type: core.ContentTypeText, Text: s}}
}

func contentText(parts []core.ContentPart) string {
	var texts []string
	for _, part := range parts {
		switch part.Type {
		case core.ContentTypeText:
			texts = append(texts, part.Text)
		case core.ContentTypeToolResult:
			texts = append(texts, part.Output)
		}
	}
	return strings.Join(texts, "\n")
}

func stringValue(v any) string {
	s, _ := v.(string)
	return s
}

func boolValue(v any) bool {
	b, _ := v.(bool)
	return b
}

func anthropicContentBlockFromPart(part core.ContentPart) anthropicproto.ContentBlock {
	blocks := coreContentToAnthropic([]core.ContentPart{part})
	if len(blocks) == 0 {
		return anthropicproto.ContentBlock{Type: "text", Text: ""}
	}
	return blocks[0]
}

func mapFinishReason(reason string) string {
	switch reason {
	case "", "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	default:
		return reason
	}
}
