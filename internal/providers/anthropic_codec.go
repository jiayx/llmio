package providers

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/jiayx/llmio/internal/core"
	anthropicproto "github.com/jiayx/llmio/internal/protocols/anthropic"
)

func coreToAnthropic(req core.ChatRequest) anthropicproto.MessagesRequest {
	messages := make([]anthropicproto.Message, 0, len(req.Messages))
	for _, msg := range req.Messages {
		messages = append(messages, anthropicproto.Message{
			Role:    msg.Role,
			Content: corePartsToAnthropic(msg.Content),
		})
	}

	var system any
	if len(req.System) > 0 {
		system = corePartsToAnthropic(req.System)
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 1024
	}

	out := anthropicproto.MessagesRequest{
		Model:       req.Model,
		System:      system,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
		Tools:       coreToolsToAnthropic(req.Tools),
		ToolChoice:  coreToolChoiceToAnthropic(req.ToolChoice),
	}
	if req.User != "" {
		out.Metadata = map[string]string{"user_id": req.User}
	}
	return out
}

func anthropicToCoreResponse(resp anthropicproto.MessagesResponse, raw []byte) *core.ChatResponse {
	out := &core.ChatResponse{
		ID:           resp.ID,
		Model:        resp.Model,
		FinishReason: resp.StopReason,
		InputTokens:  resp.Usage.InputTokens,
		OutputTokens: resp.Usage.OutputTokens,
		Raw:          raw,
	}
	out.Output = anthropicBlocksToCore(resp.Content)
	out.OutputText = extractAnthropicText(out.Output)
	return out
}

func anthropicSSEToCoreEvents(eventName, payload string, blockState map[int]anthropicproto.ContentBlock, out chan<- core.StreamEvent) error {
	switch eventName {
	case "message_start":
		var event anthropicproto.StreamMessageStart
		if err := json.Unmarshal([]byte(payload), &event); err != nil {
			return fmt.Errorf("decode anthropic message_start: %w", err)
		}
		out <- core.StreamEvent{
			Type:        core.StreamEventUsage,
			InputTokens: event.Message.Usage.InputTokens,
			Raw:         json.RawMessage(payload),
		}
	case "content_block_start":
		var event anthropicproto.StreamContentBlockStart
		if err := json.Unmarshal([]byte(payload), &event); err != nil {
			return fmt.Errorf("decode anthropic content_block_start: %w", err)
		}
		blockState[event.Index] = event.ContentBlock
		out <- core.StreamEvent{
			Type:       core.StreamEventContentStart,
			BlockIndex: event.Index,
			Part:       anthropicBlockToStreamPart(event.ContentBlock),
			Raw:        json.RawMessage(payload),
		}
		if event.ContentBlock.Type == "tool_use" {
			input, _ := json.Marshal(event.ContentBlock.Input)
			out <- core.StreamEvent{
				Type:       core.StreamEventTool,
				BlockIndex: event.Index,
				Part:       core.ContentPart{Type: core.ContentTypeToolCall},
				ToolCallID: event.ContentBlock.ID,
				ToolName:   event.ContentBlock.Name,
				ToolInput:  string(input),
				Raw:        json.RawMessage(payload),
			}
		}
	case "content_block_delta":
		var event anthropicproto.StreamContentBlockDelta
		if err := json.Unmarshal([]byte(payload), &event); err != nil {
			return fmt.Errorf("decode anthropic content_block_delta: %w", err)
		}
		if delta, ok := event.Delta.(map[string]any); ok {
			switch stringValue(delta["type"]) {
			case "text_delta":
				if text := stringValue(delta["text"]); text != "" {
					out <- core.StreamEvent{
						Type:       core.StreamEventDelta,
						BlockIndex: event.Index,
						Part:       core.ContentPart{Type: core.ContentTypeText},
						TextDelta:  text,
						Raw:        json.RawMessage(payload),
					}
				}
			case "input_json_delta":
				if partial := stringValue(delta["partial_json"]); partial != "" {
					state := blockState[event.Index]
					out <- core.StreamEvent{
						Type:       core.StreamEventTool,
						ToolCallID: state.ID,
						ToolName:   state.Name,
						BlockIndex: event.Index,
						Part:       core.ContentPart{Type: core.ContentTypeToolCall},
						ToolInput:  partial,
						Raw:        json.RawMessage(payload),
					}
				}
			}
		}
	case "content_block_stop":
		var event struct {
			Index int `json:"index"`
		}
		if err := json.Unmarshal([]byte(payload), &event); err == nil {
			part := anthropicBlockToStreamPart(blockState[event.Index])
			delete(blockState, event.Index)
			out <- core.StreamEvent{
				Type:       core.StreamEventContentStop,
				BlockIndex: event.Index,
				Part:       part,
				Raw:        json.RawMessage(payload),
			}
		}
	case "message_delta":
		var event anthropicproto.StreamMessageDelta
		if err := json.Unmarshal([]byte(payload), &event); err != nil {
			return fmt.Errorf("decode anthropic message_delta: %w", err)
		}
		if event.Usage.OutputTokens > 0 {
			out <- core.StreamEvent{
				Type:         core.StreamEventUsage,
				OutputTokens: event.Usage.OutputTokens,
				Raw:          json.RawMessage(payload),
			}
		}
		if event.Delta.StopReason != "" {
			out <- core.StreamEvent{
				Type:         core.StreamEventStop,
				FinishReason: event.Delta.StopReason,
				Raw:          json.RawMessage(payload),
			}
		}
	case "message_stop":
		out <- core.StreamEvent{Type: core.StreamEventStop, Raw: json.RawMessage(payload)}
	case "error":
		var wrapper struct {
			Error anthropicproto.Error `json:"error"`
		}
		if err := json.Unmarshal([]byte(payload), &wrapper); err != nil {
			return fmt.Errorf("decode anthropic error: %w", err)
		}
		return fmt.Errorf("anthropic stream error: %s", wrapper.Error.Message)
	}
	return nil
}

func extractAnthropicText(content []core.ContentPart) string {
	parts := make([]string, 0, len(content))
	for _, block := range content {
		if block.Type != core.ContentTypeText || block.Text == "" {
			continue
		}
		parts = append(parts, block.Text)
	}
	return strings.Join(parts, "\n")
}

func corePartsToAnthropic(parts []core.ContentPart) []anthropicproto.ContentBlock {
	out := make([]anthropicproto.ContentBlock, 0, len(parts))
	for _, part := range parts {
		switch part.Type {
		case core.ContentTypeText:
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

func coreToolsToAnthropic(tools []core.ToolDefinition) []anthropicproto.ToolDefinition {
	out := make([]anthropicproto.ToolDefinition, 0, len(tools))
	for _, tool := range tools {
		var schema any
		if tool.InputSchema != "" {
			_ = json.Unmarshal([]byte(tool.InputSchema), &schema)
		}
		out = append(out, anthropicproto.ToolDefinition{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: schema,
		})
	}
	return out
}

func coreToolChoiceToAnthropic(choice *core.ToolChoice) any {
	if choice == nil {
		return nil
	}
	if choice.Name == "" {
		return map[string]any{"type": choice.Type}
	}
	return map[string]any{"type": choice.Type, "name": choice.Name}
}

func anthropicBlocksToCore(blocks []anthropicproto.ContentBlock) []core.ContentPart {
	out := make([]core.ContentPart, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case "text":
			out = append(out, core.ContentPart{Type: core.ContentTypeText, Text: block.Text})
		case "tool_use":
			input, _ := json.Marshal(block.Input)
			out = append(out, core.ContentPart{
				Type:       core.ContentTypeToolCall,
				Name:       block.Name,
				ToolCallID: block.ID,
				Input:      string(input),
			})
		}
	}
	return out
}

func anthropicBlockToStreamPart(block anthropicproto.ContentBlock) core.ContentPart {
	switch block.Type {
	case "text":
		return core.ContentPart{Type: core.ContentTypeText, Text: block.Text}
	case "image":
		part := core.ContentPart{Type: core.ContentTypeImage}
		if block.Source != nil {
			part.MediaType = block.Source.MediaType
			part.Data = block.Source.Data
		}
		return part
	case "tool_use":
		input, _ := json.Marshal(block.Input)
		return core.ContentPart{
			Type:       core.ContentTypeToolCall,
			Name:       block.Name,
			ToolCallID: block.ID,
			Input:      string(input),
		}
	default:
		return core.ContentPart{Type: block.Type}
	}
}
