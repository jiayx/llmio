package anthropic

import (
	"encoding/json"
	"fmt"

	"github.com/jiayx/llmio/internal/llm"
	anthropicproto "github.com/jiayx/llmio/internal/wire/anthropic"
)

func llmToAnthropic(req llm.ChatRequest) anthropicproto.MessagesRequest {
	messages := make([]anthropicproto.Message, 0, len(req.Messages))
	for _, msg := range req.Messages {
		messages = append(messages, anthropicproto.Message{
			Role:    msg.Role,
			Content: llmPartsToAnthropic(msg.Content),
		})
	}

	var system any
	if len(req.System) > 0 {
		system = llmPartsToAnthropic(req.System)
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
		Tools:       llmToolsToAnthropic(req.Tools),
		ToolChoice:  llmToolChoiceToAnthropic(req.ToolChoice),
	}
	if req.User != "" {
		out.Metadata = map[string]string{"user_id": req.User}
	}
	return out
}

func anthropicToLLMResponse(resp anthropicproto.MessagesResponse, raw []byte) *llm.ChatResponse {
	out := &llm.ChatResponse{
		ID:           resp.ID,
		Model:        resp.Model,
		FinishReason: resp.StopReason,
		InputTokens:  resp.Usage.InputTokens,
		OutputTokens: resp.Usage.OutputTokens,
		Raw:          raw,
	}
	out.Output = anthropicBlocksToLLM(resp.Content)
	out.OutputText = llm.ExtractText(out.Output)
	return out
}

func anthropicSSEToLLMEvents(eventName, payload string, blockState map[int]anthropicproto.ContentBlock, out chan<- llm.StreamEvent) error {
	switch eventName {
	case "message_start":
		var event anthropicproto.StreamMessageStart
		if err := json.Unmarshal([]byte(payload), &event); err != nil {
			return fmt.Errorf("decode anthropic message_start: %w", err)
		}
		out <- llm.StreamEvent{
			Type:        llm.StreamEventUsage,
			InputTokens: event.Message.Usage.InputTokens,
			Raw:         json.RawMessage(payload),
		}
	case "content_block_start":
		var event anthropicproto.StreamContentBlockStart
		if err := json.Unmarshal([]byte(payload), &event); err != nil {
			return fmt.Errorf("decode anthropic content_block_start: %w", err)
		}
		blockState[event.Index] = event.ContentBlock
		out <- llm.StreamEvent{
			Type:       llm.StreamEventContentStart,
			BlockIndex: event.Index,
			Part:       anthropicBlockToStreamPart(event.ContentBlock),
			Raw:        json.RawMessage(payload),
		}
		if event.ContentBlock.Type == "tool_use" {
			input, _ := json.Marshal(event.ContentBlock.Input)
			out <- llm.StreamEvent{
				Type:       llm.StreamEventTool,
				BlockIndex: event.Index,
				Part:       llm.ContentPart{Type: llm.ContentTypeToolCall},
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
					out <- llm.StreamEvent{
						Type:       llm.StreamEventDelta,
						BlockIndex: event.Index,
						Part:       llm.ContentPart{Type: llm.ContentTypeText},
						TextDelta:  text,
						Raw:        json.RawMessage(payload),
					}
				}
			case "input_json_delta":
				if partial := stringValue(delta["partial_json"]); partial != "" {
					state := blockState[event.Index]
					out <- llm.StreamEvent{
						Type:       llm.StreamEventTool,
						ToolCallID: state.ID,
						ToolName:   state.Name,
						BlockIndex: event.Index,
						Part:       llm.ContentPart{Type: llm.ContentTypeToolCall},
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
			out <- llm.StreamEvent{
				Type:       llm.StreamEventContentStop,
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
			out <- llm.StreamEvent{
				Type:         llm.StreamEventUsage,
				OutputTokens: event.Usage.OutputTokens,
				Raw:          json.RawMessage(payload),
			}
		}
		if event.Delta.StopReason != "" {
			out <- llm.StreamEvent{
				Type:         llm.StreamEventStop,
				FinishReason: event.Delta.StopReason,
				Raw:          json.RawMessage(payload),
			}
		}
	case "message_stop":
		out <- llm.StreamEvent{Type: llm.StreamEventStop, Raw: json.RawMessage(payload)}
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

func llmPartsToAnthropic(parts []llm.ContentPart) []anthropicproto.ContentBlock {
	out := make([]anthropicproto.ContentBlock, 0, len(parts))
	for _, part := range parts {
		switch part.Type {
		case llm.ContentTypeText:
			out = append(out, anthropicproto.ContentBlock{Type: "text", Text: part.Text})
		case llm.ContentTypeImage:
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
		case llm.ContentTypeToolCall:
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
		case llm.ContentTypeToolResult:
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

func llmToolsToAnthropic(tools []llm.ToolDefinition) []anthropicproto.ToolDefinition {
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

func llmToolChoiceToAnthropic(choice *llm.ToolChoice) any {
	if choice == nil {
		return nil
	}
	if choice.Name == "" {
		return map[string]any{"type": choice.Type}
	}
	return map[string]any{"type": choice.Type, "name": choice.Name}
}

func anthropicBlocksToLLM(blocks []anthropicproto.ContentBlock) []llm.ContentPart {
	out := make([]llm.ContentPart, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case "text":
			out = append(out, llm.ContentPart{Type: llm.ContentTypeText, Text: block.Text})
		case "tool_use":
			input, _ := json.Marshal(block.Input)
			out = append(out, llm.ContentPart{
				Type:       llm.ContentTypeToolCall,
				Name:       block.Name,
				ToolCallID: block.ID,
				Input:      string(input),
			})
		}
	}
	return out
}

func anthropicBlockToStreamPart(block anthropicproto.ContentBlock) llm.ContentPart {
	switch block.Type {
	case "text":
		return llm.ContentPart{Type: llm.ContentTypeText, Text: block.Text}
	case "image":
		part := llm.ContentPart{Type: llm.ContentTypeImage}
		if block.Source != nil {
			part.MediaType = block.Source.MediaType
			part.Data = block.Source.Data
		}
		return part
	case "tool_use":
		input, _ := json.Marshal(block.Input)
		return llm.ContentPart{
			Type:       llm.ContentTypeToolCall,
			Name:       block.Name,
			ToolCallID: block.ID,
			Input:      string(input),
		}
	default:
		return llm.ContentPart{Type: block.Type}
	}
}

func stringValue(v any) string {
	s, _ := v.(string)
	return s
}
