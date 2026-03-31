package anthropic

import (
	"encoding/json"
	"fmt"

	"github.com/jiayx/llmio/internal/llm"
)

// MessagesRequestFromLLM encodes one internal request as an Anthropic messages payload.
func MessagesRequestFromLLM(req llm.ChatRequest) MessagesRequest {
	messages := make([]Message, 0, len(req.Messages))
	for _, msg := range req.Messages {
		messages = append(messages, Message{
			Role:    msg.Role,
			Content: llmContentToAnthropic(msg.Content),
		})
	}

	var system any
	if len(req.System) > 0 {
		system = llmContentToAnthropic(req.System)
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 1024
	}

	out := MessagesRequest{
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

// MessagesResponseToLLM decodes one Anthropic messages payload into the internal response model.
func MessagesResponseToLLM(resp MessagesResponse, raw []byte) *llm.ChatResponse {
	out := &llm.ChatResponse{
		ID:                       resp.ID,
		Model:                    resp.Model,
		FinishReason:             resp.StopReason,
		InputTokens:              resp.Usage.InputTokens,
		CachedInputTokens:        resp.Usage.CacheReadInputTokens,
		CacheReadInputTokens:     resp.Usage.CacheReadInputTokens,
		CacheCreationInputTokens: resp.Usage.CacheCreationInputTokens,
		OutputTokens:             resp.Usage.OutputTokens,
		Raw:                      raw,
	}
	out.Output = contentBlocksToLLM(resp.Content)
	out.OutputText = llm.ExtractText(out.Output)
	return out
}

// SSEEventToLLMEvents decodes one Anthropic SSE event payload into internal stream events.
func SSEEventToLLMEvents(eventName, payload string, blockState map[int]ContentBlock, out chan<- llm.StreamEvent) error {
	switch eventName {
	case "message_start":
		var event StreamMessageStart
		if err := json.Unmarshal([]byte(payload), &event); err != nil {
			return fmt.Errorf("decode anthropic message_start: %w", err)
		}
		out <- llm.StreamEvent{
			Type:                     llm.StreamEventUsage,
			InputTokens:              event.Message.Usage.InputTokens,
			CachedInputTokens:        event.Message.Usage.CacheReadInputTokens,
			CacheReadInputTokens:     event.Message.Usage.CacheReadInputTokens,
			CacheCreationInputTokens: event.Message.Usage.CacheCreationInputTokens,
			Raw:                      json.RawMessage(payload),
		}
	case "content_block_start":
		var event StreamContentBlockStart
		if err := json.Unmarshal([]byte(payload), &event); err != nil {
			return fmt.Errorf("decode anthropic content_block_start: %w", err)
		}
		blockState[event.Index] = event.ContentBlock
		out <- llm.StreamEvent{
			Type:       llm.StreamEventContentStart,
			BlockIndex: event.Index,
			Part:       contentBlockToStreamPart(event.ContentBlock),
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
		var event StreamContentBlockDelta
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
			part := contentBlockToStreamPart(blockState[event.Index])
			delete(blockState, event.Index)
			out <- llm.StreamEvent{
				Type:       llm.StreamEventContentStop,
				BlockIndex: event.Index,
				Part:       part,
				Raw:        json.RawMessage(payload),
			}
		}
	case "message_delta":
		var event StreamMessageDelta
		if err := json.Unmarshal([]byte(payload), &event); err != nil {
			return fmt.Errorf("decode anthropic message_delta: %w", err)
		}
		if event.Usage.OutputTokens > 0 {
			out <- llm.StreamEvent{
				Type:                     llm.StreamEventUsage,
				OutputTokens:             event.Usage.OutputTokens,
				CacheReadInputTokens:     event.Usage.CacheReadInputTokens,
				CacheCreationInputTokens: event.Usage.CacheCreationInputTokens,
				CachedInputTokens:        event.Usage.CacheReadInputTokens,
				Raw:                      json.RawMessage(payload),
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
			Error Error `json:"error"`
		}
		if err := json.Unmarshal([]byte(payload), &wrapper); err != nil {
			return fmt.Errorf("decode anthropic error: %w", err)
		}
		return fmt.Errorf("anthropic stream error: %s", wrapper.Error.Message)
	}
	return nil
}

func contentBlocksToLLM(blocks []ContentBlock) []llm.ContentPart {
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

func contentBlockToStreamPart(block ContentBlock) llm.ContentPart {
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
		return llm.ContentPart{}
	}
}

func llmToolsToAnthropic(tools []llm.ToolDefinition) []ToolDefinition {
	out := make([]ToolDefinition, 0, len(tools))
	for _, tool := range tools {
		var schema any
		if tool.InputSchema != "" {
			_ = json.Unmarshal([]byte(tool.InputSchema), &schema)
		}
		out = append(out, ToolDefinition{
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
