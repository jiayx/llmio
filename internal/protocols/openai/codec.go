package openai

import (
	"encoding/json"
	"fmt"

	"github.com/jiayx/llmio/internal/llm"
)

// ChatCompletionRequestFromLLM encodes one internal request as an OpenAI chat completions payload.
func ChatCompletionRequestFromLLM(req llm.ChatRequest) ChatCompletionRequest {
	messages := make([]Message, 0, len(req.Messages)+1)
	if len(req.System) > 0 {
		systemContent, _ := llmContentToOpenAI(req.System)
		messages = append(messages, Message{
			Role:    "system",
			Content: systemContent,
		})
	}
	for _, msg := range req.Messages {
		content, toolCalls := llmContentToOpenAI(msg.Content)
		role := msg.Role
		if role == "user" && llm.HasOnlyContentType(msg.Content, llm.ContentTypeToolResult) {
			role = "tool"
			for _, part := range msg.Content {
				if part.Type != llm.ContentTypeToolResult {
					continue
				}
				messages = append(messages, Message{
					Role:       role,
					Content:    part.Output,
					ToolCallID: part.ToolCallID,
				})
			}
			continue
		}
		messages = append(messages, Message{
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

	return ChatCompletionRequest{
		Model:       req.Model,
		Messages:    messages,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		MaxTokens:   maxTokens,
		Stream:      req.Stream,
		User:        req.User,
		Tools:       llmToolsToOpenAI(req.Tools),
		ToolChoice:  llmToolChoiceToOpenAI(req.ToolChoice),
	}
}

// ChatCompletionResponseToLLM decodes one OpenAI chat completions payload into the internal response model.
func ChatCompletionResponseToLLM(resp ChatCompletionResponse, raw []byte) *llm.ChatResponse {
	out := &llm.ChatResponse{
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
		out.Output = chatCompletionMessageToLLM(resp.Choices[0].Message)
		out.OutputText = llm.ExtractText(out.Output)
	}
	return out
}

// StreamChunkPayloadToLLMEvents decodes one OpenAI streaming chunk payload into internal events.
func StreamChunkPayloadToLLMEvents(payload string) ([]llm.StreamEvent, error) {
	var chunk StreamChunk
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
		return nil, fmt.Errorf("decode openai stream chunk: %w", err)
	}

	events := make([]llm.StreamEvent, 0, 4)
	raw := json.RawMessage(payload)
	if len(chunk.Choices) > 0 {
		choice := chunk.Choices[0]
		if choice.Delta.Content != "" {
			events = append(events, llm.StreamEvent{
				Type:      llm.StreamEventDelta,
				Part:      llm.ContentPart{Type: llm.ContentTypeText},
				TextDelta: choice.Delta.Content,
				Raw:       raw,
			})
		}
		if choice.Delta.Reasoning != "" {
			events = append(events, llm.StreamEvent{
				Type:      llm.StreamEventDelta,
				Part:      llm.ContentPart{Type: llm.ContentTypeReasoning},
				TextDelta: choice.Delta.Reasoning,
				Raw:       raw,
			})
		}
		for _, toolCall := range choice.Delta.ToolCalls {
			if toolCall.ID == "" && toolCall.Function.Name == "" && toolCall.Function.Arguments == "" {
				continue
			}
			events = append(events, llm.StreamEvent{
				Type:       llm.StreamEventTool,
				ToolIndex:  toolCall.Index,
				ToolCallID: toolCall.ID,
				ToolName:   toolCall.Function.Name,
				ToolInput:  toolCall.Function.Arguments,
				Raw:        raw,
			})
		}
		if choice.FinishReason != "" {
			events = append(events, llm.StreamEvent{
				Type:         llm.StreamEventStop,
				FinishReason: choice.FinishReason,
				Raw:          raw,
			})
		}
	}
	if chunk.Usage != nil {
		events = append(events, llm.StreamEvent{
			Type:         llm.StreamEventUsage,
			InputTokens:  chunk.Usage.PromptTokens,
			OutputTokens: chunk.Usage.CompletionTokens,
			Raw:          raw,
		})
	}
	return events, nil
}

func chatCompletionMessageToLLM(msg Message) []llm.ContentPart {
	parts := make([]llm.ContentPart, 0)
	switch v := msg.Content.(type) {
	case string:
		if v != "" {
			parts = append(parts, llm.ContentPart{Type: llm.ContentTypeText, Text: v})
		}
	case []any:
		for _, item := range v {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}
			switch m["type"] {
			case "text":
				parts = append(parts, llm.ContentPart{Type: llm.ContentTypeText, Text: stringValue(m["text"])})
			case "image_url":
				if image, ok := m["image_url"].(map[string]any); ok {
					parts = append(parts, llm.ContentPart{Type: llm.ContentTypeImage, URL: stringValue(image["url"])})
				}
			}
		}
	}
	for _, call := range msg.ToolCalls {
		parts = append(parts, llm.ContentPart{
			Type:       llm.ContentTypeToolCall,
			Name:       call.Function.Name,
			ToolCallID: call.ID,
			Input:      call.Function.Arguments,
		})
	}
	return parts
}

func llmToolsToOpenAI(tools []llm.ToolDefinition) []Tool {
	out := make([]Tool, 0, len(tools))
	for _, tool := range tools {
		out = append(out, Tool{
			Type: "function",
			Function: FunctionDefinition{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  []byte(tool.InputSchema),
			},
		})
	}
	return out
}

func llmToolChoiceToOpenAI(choice *llm.ToolChoice) any {
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
