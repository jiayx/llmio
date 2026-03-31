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
		systemContent, _, systemReasoning := llmContentToOpenAI(req.System)
		messages = append(messages, Message{
			Role:             "system",
			Content:          systemContent,
			ReasoningContent: systemReasoning,
		})
	}
	for _, msg := range req.Messages {
		content, toolCalls, reasoningContent := llmContentToOpenAI(msg.Content)
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
			Role:             role,
			Content:          content,
			Name:             msg.Name,
			ToolCalls:        toolCalls,
			ReasoningContent: reasoningContent,
		})
	}

	var maxTokens *int
	if req.MaxTokens > 0 {
		maxTokens = &req.MaxTokens
	}

	var streamOptions *ChatCompletionStreamOptions
	if req.Stream {
		streamOptions = &ChatCompletionStreamOptions{IncludeUsage: true}
	}

	return ChatCompletionRequest{
		Model:         req.Model,
		Messages:      messages,
		Temperature:   req.Temperature,
		TopP:          req.TopP,
		MaxTokens:     maxTokens,
		Stream:        req.Stream,
		StreamOptions: streamOptions,
		User:          req.User,
		Tools:         llmToolsToOpenAI(req.Tools),
		ToolChoice:    llmToolChoiceToOpenAI(req.ToolChoice),
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
		out.CachedInputTokens = cachedPromptTokens(resp.Usage)
		out.CacheReadInputTokens = cachedPromptTokens(resp.Usage)
		out.OutputTokens = resp.Usage.CompletionTokens
	}
	if len(resp.Choices) > 0 {
		out.FinishReason = resp.Choices[0].FinishReason
		out.Output = chatCompletionMessageToLLM(resp.Choices[0].Message)
		out.OutputText = llm.ExtractText(out.Output)
	}
	return out
}

// ResponsesResponseToLLM decodes one OpenAI responses payload into the internal response model.
func ResponsesResponseToLLM(resp ResponsesResponse, raw []byte) *llm.ChatResponse {
	out := &llm.ChatResponse{
		ID:         resp.ID,
		Model:      resp.Model,
		OutputText: resp.OutputText,
		Raw:        raw,
	}
	if resp.Usage != nil {
		out.InputTokens = resp.Usage.InputTokens
		out.CachedInputTokens = cachedResponseInputTokens(resp.Usage)
		out.CacheReadInputTokens = cachedResponseInputTokens(resp.Usage)
		out.OutputTokens = resp.Usage.OutputTokens
	}
	out.Output = responsesOutputToLLM(resp.Output)
	if out.OutputText == "" {
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
		reasoning := choice.Delta.Reasoning
		if reasoning == "" {
			reasoning = choice.Delta.ReasoningContent
		}
		if reasoning != "" {
			events = append(events, llm.StreamEvent{
				Type:      llm.StreamEventDelta,
				Part:      llm.ContentPart{Type: llm.ContentTypeReasoning},
				TextDelta: reasoning,
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
		if choice.Usage != nil {
			events = append(events, llm.StreamEvent{
				Type:                 llm.StreamEventUsage,
				InputTokens:          choice.Usage.PromptTokens,
				CachedInputTokens:    cachedPromptTokens(choice.Usage),
				CacheReadInputTokens: cachedPromptTokens(choice.Usage),
				OutputTokens:         choice.Usage.CompletionTokens,
				Raw:                  raw,
			})
		}
	}
	if chunk.Usage != nil {
		events = append(events, llm.StreamEvent{
			Type:                 llm.StreamEventUsage,
			InputTokens:          chunk.Usage.PromptTokens,
			CachedInputTokens:    cachedPromptTokens(chunk.Usage),
			CacheReadInputTokens: cachedPromptTokens(chunk.Usage),
			OutputTokens:         chunk.Usage.CompletionTokens,
			Raw:                  raw,
		})
	}
	return events, nil
}

func cachedPromptTokens(usage *CompletionUsage) int {
	if usage == nil || usage.PromptTokensDetails == nil {
		return 0
	}
	return usage.PromptTokensDetails.CachedTokens
}

func cachedResponseInputTokens(usage *ResponseUsage) int {
	if usage == nil || usage.InputTokensDetails == nil {
		return 0
	}
	return usage.InputTokensDetails.CachedTokens
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

func responsesOutputToLLM(items []ResponseOutputItem) []llm.ContentPart {
	parts := make([]llm.ContentPart, 0, len(items))
	for _, item := range items {
		switch item.Type {
		case "message":
			for _, content := range item.Content {
				switch content.Type {
				case "output_text", "text":
					parts = append(parts, llm.ContentPart{Type: llm.ContentTypeText, Text: content.Text})
				case "input_image":
					parts = append(parts, llm.ContentPart{Type: llm.ContentTypeImage, URL: content.ImageURL})
				}
			}
		case "function_call":
			parts = append(parts, llm.ContentPart{
				Type:       llm.ContentTypeToolCall,
				Name:       item.Name,
				ToolCallID: item.CallID,
				Input:      item.Arguments,
			})
		case "function_call_output":
			parts = append(parts, llm.ContentPart{
				Type:       llm.ContentTypeToolResult,
				ToolCallID: item.CallID,
				Output:     item.OutputData,
			})
		}
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
