package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/jiayx/llmio/internal/llm"
	httpio "github.com/jiayx/llmio/internal/protocols/httpio"
	providerapi "github.com/jiayx/llmio/internal/providers/api"
	"github.com/jiayx/llmio/internal/routing"
)

// DecodeChatCompletionRequest decodes a chat completions request body.
func DecodeChatCompletionRequest(body []byte) (ChatCompletionRequest, error) {
	var req ChatCompletionRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return ChatCompletionRequest{}, fmt.Errorf("invalid json: %w", err)
	}
	return req, nil
}

// DecodeResponsesRequest decodes a responses request body.
func DecodeResponsesRequest(body []byte) (ResponsesRequest, error) {
	var req ResponsesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return ResponsesRequest{}, fmt.Errorf("invalid json: %w", err)
	}
	return req, nil
}

// WriteError writes an OpenAI-style error response.
func WriteError(w http.ResponseWriter, status int, message string) {
	slog.Error("request failed",
		"protocol", "openai",
		"status", status,
		"error_type", "invalid_request_error",
		"message", message,
	)
	httpio.WriteJSON(w, status, ErrorResponse{
		Error: &CompletionError{
			Message: message,
			Type:    "invalid_request_error",
		},
	})
}

// WriteModels writes the OpenAI model inventory response.
func WriteModels(w http.ResponseWriter, infos []routing.ModelInfo) {
	models := make([]ModelInfo, 0, len(infos))
	for _, info := range infos {
		models = append(models, ModelInfo{
			ID:      info.ID,
			Object:  "model",
			OwnedBy: info.OwnedBy,
		})
	}
	httpio.WriteJSON(w, http.StatusOK, ModelsResponse{
		Object: "list",
		Data:   models,
	})
}

// WriteChatCompletionResponse writes one normalized response as OpenAI chat completions JSON.
func WriteChatCompletionResponse(w http.ResponseWriter, externalModel string, resp *llm.ChatResponse) {
	parts := resp.EffectiveOutput()
	content, toolCalls, reasoningContent := llmContentToOpenAI(parts)
	created := time.Now().Unix()
	httpio.WriteJSON(w, http.StatusOK, ChatCompletionResponse{
		ID:      resp.ID,
		Object:  "chat.completion",
		Created: created,
		Model:   externalModel,
		Choices: []Choice{{Index: 0, Message: Message{Role: "assistant", Content: content, ToolCalls: toolCalls, ReasoningContent: reasoningContent}, FinishReason: resp.FinishReason}},
		Usage: &CompletionUsage{
			PromptTokens:     resp.InputTokens,
			CompletionTokens: resp.OutputTokens,
			TotalTokens:      resp.InputTokens + resp.OutputTokens,
		},
	})
}

// WriteResponsesResponse writes one normalized response as OpenAI responses JSON.
func WriteResponsesResponse(w http.ResponseWriter, externalModel string, resp *llm.ChatResponse) {
	parts := resp.EffectiveOutput()
	httpio.WriteJSON(w, http.StatusOK, ResponsesResponse{
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
	})
}

// WritePassthroughResponse rewrites an upstream OpenAI response back to the external model name.
func WritePassthroughResponse(w http.ResponseWriter, resp *http.Response, externalModel string) {
	httpio.WritePassthroughResponse(w, resp, func(data []byte) ([]byte, error) {
		return rewritePassthroughJSONPayload(data, externalModel)
	})
}

// ServeChatCompletionStream renders normalized events as OpenAI chat.completion.chunk SSE.
func ServeChatCompletionStream(w http.ResponseWriter, r *http.Request, externalModel string, stream *providerapi.StreamReader) {
	defer closeStream(stream)

	flusher, ok := w.(http.Flusher)
	if !ok {
		WriteError(w, http.StatusInternalServerError, "streaming is not supported by this server")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	toolStates := make(map[string]*streamToolState)
	toolOrder := make([]string, 0)
	chunkID := streamObjectID("chatcmpl")
	created := time.Now().Unix()
	emittedRole := false

	for {
		select {
		case event, ok := <-stream.Events:
			if !ok {
				httpio.WriteSSE(w, "", "[DONE]")
				flusher.Flush()
				return
			}
			switch event.Type {
			case llm.StreamEventDelta:
				chunk := StreamChunk{
					ID:      chunkID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   externalModel,
					Choices: []StreamChoice{{
						Index: 0,
						Delta: StreamDelta{},
					}},
				}
				if !emittedRole {
					chunk.Choices[0].Delta.Role = "assistant"
					emittedRole = true
				}
				if event.Part.Type == llm.ContentTypeReasoning {
					chunk.Choices[0].Delta.Reasoning = event.TextDelta
				} else {
					chunk.Choices[0].Delta.Content = event.TextDelta
				}
				httpio.WriteSSEJSON(w, "", chunk)
				flusher.Flush()
			case llm.StreamEventTool:
				key, state := upsertStreamToolState(toolStates, &toolOrder, event)
				chunk := StreamChunk{
					ID:      chunkID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   externalModel,
					Choices: []StreamChoice{{
						Index: 0,
						Delta: StreamDelta{
							ToolCalls: []ToolCall{{
								Index: state.Index,
								ID:    state.ID,
								Type:  "function",
								Function: FunctionCall{
									Name:      state.Name,
									Arguments: event.ToolInput,
								},
							}},
						},
					}},
				}
				if key != "" {
					httpio.WriteSSEJSON(w, "", chunk)
					flusher.Flush()
				}
			case llm.StreamEventStop:
				chunk := StreamChunk{
					ID:      chunkID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   externalModel,
					Choices: []StreamChoice{{
						Index:        0,
						Delta:        StreamDelta{},
						FinishReason: event.FinishReason,
					}},
				}
				httpio.WriteSSEJSON(w, "", chunk)
				httpio.WriteSSE(w, "", "[DONE]")
				flusher.Flush()
				return
			}
		case err, ok := <-stream.Err:
			if ok && err != nil {
				if shouldIgnoreStreamError(r, err) {
					logDownstreamDisconnected(r, "openai", "chat_completions", err)
				} else {
					slog.Error("stream failed", "protocol", "openai", "api_type", "chat_completions", "err", err)
					httpio.WriteSSE(w, "", "[DONE]")
					flusher.Flush()
				}
			}
			return
		}
	}
}

// ServeResponsesStream renders normalized events as OpenAI responses SSE.
func ServeResponsesStream(w http.ResponseWriter, r *http.Request, externalModel string, stream *providerapi.StreamReader) {
	defer closeStream(stream)

	flusher, ok := w.(http.Flusher)
	if !ok {
		WriteError(w, http.StatusInternalServerError, "streaming is not supported by this server")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	responseID := streamObjectID("resp")
	createdAt := time.Now().Unix()
	httpio.WriteSSEJSON(w, "response.created", ResponsesResponse{
		ID:        responseID,
		Object:    "response",
		CreatedAt: createdAt,
		Model:     externalModel,
		Status:    "in_progress",
		Output:    []ResponseOutputItem{},
	})
	flusher.Flush()

	var output strings.Builder
	var reasoning strings.Builder
	inputTokens := 0
	outputTokens := 0
	toolStates := make(map[string]*streamToolState)
	toolOrder := make([]string, 0)
	imageBlocks := make(map[int]llm.ContentPart)
	imageOrder := make([]int, 0)

	for {
		select {
		case event, ok := <-stream.Events:
			if !ok {
				httpio.WriteSSEJSON(w, "response.completed", ResponsesResponse{
					ID:         responseID,
					Object:     "response",
					CreatedAt:  createdAt,
					Model:      externalModel,
					Status:     "completed",
					Output:     responseOutputItems(streamResponseParts(output.String(), reasoning.String(), imageBlocks, imageOrder, toolStates, toolOrder)),
					OutputText: output.String(),
					Usage: &ResponseUsage{
						InputTokens:  inputTokens,
						OutputTokens: outputTokens,
						TotalTokens:  inputTokens + outputTokens,
					},
				})
				flusher.Flush()
				return
			}

			switch event.Type {
			case llm.StreamEventContentStart:
				if event.Part.Type == llm.ContentTypeImage {
					imageBlocks[event.BlockIndex] = event.Part
					imageOrder = append(imageOrder, event.BlockIndex)
					httpio.WriteSSEJSON(w, "response.output_item.added", map[string]any{
						"type":  "response.output_item.added",
						"index": event.BlockIndex,
						"item": ResponseOutputItem{
							Type:   "message",
							Role:   "assistant",
							Status: "in_progress",
							Content: []ResponseContentPart{{
								Type:     "input_image",
								ImageURL: event.Part.URL,
							}},
						},
					})
					flusher.Flush()
				}
			case llm.StreamEventDelta:
				if event.Part.Type == llm.ContentTypeReasoning {
					reasoning.WriteString(event.TextDelta)
					httpio.WriteSSEJSON(w, "response.reasoning.delta", map[string]any{
						"type":  "response.reasoning.delta",
						"delta": event.TextDelta,
					})
				} else {
					output.WriteString(event.TextDelta)
					httpio.WriteSSEJSON(w, "response.output_text.delta", map[string]any{
						"type":  "response.output_text.delta",
						"delta": event.TextDelta,
					})
				}
				flusher.Flush()
			case llm.StreamEventTool:
				key, state := upsertStreamToolState(toolStates, &toolOrder, event)
				if !state.Emitted {
					httpio.WriteSSEJSON(w, "response.output_item.added", map[string]any{
						"type":  "response.output_item.added",
						"item":  ResponseOutputItem{Type: "function_call", Name: state.Name, CallID: state.ID, Status: "in_progress"},
						"index": state.Index,
					})
					state.Emitted = true
				}
				httpio.WriteSSEJSON(w, "response.function_call_arguments.delta", map[string]any{
					"type":         "response.function_call_arguments.delta",
					"call_id":      state.ID,
					"name":         state.Name,
					"output_index": state.Index,
					"delta":        event.ToolInput,
				})
				if key != "" {
					flusher.Flush()
				}
			case llm.StreamEventContentStop:
				if event.Part.Type == llm.ContentTypeImage {
					part := imageBlocks[event.BlockIndex]
					httpio.WriteSSEJSON(w, "response.output_item.done", map[string]any{
						"type":  "response.output_item.done",
						"index": event.BlockIndex,
						"item": ResponseOutputItem{
							Type:   "message",
							Role:   "assistant",
							Status: "completed",
							Content: []ResponseContentPart{{
								Type:     "input_image",
								ImageURL: part.URL,
							}},
						},
					})
					flusher.Flush()
				}
			case llm.StreamEventUsage:
				inputTokens = event.InputTokens
				outputTokens = event.OutputTokens
			case llm.StreamEventStop:
				httpio.WriteSSEJSON(w, "response.output_text.done", map[string]any{
					"type": "response.output_text.done",
					"text": output.String(),
				})
				if reasoning.Len() > 0 {
					httpio.WriteSSEJSON(w, "response.reasoning.done", map[string]any{
						"type": "response.reasoning.done",
						"text": reasoning.String(),
					})
				}
				for _, key := range toolOrder {
					state := toolStates[key]
					httpio.WriteSSEJSON(w, "response.output_item.done", map[string]any{
						"type":  "response.output_item.done",
						"index": state.Index,
						"item": ResponseOutputItem{
							Type:      "function_call",
							Name:      state.Name,
							CallID:    state.ID,
							Arguments: state.Input.String(),
							Status:    "completed",
						},
					})
				}
				httpio.WriteSSEJSON(w, "response.completed", ResponsesResponse{
					ID:         responseID,
					Object:     "response",
					CreatedAt:  createdAt,
					Model:      externalModel,
					Status:     "completed",
					Output:     responseOutputItems(streamResponseParts(output.String(), reasoning.String(), imageBlocks, imageOrder, toolStates, toolOrder)),
					OutputText: output.String(),
					Usage: &ResponseUsage{
						InputTokens:  inputTokens,
						OutputTokens: outputTokens,
						TotalTokens:  inputTokens + outputTokens,
					},
				})
				flusher.Flush()
				return
			}
		case err, ok := <-stream.Err:
			if ok && err != nil {
				if shouldIgnoreStreamError(r, err) {
					logDownstreamDisconnected(r, "openai", "responses", err)
				} else {
					slog.Error("stream failed", "protocol", "openai", "api_type", "responses", "err", err)
				}
			}
			return
		}
	}
}

func rewritePassthroughJSONPayload(data []byte, externalModel string) ([]byte, error) {
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil, err
	}
	rewriteMapModel(payload, externalModel)
	return json.Marshal(payload)
}

func rewriteMapModel(m map[string]any, model string) {
	if _, ok := m["model"]; ok {
		m["model"] = model
	}
}

func responseOutputItems(parts []llm.ContentPart) []ResponseOutputItem {
	if len(parts) == 0 {
		return nil
	}
	var items []ResponseOutputItem
	var messageParts []ResponseContentPart
	for _, part := range parts {
		switch part.Type {
		case llm.ContentTypeText:
			messageParts = append(messageParts, ResponseContentPart{Type: "output_text", Text: part.Text})
		case llm.ContentTypeImage:
			messageParts = append(messageParts, ResponseContentPart{Type: "input_image", ImageURL: part.URL})
		case llm.ContentTypeReasoning:
			items = append(items, ResponseOutputItem{
				Type:   "reasoning",
				Status: "completed",
				Content: []ResponseContentPart{{
					Type: "output_text",
					Text: part.Text,
				}},
			})
		case llm.ContentTypeToolCall:
			items = append(items, ResponseOutputItem{
				Type:      "function_call",
				Name:      part.Name,
				CallID:    part.ToolCallID,
				Arguments: part.Input,
				Status:    "completed",
			})
		}
	}
	if len(messageParts) > 0 {
		items = append([]ResponseOutputItem{{
			Type:    "message",
			Role:    "assistant",
			Status:  "completed",
			Content: messageParts,
		}}, items...)
	}
	return items
}

func llmContentToOpenAI(parts []llm.ContentPart) (any, []ToolCall, string) {
	var content []map[string]any
	var toolCalls []ToolCall
	var reasoningParts []string
	for _, part := range parts {
		switch part.Type {
		case llm.ContentTypeText:
			content = append(content, map[string]any{"type": "text", "text": part.Text})
		case llm.ContentTypeImage:
			if part.URL != "" {
				content = append(content, map[string]any{"type": "image_url", "image_url": map[string]any{"url": part.URL}})
			}
		case llm.ContentTypeReasoning:
			if strings.TrimSpace(part.Text) != "" {
				reasoningParts = append(reasoningParts, part.Text)
			}
		case llm.ContentTypeToolCall:
			toolCalls = append(toolCalls, ToolCall{
				ID:   part.ToolCallID,
				Type: "function",
				Function: FunctionCall{
					Name:      part.Name,
					Arguments: part.Input,
				},
			})
		}
	}
	reasoningContent := strings.Join(reasoningParts, "\n")
	if len(content) == 0 {
		return nil, toolCalls, reasoningContent
	}
	if len(content) == 1 && content[0]["type"] == "text" {
		return content[0]["text"], toolCalls, reasoningContent
	}
	out := make([]any, 0, len(content))
	for _, item := range content {
		out = append(out, item)
	}
	return out, toolCalls, reasoningContent
}

func streamObjectID(prefix string) string {
	return fmt.Sprintf("%s_%d", prefix, time.Now().UnixNano())
}

func closeStream(stream *providerapi.StreamReader) {
	if stream != nil && stream.Close != nil {
		_ = stream.Close()
	}
}

func shouldIgnoreStreamError(r *http.Request, err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) || errors.Is(r.Context().Err(), context.Canceled) {
		return true
	}
	return strings.Contains(strings.ToLower(err.Error()), "context canceled")
}

func logDownstreamDisconnected(r *http.Request, protocol, apiType string, err error) {
	slog.Info("downstream disconnected",
		"protocol", protocol,
		"api_type", apiType,
		"path", r.URL.Path,
		"remote", r.RemoteAddr,
		"err", err,
	)
}

type streamToolState struct {
	ID      string
	Name    string
	Index   int
	Input   strings.Builder
	Emitted bool
}

func upsertStreamToolState(states map[string]*streamToolState, order *[]string, event llm.StreamEvent) (string, *streamToolState) {
	key := event.ToolCallID
	if key == "" {
		key = fmt.Sprintf("idx:%d:%s", event.ToolIndex, event.ToolName)
	}
	state, ok := states[key]
	if !ok {
		index := len(states)
		if event.ToolIndex >= 0 {
			index = event.ToolIndex
		}
		state = &streamToolState{ID: event.ToolCallID, Name: event.ToolName, Index: index}
		states[key] = state
		if order != nil {
			*order = append(*order, key)
		}
	}
	if event.ToolCallID != "" {
		state.ID = event.ToolCallID
	}
	if event.ToolName != "" {
		state.Name = event.ToolName
	}
	if event.ToolInput != "" {
		state.Input.WriteString(event.ToolInput)
	}
	return key, state
}

func streamToolParts(states map[string]*streamToolState, order []string) []llm.ContentPart {
	out := make([]llm.ContentPart, 0, len(order))
	for _, key := range order {
		state := states[key]
		out = append(out, llm.ContentPart{
			Type:       llm.ContentTypeToolCall,
			Name:       state.Name,
			ToolCallID: state.ID,
			Input:      state.Input.String(),
		})
	}
	return out
}

func streamResponseParts(text, reasoning string, imageBlocks map[int]llm.ContentPart, imageOrder []int, toolStates map[string]*streamToolState, toolOrder []string) []llm.ContentPart {
	parts := make([]llm.ContentPart, 0, 1+len(imageOrder)+len(toolOrder)+1)
	parts = append(parts, llm.TextParts(text)...)
	if reasoning != "" {
		parts = append(parts, llm.ContentPart{Type: llm.ContentTypeReasoning, Text: reasoning})
	}
	for _, index := range imageOrder {
		if part, ok := imageBlocks[index]; ok {
			parts = append(parts, part)
		}
	}
	parts = append(parts, streamToolParts(toolStates, toolOrder)...)
	return parts
}
