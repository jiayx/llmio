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

	clientmeta "github.com/jiayx/llmio/internal/clients"
	"github.com/jiayx/llmio/internal/llm"
	providerapi "github.com/jiayx/llmio/internal/providers/api"
	"github.com/jiayx/llmio/internal/routing"
	transporthttp "github.com/jiayx/llmio/internal/transport/http"
	wire "github.com/jiayx/llmio/internal/wire/openai"
)

// DecodeChatCompletionRequest decodes a chat completions request body.
func DecodeChatCompletionRequest(body []byte) (wire.ChatCompletionRequest, error) {
	var req wire.ChatCompletionRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return wire.ChatCompletionRequest{}, fmt.Errorf("invalid json: %w", err)
	}
	return req, nil
}

// DecodeResponsesRequest decodes a responses request body.
func DecodeResponsesRequest(body []byte) (wire.ResponsesRequest, error) {
	var req wire.ResponsesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return wire.ResponsesRequest{}, fmt.Errorf("invalid json: %w", err)
	}
	return req, nil
}

// NewChatCompletionRequestMeta describes an inbound OpenAI chat completions request.
func NewChatCompletionRequestMeta(req wire.ChatCompletionRequest, body []byte, headers http.Header) clientmeta.RequestMeta {
	return clientmeta.RequestMeta{
		Protocol:      clientmeta.ProtocolOpenAI,
		APIType:       clientmeta.APIChatCompletions,
		Path:          "/chat/completions",
		ExternalModel: req.Model,
		Body:          body,
		Headers:       headers,
		Stream:        req.Stream,
	}
}

// NewResponsesRequestMeta describes an inbound OpenAI responses request.
func NewResponsesRequestMeta(req wire.ResponsesRequest, body []byte, headers http.Header) clientmeta.RequestMeta {
	return clientmeta.RequestMeta{
		Protocol:      clientmeta.ProtocolOpenAI,
		APIType:       clientmeta.APIResponses,
		Path:          "/responses",
		ExternalModel: req.Model,
		Body:          body,
		Headers:       headers,
		Stream:        req.Stream,
	}
}

// WriteError writes an OpenAI-style error response.
func WriteError(w http.ResponseWriter, status int, message string) {
	slog.Error("request failed",
		"protocol", "openai",
		"status", status,
		"error_type", "invalid_request_error",
		"message", message,
	)
	transporthttp.WriteJSON(w, status, wire.ChatCompletionResponse{
		Error: &wire.CompletionError{
			Message: message,
			Type:    "invalid_request_error",
		},
	})
}

// WriteModels writes the OpenAI model inventory response.
func WriteModels(w http.ResponseWriter, infos []routing.ModelInfo) {
	models := make([]wire.ModelInfo, 0, len(infos))
	for _, info := range infos {
		models = append(models, wire.ModelInfo{
			ID:      info.ID,
			Object:  "model",
			OwnedBy: info.OwnedBy,
		})
	}
	transporthttp.WriteJSON(w, http.StatusOK, wire.ModelsResponse{
		Object: "list",
		Data:   models,
	})
}

// WriteChatCompletionResponse writes one normalized response as OpenAI chat completions JSON.
func WriteChatCompletionResponse(w http.ResponseWriter, externalModel string, resp *llm.ChatResponse) {
	parts := resp.EffectiveOutput()
	content, toolCalls := llmContentToOpenAI(parts)
	transporthttp.WriteJSON(w, http.StatusOK, wire.ChatCompletionResponse{
		ID:      resp.ID,
		Object:  "chat.completion",
		Model:   externalModel,
		Choices: []wire.Choice{{Index: 0, Message: wire.Message{Role: "assistant", Content: content, ToolCalls: toolCalls}, FinishReason: resp.FinishReason}},
		Usage: &wire.CompletionUsage{
			PromptTokens:     resp.InputTokens,
			CompletionTokens: resp.OutputTokens,
			TotalTokens:      resp.InputTokens + resp.OutputTokens,
		},
	})
}

// WriteResponsesResponse writes one normalized response as OpenAI responses JSON.
func WriteResponsesResponse(w http.ResponseWriter, externalModel string, resp *llm.ChatResponse) {
	parts := resp.EffectiveOutput()
	transporthttp.WriteJSON(w, http.StatusOK, wire.ResponsesResponse{
		ID:         resp.ID,
		Object:     "response",
		CreatedAt:  time.Now().Unix(),
		Model:      externalModel,
		Status:     "completed",
		Output:     responseOutputItems(parts),
		OutputText: resp.OutputText,
		Usage: &wire.ResponseUsage{
			InputTokens:  resp.InputTokens,
			OutputTokens: resp.OutputTokens,
			TotalTokens:  resp.InputTokens + resp.OutputTokens,
		},
	})
}

// WritePassthroughResponse rewrites an upstream OpenAI response back to the external model name.
func WritePassthroughResponse(w http.ResponseWriter, resp *http.Response, externalModel string) {
	transporthttp.WritePassthroughResponse(w, resp, func(data []byte) ([]byte, error) {
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

	for {
		select {
		case event, ok := <-stream.Events:
			if !ok {
				transporthttp.WriteSSE(w, "", "[DONE]")
				flusher.Flush()
				return
			}
			switch event.Type {
			case llm.StreamEventDelta:
				chunk := wire.StreamChunk{
					Object: "chat.completion.chunk",
					Model:  externalModel,
					Choices: []wire.StreamChoice{{
						Index: 0,
						Delta: wire.StreamDelta{},
					}},
				}
				if event.Part.Type == llm.ContentTypeReasoning {
					chunk.Choices[0].Delta.Reasoning = event.TextDelta
				} else {
					chunk.Choices[0].Delta.Content = event.TextDelta
				}
				transporthttp.WriteSSEJSON(w, "", chunk)
				flusher.Flush()
			case llm.StreamEventTool:
				key, state := upsertStreamToolState(toolStates, &toolOrder, event)
				chunk := wire.StreamChunk{
					Object: "chat.completion.chunk",
					Model:  externalModel,
					Choices: []wire.StreamChoice{{
						Index: 0,
						Delta: wire.StreamDelta{
							ToolCalls: []wire.ToolCall{{
								Index: state.Index,
								ID:    state.ID,
								Type:  "function",
								Function: wire.FunctionCall{
									Name:      state.Name,
									Arguments: event.ToolInput,
								},
							}},
						},
					}},
				}
				if key != "" {
					transporthttp.WriteSSEJSON(w, "", chunk)
					flusher.Flush()
				}
			case llm.StreamEventStop:
				chunk := wire.StreamChunk{
					Object: "chat.completion.chunk",
					Model:  externalModel,
					Choices: []wire.StreamChoice{{
						Index:        0,
						Delta:        wire.StreamDelta{},
						FinishReason: event.FinishReason,
					}},
				}
				transporthttp.WriteSSEJSON(w, "", chunk)
				transporthttp.WriteSSE(w, "", "[DONE]")
				flusher.Flush()
				return
			}
		case err, ok := <-stream.Err:
			if ok && err != nil && !shouldIgnoreStreamError(r, err) {
				slog.Error("stream failed", "protocol", "openai", "api_type", "chat_completions", "err", err)
				transporthttp.WriteSSE(w, "", "[DONE]")
				flusher.Flush()
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

	transporthttp.WriteSSEJSON(w, "response.created", wire.ResponsesResponse{
		Object: "response",
		Model:  externalModel,
		Status: "in_progress",
		Output: []wire.ResponseOutputItem{},
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
				transporthttp.WriteSSEJSON(w, "response.completed", wire.ResponsesResponse{
					Object:     "response",
					Model:      externalModel,
					Status:     "completed",
					Output:     responseOutputItems(streamResponseParts(output.String(), reasoning.String(), imageBlocks, imageOrder, toolStates, toolOrder)),
					OutputText: output.String(),
					Usage: &wire.ResponseUsage{
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
					transporthttp.WriteSSEJSON(w, "response.output_item.added", map[string]any{
						"type":  "response.output_item.added",
						"index": event.BlockIndex,
						"item": wire.ResponseOutputItem{
							Type:   "message",
							Role:   "assistant",
							Status: "in_progress",
							Content: []wire.ResponseContentPart{{
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
					transporthttp.WriteSSEJSON(w, "response.reasoning.delta", map[string]any{
						"type":  "response.reasoning.delta",
						"delta": event.TextDelta,
					})
				} else {
					output.WriteString(event.TextDelta)
					transporthttp.WriteSSEJSON(w, "response.output_text.delta", map[string]any{
						"type":  "response.output_text.delta",
						"delta": event.TextDelta,
					})
				}
				flusher.Flush()
			case llm.StreamEventTool:
				key, state := upsertStreamToolState(toolStates, &toolOrder, event)
				if !state.Emitted {
					transporthttp.WriteSSEJSON(w, "response.output_item.added", map[string]any{
						"type":  "response.output_item.added",
						"item":  wire.ResponseOutputItem{Type: "function_call", Name: state.Name, CallID: state.ID, Status: "in_progress"},
						"index": state.Index,
					})
					state.Emitted = true
				}
				transporthttp.WriteSSEJSON(w, "response.function_call_arguments.delta", map[string]any{
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
					transporthttp.WriteSSEJSON(w, "response.output_item.done", map[string]any{
						"type":  "response.output_item.done",
						"index": event.BlockIndex,
						"item": wire.ResponseOutputItem{
							Type:   "message",
							Role:   "assistant",
							Status: "completed",
							Content: []wire.ResponseContentPart{{
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
				transporthttp.WriteSSEJSON(w, "response.output_text.done", map[string]any{
					"type": "response.output_text.done",
					"text": output.String(),
				})
				if reasoning.Len() > 0 {
					transporthttp.WriteSSEJSON(w, "response.reasoning.done", map[string]any{
						"type": "response.reasoning.done",
						"text": reasoning.String(),
					})
				}
				for _, key := range toolOrder {
					state := toolStates[key]
					transporthttp.WriteSSEJSON(w, "response.output_item.done", map[string]any{
						"type":  "response.output_item.done",
						"index": state.Index,
						"item": wire.ResponseOutputItem{
							Type:      "function_call",
							Name:      state.Name,
							CallID:    state.ID,
							Arguments: state.Input.String(),
							Status:    "completed",
						},
					})
				}
				transporthttp.WriteSSEJSON(w, "response.completed", wire.ResponsesResponse{
					Object:     "response",
					Model:      externalModel,
					Status:     "completed",
					Output:     responseOutputItems(streamResponseParts(output.String(), reasoning.String(), imageBlocks, imageOrder, toolStates, toolOrder)),
					OutputText: output.String(),
					Usage: &wire.ResponseUsage{
						InputTokens:  inputTokens,
						OutputTokens: outputTokens,
						TotalTokens:  inputTokens + outputTokens,
					},
				})
				flusher.Flush()
				return
			}
		case err, ok := <-stream.Err:
			if ok && err != nil && !shouldIgnoreStreamError(r, err) {
				slog.Error("stream failed", "protocol", "openai", "api_type", "responses", "err", err)
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

func responseOutputItems(parts []llm.ContentPart) []wire.ResponseOutputItem {
	if len(parts) == 0 {
		return nil
	}
	var items []wire.ResponseOutputItem
	var messageParts []wire.ResponseContentPart
	for _, part := range parts {
		switch part.Type {
		case llm.ContentTypeText:
			messageParts = append(messageParts, wire.ResponseContentPart{Type: "output_text", Text: part.Text})
		case llm.ContentTypeImage:
			messageParts = append(messageParts, wire.ResponseContentPart{Type: "input_image", ImageURL: part.URL})
		case llm.ContentTypeReasoning:
			items = append(items, wire.ResponseOutputItem{
				Type:   "reasoning",
				Status: "completed",
				Content: []wire.ResponseContentPart{{
					Type: "output_text",
					Text: part.Text,
				}},
			})
		case llm.ContentTypeToolCall:
			items = append(items, wire.ResponseOutputItem{
				Type:      "function_call",
				Name:      part.Name,
				CallID:    part.ToolCallID,
				Arguments: part.Input,
				Status:    "completed",
			})
		}
	}
	if len(messageParts) > 0 {
		items = append([]wire.ResponseOutputItem{{
			Type:    "message",
			Role:    "assistant",
			Status:  "completed",
			Content: messageParts,
		}}, items...)
	}
	return items
}

func llmContentToOpenAI(parts []llm.ContentPart) (any, []wire.ToolCall) {
	var content []map[string]any
	var toolCalls []wire.ToolCall
	for _, part := range parts {
		switch part.Type {
		case llm.ContentTypeText:
			content = append(content, map[string]any{"type": "text", "text": part.Text})
		case llm.ContentTypeImage:
			if part.URL != "" {
				content = append(content, map[string]any{"type": "image_url", "image_url": map[string]any{"url": part.URL}})
			}
		case llm.ContentTypeToolCall:
			toolCalls = append(toolCalls, wire.ToolCall{
				ID:   part.ToolCallID,
				Type: "function",
				Function: wire.FunctionCall{
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
