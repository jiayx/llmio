package gateway

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	"github.com/jiayx/llmio/internal/core"
	anthropicproto "github.com/jiayx/llmio/internal/protocols/anthropic"
	openaiproto "github.com/jiayx/llmio/internal/protocols/openai"
)

func (s *Server) handleOpenAIStream(w http.ResponseWriter, r *http.Request, route modelRoute, req core.ChatRequest) {
	stream, err := s.dispatchChatStream(r.Context(), route, req)
	if err != nil {
		writeOpenAIError(w, http.StatusBadGateway, err.Error())
		return
	}
	defer closeStream(stream)

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeOpenAIError(w, http.StatusInternalServerError, "streaming is not supported by this server")
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
				writeSSE(w, "", "[DONE]")
				flusher.Flush()
				return
			}
			if event.Type == core.StreamEventDelta {
				chunk := openaiproto.StreamChunk{
					Object: "chat.completion.chunk",
					Model:  req.Model,
					Choices: []openaiproto.StreamChoice{{
						Index: 0,
						Delta: openaiproto.StreamDelta{},
					}},
				}
				if event.Part.Type == core.ContentTypeReasoning {
					chunk.Choices[0].Delta.Reasoning = event.TextDelta
				} else {
					chunk.Choices[0].Delta.Content = event.TextDelta
				}
				writeSSEJSON(w, "", chunk)
				flusher.Flush()
			}
			if event.Type == core.StreamEventTool {
				key, state := upsertStreamToolState(toolStates, &toolOrder, event)
				chunk := openaiproto.StreamChunk{
					Object: "chat.completion.chunk",
					Model:  req.Model,
					Choices: []openaiproto.StreamChoice{{
						Index: 0,
						Delta: openaiproto.StreamDelta{
							ToolCalls: []openaiproto.ToolCall{{
								Index: state.Index,
								ID:    state.ID,
								Type:  "function",
								Function: openaiproto.FunctionCall{
									Name:      state.Name,
									Arguments: event.ToolInput,
								},
							}},
						},
					}},
				}
				if key != "" {
					writeSSEJSON(w, "", chunk)
					flusher.Flush()
				}
			}
			if event.Type == core.StreamEventStop {
				chunk := openaiproto.StreamChunk{
					Object: "chat.completion.chunk",
					Model:  req.Model,
					Choices: []openaiproto.StreamChoice{{
						Index:        0,
						Delta:        openaiproto.StreamDelta{},
						FinishReason: event.FinishReason,
					}},
				}
				writeSSEJSON(w, "", chunk)
				writeSSE(w, "", "[DONE]")
				flusher.Flush()
				return
			}
		case err, ok := <-stream.Err:
			if ok && err != nil {
				if shouldIgnoreStreamError(r, err) {
				} else {
					slog.Error("stream failed",
						"protocol", "openai",
						"api_type", "chat_completions",
						"err", err,
					)
					writeSSE(w, "", "[DONE]")
					flusher.Flush()
				}
			}
			return
		}
	}
}

func (s *Server) handleAnthropicStream(w http.ResponseWriter, r *http.Request, externalModel string, route modelRoute, req core.ChatRequest) {
	stream, err := s.dispatchChatStream(r.Context(), route, req)
	if err != nil {
		writeAnthropicError(w, http.StatusBadGateway, "api_error", err.Error())
		return
	}
	defer closeStream(stream)

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeAnthropicError(w, http.StatusInternalServerError, "api_error", "streaming is not supported by this server")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	writeSSEJSON(w, "message_start", anthropicproto.StreamMessageStart{
		Type: "message_start",
		Message: anthropicproto.MessagesResponse{
			Type:    "message",
			Role:    "assistant",
			Model:   externalModel,
			Content: []anthropicproto.ContentBlock{},
			Usage:   anthropicproto.Usage{},
		},
	})
	flusher.Flush()

	outputTokens := 0
	openBlocks := make(map[int]core.ContentPart)
	nextBlockIndex := 0
	startBlock := func(index int, part core.ContentPart) {
		openBlocks[index] = part
		block := anthropicContentBlockFromPart(part)
		writeSSEJSON(w, "content_block_start", anthropicproto.StreamContentBlockStart{
			Type:         "content_block_start",
			Index:        index,
			ContentBlock: block,
		})
	}
	ensureBlock := func(index int, part core.ContentPart) int {
		if index < 0 {
			index = nextBlockIndex
			nextBlockIndex++
		}
		if _, ok := openBlocks[index]; !ok {
			startBlock(index, part)
		}
		return index
	}
	stopBlock := func(index int) {
		if _, ok := openBlocks[index]; !ok {
			return
		}
		writeSSEJSON(w, "content_block_stop", map[string]any{"type": "content_block_stop", "index": index})
		delete(openBlocks, index)
	}
	for {
		select {
		case event, ok := <-stream.Events:
			if !ok {
				for index := range openBlocks {
					stopBlock(index)
				}
				writeSSEJSON(w, "message_delta", anthropicproto.StreamMessageDelta{
					Type: "message_delta",
					Delta: anthropicproto.MessageDelta{
						StopReason: "end_turn",
					},
					Usage: anthropicproto.StreamUsage{
						OutputTokens: outputTokens,
					},
				})
				writeSSEJSON(w, "message_stop", map[string]string{"type": "message_stop"})
				flusher.Flush()
				return
			}
			switch event.Type {
			case core.StreamEventContentStart:
				index := event.BlockIndex
				if index < 0 {
					index = nextBlockIndex
					nextBlockIndex++
				}
				startBlock(index, event.Part)
				flusher.Flush()
			case core.StreamEventDelta:
				index := ensureBlock(event.BlockIndex, core.ContentPart{Type: core.ContentTypeText})
				writeSSEJSON(w, "content_block_delta", anthropicproto.StreamContentBlockDelta{
					Type:  "content_block_delta",
					Index: index,
					Delta: anthropicproto.ContentTextDelta{
						Type: "text_delta",
						Text: event.TextDelta,
					},
				})
				flusher.Flush()
			case core.StreamEventTool:
				index := ensureBlock(event.BlockIndex, core.ContentPart{
					Type:       core.ContentTypeToolCall,
					ToolCallID: event.ToolCallID,
					Name:       event.ToolName,
				})
				if event.ToolInput != "" {
					writeSSEJSON(w, "content_block_delta", anthropicproto.StreamContentBlockDelta{
						Type:  "content_block_delta",
						Index: index,
						Delta: anthropicproto.InputJSONDelta{
							Type:        "input_json_delta",
							PartialJSON: event.ToolInput,
						},
					})
				}
				flusher.Flush()
			case core.StreamEventContentStop:
				stopBlock(event.BlockIndex)
				flusher.Flush()
			case core.StreamEventUsage:
				outputTokens = event.OutputTokens
			case core.StreamEventStop:
				for index := range openBlocks {
					stopBlock(index)
				}
				writeSSEJSON(w, "message_delta", anthropicproto.StreamMessageDelta{
					Type: "message_delta",
					Delta: anthropicproto.MessageDelta{
						StopReason: mapFinishReason(event.FinishReason),
					},
					Usage: anthropicproto.StreamUsage{
						OutputTokens: outputTokens,
					},
				})
				writeSSEJSON(w, "message_stop", map[string]string{"type": "message_stop"})
				flusher.Flush()
				return
			}
		case err, ok := <-stream.Err:
			if ok && err != nil {
				if shouldIgnoreStreamError(r, err) {
				} else {
					writeSSEJSON(w, "error", map[string]any{
						"type": "error",
						"error": anthropicproto.Error{
							Type:    "api_error",
							Message: err.Error(),
						},
					})
					flusher.Flush()
				}
			}
			return
		}
	}
}

func (s *Server) handleOpenAIResponsesStream(w http.ResponseWriter, r *http.Request, externalModel string, route modelRoute, req core.ChatRequest) {
	stream, err := s.dispatchChatStream(r.Context(), route, req)
	if err != nil {
		writeOpenAIError(w, http.StatusBadGateway, err.Error())
		return
	}
	defer closeStream(stream)

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeOpenAIError(w, http.StatusInternalServerError, "streaming is not supported by this server")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	writeSSEJSON(w, "response.created", openaiproto.ResponsesResponse{
		Object: "response",
		Model:  externalModel,
		Status: "in_progress",
		Output: []openaiproto.ResponseOutputItem{},
	})
	flusher.Flush()

	var output strings.Builder
	var reasoning strings.Builder
	inputTokens := 0
	outputTokens := 0
	toolStates := make(map[string]*streamToolState)
	toolOrder := make([]string, 0)
	imageBlocks := make(map[int]core.ContentPart)
	imageOrder := make([]int, 0)

	for {
		select {
		case event, ok := <-stream.Events:
			if !ok {
				writeSSEJSON(w, "response.completed", openaiproto.ResponsesResponse{
					Object:     "response",
					Model:      externalModel,
					Status:     "completed",
					Output:     responseOutputItems(streamResponseParts(output.String(), reasoning.String(), imageBlocks, imageOrder, toolStates, toolOrder)),
					OutputText: output.String(),
					Usage: &openaiproto.ResponseUsage{
						InputTokens:  inputTokens,
						OutputTokens: outputTokens,
						TotalTokens:  inputTokens + outputTokens,
					},
				})
				flusher.Flush()
				return
			}
			switch event.Type {
			case core.StreamEventContentStart:
				if event.Part.Type == core.ContentTypeImage {
					imageBlocks[event.BlockIndex] = event.Part
					imageOrder = append(imageOrder, event.BlockIndex)
					writeSSEJSON(w, "response.output_item.added", map[string]any{
						"type":  "response.output_item.added",
						"index": event.BlockIndex,
						"item": openaiproto.ResponseOutputItem{
							Type:   "message",
							Role:   "assistant",
							Status: "in_progress",
							Content: []openaiproto.ResponseContentPart{{
								Type:     "input_image",
								ImageURL: event.Part.URL,
							}},
						},
					})
					flusher.Flush()
				}
			case core.StreamEventDelta:
				if event.Part.Type == core.ContentTypeReasoning {
					reasoning.WriteString(event.TextDelta)
					writeSSEJSON(w, "response.reasoning.delta", map[string]any{
						"type":  "response.reasoning.delta",
						"delta": event.TextDelta,
					})
				} else {
					output.WriteString(event.TextDelta)
					writeSSEJSON(w, "response.output_text.delta", map[string]any{
						"type":  "response.output_text.delta",
						"delta": event.TextDelta,
					})
				}
				flusher.Flush()
			case core.StreamEventTool:
				key, state := upsertStreamToolState(toolStates, &toolOrder, event)
				if !state.Emitted {
					writeSSEJSON(w, "response.output_item.added", map[string]any{
						"type":  "response.output_item.added",
						"item":  openaiproto.ResponseOutputItem{Type: "function_call", Name: state.Name, CallID: state.ID, Status: "in_progress"},
						"index": state.Index,
					})
					state.Emitted = true
				}
				writeSSEJSON(w, "response.function_call_arguments.delta", map[string]any{
					"type":         "response.function_call_arguments.delta",
					"call_id":      state.ID,
					"name":         state.Name,
					"output_index": state.Index,
					"delta":        event.ToolInput,
				})
				if key != "" {
					flusher.Flush()
				}
			case core.StreamEventContentStop:
				if event.Part.Type == core.ContentTypeImage {
					part := imageBlocks[event.BlockIndex]
					writeSSEJSON(w, "response.output_item.done", map[string]any{
						"type":  "response.output_item.done",
						"index": event.BlockIndex,
						"item": openaiproto.ResponseOutputItem{
							Type:   "message",
							Role:   "assistant",
							Status: "completed",
							Content: []openaiproto.ResponseContentPart{{
								Type:     "input_image",
								ImageURL: part.URL,
							}},
						},
					})
					flusher.Flush()
				}
			case core.StreamEventUsage:
				inputTokens = event.InputTokens
				outputTokens = event.OutputTokens
			case core.StreamEventStop:
				writeSSEJSON(w, "response.output_text.done", map[string]any{
					"type": "response.output_text.done",
					"text": output.String(),
				})
				if reasoning.Len() > 0 {
					writeSSEJSON(w, "response.reasoning.done", map[string]any{
						"type": "response.reasoning.done",
						"text": reasoning.String(),
					})
				}
				for _, key := range toolOrder {
					state := toolStates[key]
					writeSSEJSON(w, "response.output_item.done", map[string]any{
						"type":  "response.output_item.done",
						"index": state.Index,
						"item": openaiproto.ResponseOutputItem{
							Type:      "function_call",
							Name:      state.Name,
							CallID:    state.ID,
							Arguments: state.Input.String(),
							Status:    "completed",
						},
					})
				}
				writeSSEJSON(w, "response.completed", openaiproto.ResponsesResponse{
					Object:     "response",
					Model:      externalModel,
					Status:     "completed",
					Output:     responseOutputItems(streamResponseParts(output.String(), reasoning.String(), imageBlocks, imageOrder, toolStates, toolOrder)),
					OutputText: output.String(),
					Usage: &openaiproto.ResponseUsage{
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
				} else {
					slog.Error("stream failed",
						"protocol", "openai",
						"api_type", "responses",
						"err", err,
					)
				}
			}
			return
		}
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

func upsertStreamToolState(states map[string]*streamToolState, order *[]string, event core.StreamEvent) (string, *streamToolState) {
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
		state = &streamToolState{
			ID:    event.ToolCallID,
			Name:  event.ToolName,
			Index: index,
		}
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

func streamToolParts(states map[string]*streamToolState, order []string) []core.ContentPart {
	out := make([]core.ContentPart, 0, len(order))
	for _, key := range order {
		state := states[key]
		out = append(out, core.ContentPart{
			Type:       core.ContentTypeToolCall,
			Name:       state.Name,
			ToolCallID: state.ID,
			Input:      state.Input.String(),
		})
	}
	return out
}

func streamResponseParts(text, reasoning string, imageBlocks map[int]core.ContentPart, imageOrder []int, toolStates map[string]*streamToolState, toolOrder []string) []core.ContentPart {
	parts := make([]core.ContentPart, 0, 1+len(imageOrder)+len(toolOrder)+1)
	parts = append(parts, textParts(text)...)
	if reasoning != "" {
		parts = append(parts, core.ContentPart{Type: core.ContentTypeReasoning, Text: reasoning})
	}
	for _, index := range imageOrder {
		if part, ok := imageBlocks[index]; ok {
			parts = append(parts, part)
		}
	}
	parts = append(parts, streamToolParts(toolStates, toolOrder)...)
	return parts
}
