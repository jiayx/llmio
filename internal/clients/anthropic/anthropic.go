package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	clientmeta "github.com/jiayx/llmio/internal/clients"
	"github.com/jiayx/llmio/internal/llm"
	providerapi "github.com/jiayx/llmio/internal/providers/api"
	transporthttp "github.com/jiayx/llmio/internal/transport/http"
	wire "github.com/jiayx/llmio/internal/wire/anthropic"
)

// DecodeMessagesRequest decodes an Anthropic messages request body.
func DecodeMessagesRequest(body []byte) (wire.MessagesRequest, error) {
	var req wire.MessagesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return wire.MessagesRequest{}, fmt.Errorf("invalid json: %w", err)
	}
	return req, nil
}

// NewMessagesRequestMeta describes an inbound Anthropic messages request.
func NewMessagesRequestMeta(req wire.MessagesRequest, body []byte, headers http.Header) clientmeta.RequestMeta {
	return clientmeta.RequestMeta{
		Protocol:      clientmeta.ProtocolAnthropic,
		APIType:       clientmeta.APIMessages,
		Path:          "/messages",
		ExternalModel: req.Model,
		Body:          body,
		Headers:       headers,
		Stream:        req.Stream,
	}
}

// WriteError writes an Anthropic-style error response.
func WriteError(w http.ResponseWriter, status int, typ, message string) {
	slog.Error("request failed",
		"protocol", "anthropic",
		"status", status,
		"error_type", typ,
		"message", message,
	)
	transporthttp.WriteJSON(w, status, map[string]any{
		"type": "error",
		"error": wire.Error{
			Type:    typ,
			Message: message,
		},
	})
}

// WriteMessagesResponse writes one normalized response as Anthropic messages JSON.
func WriteMessagesResponse(w http.ResponseWriter, externalModel string, resp *llm.ChatResponse) {
	parts := resp.EffectiveOutput()
	out := wire.MessagesResponse{
		ID:    resp.ID,
		Type:  "message",
		Role:  "assistant",
		Model: externalModel,
		Usage: wire.Usage{
			InputTokens:  resp.InputTokens,
			OutputTokens: resp.OutputTokens,
		},
	}
	out.Content = llmContentToAnthropic(parts)
	out.StopReason = mapFinishReason(resp.FinishReason)
	transporthttp.WriteJSON(w, http.StatusOK, out)
}

// WritePassthroughResponse rewrites an upstream Anthropic response back to the external model name.
func WritePassthroughResponse(w http.ResponseWriter, resp *http.Response, externalModel string) {
	transporthttp.WritePassthroughResponse(w, resp, func(data []byte) ([]byte, error) {
		return rewritePassthroughJSONPayload(data, externalModel)
	})
}

// ServeMessagesStream renders normalized events as Anthropic message SSE.
func ServeMessagesStream(w http.ResponseWriter, r *http.Request, externalModel string, stream *providerapi.StreamReader) {
	defer closeStream(stream)

	flusher, ok := w.(http.Flusher)
	if !ok {
		WriteError(w, http.StatusInternalServerError, "api_error", "streaming is not supported by this server")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	transporthttp.WriteSSEJSON(w, "message_start", wire.StreamMessageStart{
		Type: "message_start",
		Message: wire.MessagesResponse{
			Type:    "message",
			Role:    "assistant",
			Model:   externalModel,
			Content: []wire.ContentBlock{},
			Usage:   wire.Usage{},
		},
	})
	flusher.Flush()

	outputTokens := 0
	openBlocks := make(map[int]llm.ContentPart)
	nextBlockIndex := 0
	startBlock := func(index int, part llm.ContentPart) {
		openBlocks[index] = part
		block := anthropicContentBlockFromPart(part)
		transporthttp.WriteSSEJSON(w, "content_block_start", wire.StreamContentBlockStart{
			Type:         "content_block_start",
			Index:        index,
			ContentBlock: block,
		})
	}
	ensureBlock := func(index int, part llm.ContentPart) int {
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
		transporthttp.WriteSSEJSON(w, "content_block_stop", map[string]any{"type": "content_block_stop", "index": index})
		delete(openBlocks, index)
	}

	for {
		select {
		case event, ok := <-stream.Events:
			if !ok {
				for index := range openBlocks {
					stopBlock(index)
				}
				transporthttp.WriteSSEJSON(w, "message_delta", wire.StreamMessageDelta{
					Type: "message_delta",
					Delta: wire.MessageDelta{
						StopReason: "end_turn",
					},
					Usage: wire.StreamUsage{
						OutputTokens: outputTokens,
					},
				})
				transporthttp.WriteSSEJSON(w, "message_stop", map[string]string{"type": "message_stop"})
				flusher.Flush()
				return
			}

			switch event.Type {
			case llm.StreamEventContentStart:
				index := event.BlockIndex
				if index < 0 {
					index = nextBlockIndex
					nextBlockIndex++
				}
				startBlock(index, event.Part)
				flusher.Flush()
			case llm.StreamEventDelta:
				index := ensureBlock(event.BlockIndex, llm.ContentPart{Type: llm.ContentTypeText})
				transporthttp.WriteSSEJSON(w, "content_block_delta", wire.StreamContentBlockDelta{
					Type:  "content_block_delta",
					Index: index,
					Delta: wire.ContentTextDelta{
						Type: "text_delta",
						Text: event.TextDelta,
					},
				})
				flusher.Flush()
			case llm.StreamEventTool:
				index := ensureBlock(event.BlockIndex, llm.ContentPart{
					Type:       llm.ContentTypeToolCall,
					ToolCallID: event.ToolCallID,
					Name:       event.ToolName,
				})
				if event.ToolInput != "" {
					transporthttp.WriteSSEJSON(w, "content_block_delta", wire.StreamContentBlockDelta{
						Type:  "content_block_delta",
						Index: index,
						Delta: wire.InputJSONDelta{
							Type:        "input_json_delta",
							PartialJSON: event.ToolInput,
						},
					})
				}
				flusher.Flush()
			case llm.StreamEventContentStop:
				stopBlock(event.BlockIndex)
				flusher.Flush()
			case llm.StreamEventUsage:
				outputTokens = event.OutputTokens
			case llm.StreamEventStop:
				for index := range openBlocks {
					stopBlock(index)
				}
				transporthttp.WriteSSEJSON(w, "message_delta", wire.StreamMessageDelta{
					Type: "message_delta",
					Delta: wire.MessageDelta{
						StopReason: mapFinishReason(event.FinishReason),
					},
					Usage: wire.StreamUsage{
						OutputTokens: outputTokens,
					},
				})
				transporthttp.WriteSSEJSON(w, "message_stop", map[string]string{"type": "message_stop"})
				flusher.Flush()
				return
			}
		case err, ok := <-stream.Err:
			if ok && err != nil && !shouldIgnoreStreamError(r, err) {
				transporthttp.WriteSSEJSON(w, "error", map[string]any{
					"type": "error",
					"error": wire.Error{
						Type:    "api_error",
						Message: err.Error(),
					},
				})
				flusher.Flush()
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
	if nested, ok := payload["message"].(map[string]any); ok {
		rewriteMapModel(nested, externalModel)
	}
	return json.Marshal(payload)
}

func rewriteMapModel(m map[string]any, model string) {
	if _, ok := m["model"]; ok {
		m["model"] = model
	}
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

func anthropicContentBlockFromPart(part llm.ContentPart) wire.ContentBlock {
	blocks := llmContentToAnthropic([]llm.ContentPart{part})
	if len(blocks) == 0 {
		return wire.ContentBlock{Type: "text", Text: ""}
	}
	return blocks[0]
}

func llmContentToAnthropic(parts []llm.ContentPart) []wire.ContentBlock {
	out := make([]wire.ContentBlock, 0, len(parts))
	for _, part := range parts {
		switch part.Type {
		case llm.ContentTypeText, llm.ContentTypeReasoning:
			out = append(out, wire.ContentBlock{Type: "text", Text: part.Text})
		case llm.ContentTypeImage:
			if part.URL != "" {
				continue
			}
			out = append(out, wire.ContentBlock{
				Type: "image",
				Source: &wire.ImageSource{
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
			out = append(out, wire.ContentBlock{
				Type:  "tool_use",
				ID:    part.ToolCallID,
				Name:  part.Name,
				Input: input,
			})
		case llm.ContentTypeToolResult:
			out = append(out, wire.ContentBlock{
				Type:      "tool_result",
				ToolUseID: part.ToolCallID,
				Content: []wire.ContentBlock{{
					Type: "text",
					Text: part.Output,
				}},
				IsError: part.IsError,
			})
		}
	}
	return out
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
