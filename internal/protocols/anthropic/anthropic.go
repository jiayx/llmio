package anthropic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	"github.com/jiayx/llmio/internal/llm"
	httpio "github.com/jiayx/llmio/internal/protocols/httpio"
	providerapi "github.com/jiayx/llmio/internal/providers/api"
)

// DecodeMessagesRequest decodes an Anthropic messages request body.
func DecodeMessagesRequest(body []byte) (MessagesRequest, error) {
	var req MessagesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return MessagesRequest{}, fmt.Errorf("invalid json: %w", err)
	}
	return req, nil
}

// WriteError writes an Anthropic-style error response.
func WriteError(w http.ResponseWriter, status int, typ, message string) {
	slog.Error("request failed",
		"protocol", "anthropic",
		"status", status,
		"error_type", typ,
		"message", message,
	)
	httpio.WriteJSON(w, status, map[string]any{
		"type": "error",
		"error": Error{
			Type:    typ,
			Message: message,
		},
	})
}

// WriteMessagesResponse writes one normalized response as Anthropic messages JSON.
func WriteMessagesResponse(w http.ResponseWriter, externalModel string, resp *llm.ChatResponse) {
	parts := resp.EffectiveOutput()
	out := MessagesResponse{
		ID:    resp.ID,
		Type:  "message",
		Role:  "assistant",
		Model: externalModel,
		Usage: Usage{
			InputTokens:  resp.InputTokens,
			OutputTokens: resp.OutputTokens,
		},
	}
	out.Content = llmContentToAnthropic(parts)
	out.StopReason = mapFinishReason(resp.FinishReason)
	httpio.WriteJSON(w, http.StatusOK, out)
}

// WritePassthroughResponse rewrites an upstream Anthropic response back to the external model name.
func WritePassthroughResponse(w http.ResponseWriter, resp *http.Response, externalModel string) {
	httpio.WritePassthroughResponse(w, resp, func(data []byte) ([]byte, error) {
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

	httpio.WriteSSEJSON(w, "message_start", StreamMessageStart{
		Type: "message_start",
		Message: MessagesResponse{
			Type:    "message",
			Role:    "assistant",
			Model:   externalModel,
			Content: []ContentBlock{},
			Usage:   Usage{},
		},
	})
	flusher.Flush()

	outputTokens := 0
	openBlocks := make(map[int]llm.ContentPart)
	nextBlockIndex := 0
	startBlock := func(index int, part llm.ContentPart) {
		openBlocks[index] = part
		block := anthropicContentBlockFromPart(part)
		httpio.WriteSSEJSON(w, "content_block_start", StreamContentBlockStart{
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
		httpio.WriteSSEJSON(w, "content_block_stop", map[string]any{"type": "content_block_stop", "index": index})
		delete(openBlocks, index)
	}

	for {
		select {
		case event, ok := <-stream.Events:
			if !ok {
				for index := range openBlocks {
					stopBlock(index)
				}
				httpio.WriteSSEJSON(w, "message_delta", StreamMessageDelta{
					Type: "message_delta",
					Delta: MessageDelta{
						StopReason: "end_turn",
					},
					Usage: StreamUsage{
						OutputTokens: outputTokens,
					},
				})
				httpio.WriteSSEJSON(w, "message_stop", map[string]string{"type": "message_stop"})
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
				httpio.WriteSSEJSON(w, "content_block_delta", StreamContentBlockDelta{
					Type:  "content_block_delta",
					Index: index,
					Delta: ContentTextDelta{
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
					httpio.WriteSSEJSON(w, "content_block_delta", StreamContentBlockDelta{
						Type:  "content_block_delta",
						Index: index,
						Delta: InputJSONDelta{
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
				httpio.WriteSSEJSON(w, "message_delta", StreamMessageDelta{
					Type: "message_delta",
					Delta: MessageDelta{
						StopReason: mapFinishReason(event.FinishReason),
					},
					Usage: StreamUsage{
						OutputTokens: outputTokens,
					},
				})
				httpio.WriteSSEJSON(w, "message_stop", map[string]string{"type": "message_stop"})
				flusher.Flush()
				return
			}
		case err, ok := <-stream.Err:
			if ok && err != nil {
				if shouldIgnoreStreamError(r, err) {
					logDownstreamDisconnected(r, "anthropic", "messages", err)
				} else {
					httpio.WriteSSEJSON(w, "error", map[string]any{
						"type": "error",
						"error": Error{
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

func logDownstreamDisconnected(r *http.Request, protocol, apiType string, err error) {
	slog.Info("downstream disconnected",
		"protocol", protocol,
		"api_type", apiType,
		"path", r.URL.Path,
		"remote", r.RemoteAddr,
		"err", err,
	)
}

func anthropicContentBlockFromPart(part llm.ContentPart) ContentBlock {
	blocks := llmContentToAnthropic([]llm.ContentPart{part})
	if len(blocks) == 0 {
		return ContentBlock{Type: "text", Text: ""}
	}
	return blocks[0]
}

func llmContentToAnthropic(parts []llm.ContentPart) []ContentBlock {
	out := make([]ContentBlock, 0, len(parts))
	for _, part := range parts {
		switch part.Type {
		case llm.ContentTypeText, llm.ContentTypeReasoning:
			out = append(out, ContentBlock{Type: "text", Text: part.Text})
		case llm.ContentTypeImage:
			if part.URL != "" {
				continue
			}
			out = append(out, ContentBlock{
				Type: "image",
				Source: &ImageSource{
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
			out = append(out, ContentBlock{
				Type:  "tool_use",
				ID:    part.ToolCallID,
				Name:  part.Name,
				Input: input,
			})
		case llm.ContentTypeToolResult:
			out = append(out, ContentBlock{
				Type:      "tool_result",
				ToolUseID: part.ToolCallID,
				Content: []ContentBlock{{
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
