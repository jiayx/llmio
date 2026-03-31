package openai

import "encoding/json"

type ChatCompletionRequest struct {
	Model         string                       `json:"model"`
	Messages      []Message                    `json:"messages"`
	Temperature   *float64                     `json:"temperature,omitempty"`
	TopP          *float64                     `json:"top_p,omitempty"`
	MaxTokens     *int                         `json:"max_tokens,omitempty"`
	Stream        bool                         `json:"stream,omitempty"`
	StreamOptions *ChatCompletionStreamOptions `json:"stream_options,omitempty"`
	User          string                       `json:"user,omitempty"`
	Tools         []Tool                       `json:"tools,omitempty"`
	ToolChoice    any                          `json:"tool_choice,omitempty"`
}

type ChatCompletionStreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

type Message struct {
	Role             string     `json:"role"`
	Content          any        `json:"content"`
	Name             string     `json:"name,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
}

type ChatCompletionResponse struct {
	ID      string           `json:"id"`
	Object  string           `json:"object"`
	Created int64            `json:"created"`
	Model   string           `json:"model"`
	Choices []Choice         `json:"choices"`
	Usage   *CompletionUsage `json:"usage,omitempty"`
}

type ErrorResponse struct {
	Error *CompletionError `json:"error"`
}

type Choice struct {
	Index        int             `json:"index"`
	Message      Message         `json:"message"`
	FinishReason string          `json:"finish_reason"`
	LogProbs     json.RawMessage `json:"logprobs,omitempty"`
}

type StreamChunk struct {
	ID      string           `json:"id"`
	Object  string           `json:"object"`
	Created int64            `json:"created"`
	Model   string           `json:"model"`
	Choices []StreamChoice   `json:"choices"`
	Usage   *CompletionUsage `json:"usage,omitempty"`
}

type StreamChoice struct {
	Index        int              `json:"index"`
	Delta        StreamDelta      `json:"delta"`
	FinishReason string           `json:"finish_reason,omitempty"`
	Usage        *CompletionUsage `json:"usage,omitempty"`
}

type StreamDelta struct {
	Role             string     `json:"role,omitempty"`
	Content          string     `json:"content,omitempty"`
	Reasoning        string     `json:"reasoning,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
}

type CompletionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type CompletionError struct {
	Message string `json:"message"`
	Type    string `json:"type,omitempty"`
	Param   string `json:"param,omitempty"`
	Code    any    `json:"code,omitempty"`
}

type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	OwnedBy string `json:"owned_by"`
}

type ResponsesRequest struct {
	Model           string   `json:"model"`
	Input           any      `json:"input"`
	Instructions    string   `json:"instructions,omitempty"`
	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"top_p,omitempty"`
	MaxOutputTokens *int     `json:"max_output_tokens,omitempty"`
	Stream          bool     `json:"stream,omitempty"`
	User            string   `json:"user,omitempty"`
	Tools           []Tool   `json:"tools,omitempty"`
	ToolChoice      any      `json:"tool_choice,omitempty"`
}

type ResponsesResponse struct {
	ID         string               `json:"id"`
	Object     string               `json:"object"`
	CreatedAt  int64                `json:"created_at,omitempty"`
	Model      string               `json:"model"`
	Status     string               `json:"status"`
	Output     []ResponseOutputItem `json:"output"`
	OutputText string               `json:"output_text,omitempty"`
	Usage      *ResponseUsage       `json:"usage,omitempty"`
}

type ResponseOutputItem struct {
	ID         string                `json:"id,omitempty"`
	Type       string                `json:"type"`
	Role       string                `json:"role,omitempty"`
	Name       string                `json:"name,omitempty"`
	CallID     string                `json:"call_id,omitempty"`
	Arguments  string                `json:"arguments,omitempty"`
	OutputData string                `json:"output,omitempty"`
	Status     string                `json:"status,omitempty"`
	Content    []ResponseContentPart `json:"content,omitempty"`
}

type ResponseContentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
	FileID   string `json:"file_id,omitempty"`
}

type ResponseUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type Tool struct {
	Type     string             `json:"type"`
	Function FunctionDefinition `json:"function"`
}

type FunctionDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type ToolCall struct {
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function FunctionCall `json:"function"`
	Index    int          `json:"index,omitempty"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments,omitempty"`
}
