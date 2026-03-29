package core

// ChatRequest is the normalized request shape shared across providers.
type ChatRequest struct {
	Model       string
	Messages    []Message
	System      []ContentPart
	MaxTokens   int
	Temperature *float64
	TopP        *float64
	Stream      bool
	User        string
	Tools       []ToolDefinition
	ToolChoice  *ToolChoice
}

// Message is a single chat message in normalized form.
type Message struct {
	Role    string
	Content []ContentPart
	Name    string
}

// ChatResponse is the normalized non-streaming provider response.
type ChatResponse struct {
	ID           string
	Model        string
	Output       []ContentPart
	OutputText   string
	FinishReason string
	InputTokens  int
	OutputTokens int
	Raw          []byte
}

// StreamEvent is a normalized streaming event emitted by providers.
type StreamEvent struct {
	Type         string
	BlockIndex   int
	Part         ContentPart
	TextDelta    string
	ToolIndex    int
	ToolCallID   string
	ToolName     string
	ToolInput    string
	FinishReason string
	InputTokens  int
	OutputTokens int
	Raw          []byte
}

// ContentPart is a typed segment of message or response content.
type ContentPart struct {
	Type       string
	Text       string
	MediaType  string
	Data       string
	URL        string
	Name       string
	ToolCallID string
	Input      string
	Output     string
	IsError    bool
}

// ToolDefinition describes a callable tool exposed to a model.
type ToolDefinition struct {
	Name        string
	Description string
	InputSchema string
}

// ToolChoice constrains how a model may choose tools.
type ToolChoice struct {
	Type string
	Name string
}

// Stream event types emitted through StreamEvent.Type.
const (
	StreamEventStart        = "start"
	StreamEventContentStart = "content_start"
	StreamEventDelta        = "delta"
	StreamEventContentStop  = "content_stop"
	StreamEventStop         = "stop"
	StreamEventUsage        = "usage"
	StreamEventTool         = "tool"

	// Content part types used through ContentPart.Type.
	ContentTypeText       = "text"
	ContentTypeImage      = "image"
	ContentTypeReasoning  = "reasoning"
	ContentTypeToolCall   = "tool_call"
	ContentTypeToolResult = "tool_result"
)
