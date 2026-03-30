package policy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/jiayx/llmio/internal/apikeys"
	"github.com/jiayx/llmio/internal/debugtrace"
	"github.com/jiayx/llmio/internal/llm"
	protocols "github.com/jiayx/llmio/internal/protocols"
	protocolanthropic "github.com/jiayx/llmio/internal/protocols/anthropic"
	protocolopenai "github.com/jiayx/llmio/internal/protocols/openai"
	providerapi "github.com/jiayx/llmio/internal/providers/api"
	"github.com/jiayx/llmio/internal/routing"
	"github.com/jiayx/llmio/internal/usage"
)

// Config controls execution policy around provider attempts.
type Config struct {
	RequestTimeout       time.Duration
	StreamSetupTimeout   time.Duration
	BreakerFailureThresh int
	BreakerOpenFor       time.Duration
}

// DefaultConfig returns conservative execution defaults for the gateway.
func DefaultConfig() Config {
	return Config{
		RequestTimeout:       90 * time.Second,
		StreamSetupTimeout:   30 * time.Second,
		BreakerFailureThresh: 3,
		BreakerOpenFor:       30 * time.Second,
	}
}

type breakerState struct {
	consecutiveFailures int
	openUntil           time.Time
}

// Policy executes provider attempts with timeout, retry, and passive circuit breaking.
type Policy struct {
	cfg       Config
	providers map[string]providerapi.ProviderAdapter
	recorder  usage.Recorder

	mu       sync.Mutex
	breakers map[string]*breakerState
}

// New constructs an execution policy.
func New(cfg Config, adapters map[string]providerapi.ProviderAdapter) *Policy {
	return NewWithRecorder(cfg, adapters, usage.NopRecorder{})
}

// NewWithRecorder constructs an execution policy with an explicit usage recorder.
func NewWithRecorder(cfg Config, adapters map[string]providerapi.ProviderAdapter, recorder usage.Recorder) *Policy {
	if cfg.RequestTimeout <= 0 || cfg.StreamSetupTimeout <= 0 || cfg.BreakerFailureThresh <= 0 || cfg.BreakerOpenFor <= 0 {
		defaults := DefaultConfig()
		if cfg.RequestTimeout <= 0 {
			cfg.RequestTimeout = defaults.RequestTimeout
		}
		if cfg.StreamSetupTimeout <= 0 {
			cfg.StreamSetupTimeout = defaults.StreamSetupTimeout
		}
		if cfg.BreakerFailureThresh <= 0 {
			cfg.BreakerFailureThresh = defaults.BreakerFailureThresh
		}
		if cfg.BreakerOpenFor <= 0 {
			cfg.BreakerOpenFor = defaults.BreakerOpenFor
		}
	}

	return &Policy{
		cfg:       cfg,
		providers: adapters,
		recorder:  recorder,
		breakers:  make(map[string]*breakerState, len(adapters)),
	}
}

// ExecuteChat runs a normalized non-streaming provider request.
func (p *Policy) ExecuteChat(ctx context.Context, route routing.Route, req llm.ChatRequest) (*llm.ChatResponse, error) {
	var lastErr error
	for _, target := range route.Targets {
		if !p.allow(target.ProviderName) {
			lastErr = fmt.Errorf("provider %s: circuit open", target.ProviderName)
			continue
		}

		providerReq := sanitizeChatRequest(req, target)
		providerReq.Model = target.BackendModel
		attemptCtx, cancel := context.WithTimeout(ctx, p.cfg.RequestTimeout)
		resp, err := p.providers[target.ProviderName].Chat(attemptCtx, providerReq)
		cancel()
		if err == nil {
			p.markSuccess(target.ProviderName)
			p.recordUsage(ctx, target, req.Model, "", "", false, false, true, resp.InputTokens, resp.OutputTokens)
			return resp, nil
		}

		p.markFailure(target.ProviderName)
		lastErr = fmt.Errorf("provider %s: %w", target.ProviderName, err)
		if !IsRetryableError(err) {
			return nil, lastErr
		}
	}

	if lastErr == nil {
		lastErr = errors.New("no provider available")
	}
	return nil, lastErr
}

// ExecuteStream runs a normalized streaming provider request until a stream is established.
func (p *Policy) ExecuteStream(ctx context.Context, route routing.Route, req llm.ChatRequest) (*providerapi.StreamReader, error) {
	var lastErr error
	for _, target := range route.Targets {
		if !p.allow(target.ProviderName) {
			lastErr = fmt.Errorf("provider %s: circuit open", target.ProviderName)
			continue
		}

		providerReq := sanitizeChatRequest(req, target)
		providerReq.Model = target.BackendModel
		attemptCtx, cancel := context.WithCancel(ctx)
		type streamResult struct {
			stream *providerapi.StreamReader
			err    error
		}
		resultCh := make(chan streamResult, 1)
		go func() {
			stream, err := p.providers[target.ProviderName].ChatStream(attemptCtx, providerReq)
			resultCh <- streamResult{stream: stream, err: err}
		}()

		var (
			stream *providerapi.StreamReader
			err    error
		)
		select {
		case <-ctx.Done():
			cancel()
			return nil, ctx.Err()
		case <-time.After(p.cfg.StreamSetupTimeout):
			cancel()
			err = context.DeadlineExceeded
		case result := <-resultCh:
			stream = result.stream
			err = result.err
		}
		if err == nil {
			p.markSuccess(target.ProviderName)
			return p.wrapStreamUsage(ctx, target, req.Model, wrapStreamClose(stream, cancel)), nil
		}
		cancel()

		p.markFailure(target.ProviderName)
		lastErr = fmt.Errorf("provider %s: %w", target.ProviderName, err)
		if !IsRetryableError(err) {
			return nil, lastErr
		}
	}

	if lastErr == nil {
		lastErr = errors.New("no provider available")
	}
	return nil, lastErr
}

func wrapStreamClose(stream *providerapi.StreamReader, onClose func()) *providerapi.StreamReader {
	if stream == nil {
		return nil
	}
	closeFn := stream.Close
	return &providerapi.StreamReader{
		Events: stream.Events,
		Err:    stream.Err,
		Close: func() error {
			onClose()
			if closeFn != nil {
				return closeFn()
			}
			return nil
		},
	}
}

// ExecutePassthrough attempts native forwarding and falls back when only retryable failures occur.
func (p *Policy) ExecutePassthrough(ctx context.Context, route routing.Route, meta protocols.RequestMeta) (*http.Response, bool, error) {
	for _, target := range route.Targets {
		provider := p.providers[target.ProviderName]

		supporter, ok := provider.(providerapi.PassthroughSupporter)
		if !ok || !supporter.SupportsPassthrough(meta.Protocol, meta.APIType) {
			continue
		}
		forwarder, ok := provider.(providerapi.PassthroughForwarder)
		if !ok {
			continue
		}
		if !p.allow(target.ProviderName) {
			continue
		}

		payload, err := rewriteRequestForTarget(meta.Body, target)
		if err != nil {
			return nil, true, err
		}

		attemptCtx, cancel := context.WithTimeout(ctx, p.timeoutFor(meta.Stream))
		resp, err := forwarder.Forward(attemptCtx, meta.Protocol, meta.UpstreamPath, payload, meta.Headers)
		if err != nil {
			cancel()
			p.markFailure(target.ProviderName)
			if IsRetryableError(err) {
				continue
			}
			return nil, true, fmt.Errorf("provider %s: %w", target.ProviderName, err)
		}
		if IsUnsupportedPassthroughStatus(resp.StatusCode) {
			cancel()
			p.markFailure(target.ProviderName)
			closeResponseBody("passthrough unsupported", resp.Body)
			continue
		}
		if IsRetryableStatus(resp.StatusCode) {
			cancel()
			p.markFailure(target.ProviderName)
			closeResponseBody("passthrough retry", resp.Body)
			continue
		}

		resp.Body = newCancelOnCloseBody(resp.Body, cancel)
		p.markSuccess(target.ProviderName)
		p.attachPassthroughUsage(ctx, target, meta, resp)
		return resp, true, nil
	}

	return nil, false, nil
}

func (p *Policy) wrapStreamUsage(ctx context.Context, target routing.Target, externalModel string, stream *providerapi.StreamReader) *providerapi.StreamReader {
	events := make(chan llm.StreamEvent, 16)
	errs := make(chan error, 1)

	go func() {
		defer close(events)

		var (
			srcEvents  = stream.Events
			srcErrs    = stream.Err
			input      int
			output     int
			usageKnown bool
		)

		for srcEvents != nil || srcErrs != nil {
			select {
			case event, ok := <-srcEvents:
				if !ok {
					srcEvents = nil
					continue
				}
				if event.Type == llm.StreamEventUsage {
					input = event.InputTokens
					output = event.OutputTokens
					usageKnown = true
					if debugtrace.Enabled() {
						slog.Debug("usage stream event trace",
							"provider", target.ProviderName,
							"external_model", externalModel,
							"input_tokens", input,
							"output_tokens", output,
						)
					}
				}
				events <- event
			case err, ok := <-srcErrs:
				if !ok {
					srcErrs = nil
					continue
				}
				if err != nil {
					errs <- err
					return
				}
			}
		}

		p.recordUsage(ctx, target, externalModel, "", "", true, false, usageKnown, input, output)
	}()

	return &providerapi.StreamReader{
		Events: events,
		Err:    errs,
		Close:  stream.Close,
	}
}

func (p *Policy) attachPassthroughUsage(ctx context.Context, target routing.Target, meta protocols.RequestMeta, resp *http.Response) {
	if resp == nil || resp.Body == nil {
		return
	}

	contentType := strings.ToLower(resp.Header.Get("Content-Type"))
	if strings.Contains(contentType, "text/event-stream") {
		resp.Body = newPassthroughUsageBody(resp.Body, func(event usage.Event) {
			p.recordUsageEvent(ctx, event)
		}, target, meta)
		return
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		resp.Body = io.NopCloser(bytes.NewReader(nil))
		return
	}
	_ = resp.Body.Close()
	resp.Body = io.NopCloser(bytes.NewReader(data))

	input, output, known := extractPassthroughUsage(meta, data)
	if debugtrace.Enabled() {
		slog.Debug("usage passthrough body trace",
			"provider", target.ProviderName,
			"protocol", meta.Protocol,
			"api_type", meta.APIType,
			"body", debugtrace.Bytes(data),
			"input_tokens", input,
			"output_tokens", output,
			"usage_known", known,
		)
	}
	p.recordUsage(ctx, target, meta.ExternalModel, meta.Protocol, meta.APIType, meta.Stream, true, known, input, output)
}

func extractPassthroughUsage(meta protocols.RequestMeta, data []byte) (int, int, bool) {
	switch meta.Protocol {
	case protocols.ProtocolOpenAI:
		switch meta.APIType {
		case protocols.APIChatCompletions:
			var resp protocolopenai.ChatCompletionResponse
			if err := json.Unmarshal(data, &resp); err != nil || resp.Usage == nil {
				return 0, 0, false
			}
			out := protocolopenai.ChatCompletionResponseToLLM(resp, data)
			return out.InputTokens, out.OutputTokens, true
		case protocols.APIResponses:
			var resp protocolopenai.ResponsesResponse
			if err := json.Unmarshal(data, &resp); err != nil || resp.Usage == nil {
				return 0, 0, false
			}
			out := protocolopenai.ResponsesResponseToLLM(resp, data)
			return out.InputTokens, out.OutputTokens, true
		}
	case protocols.ProtocolAnthropic:
		if meta.APIType != protocols.APIMessages {
			return 0, 0, false
		}
		var resp protocolanthropic.MessagesResponse
		if err := json.Unmarshal(data, &resp); err != nil {
			return 0, 0, false
		}
		out := protocolanthropic.MessagesResponseToLLM(resp, data)
		return out.InputTokens, out.OutputTokens, true
	}
	return 0, 0, false
}

func (p *Policy) recordUsage(ctx context.Context, target routing.Target, externalModel, protocol, apiType string, stream, passthrough, usageKnown bool, inputTokens, outputTokens int) {
	var (
		apiKeyID   string
		apiKeyName string
	)
	if principal, ok := apikeys.PrincipalFromContext(ctx); ok {
		apiKeyID = principal.ID
		apiKeyName = principal.Name
	}
	p.recordUsageEvent(ctx, usage.Event{
		Timestamp:     time.Now().UTC(),
		APIKeyID:      apiKeyID,
		APIKeyName:    apiKeyName,
		ProviderName:  target.ProviderName,
		ExternalModel: externalModel,
		BackendModel:  target.BackendModel,
		Protocol:      protocol,
		APIType:       apiType,
		InputTokens:   inputTokens,
		OutputTokens:  outputTokens,
		TotalTokens:   inputTokens + outputTokens,
		Stream:        stream,
		Passthrough:   passthrough,
		UsageKnown:    usageKnown,
	})
}

func (p *Policy) recordUsageEvent(ctx context.Context, event usage.Event) {
	if debugtrace.Enabled() {
		slog.Debug("usage record trace",
			"api_key_id", event.APIKeyID,
			"api_key_name", event.APIKeyName,
			"provider", event.ProviderName,
			"external_model", event.ExternalModel,
			"backend_model", event.BackendModel,
			"protocol", event.Protocol,
			"api_type", event.APIType,
			"stream", event.Stream,
			"passthrough", event.Passthrough,
			"usage_known", event.UsageKnown,
			"input_tokens", event.InputTokens,
			"output_tokens", event.OutputTokens,
		)
	}
	recorder := p.recorder
	if recorder == nil {
		recorder = usage.NopRecorder{}
	}
	recorder.Record(ctx, event)
}

type passthroughUsageBody struct {
	body         io.ReadCloser
	record       func(usage.Event)
	target       routing.Target
	meta         protocols.RequestMeta
	lineBuf      []byte
	eventName    string
	blockState   map[int]protocolanthropic.ContentBlock
	inputTokens  int
	outputTokens int
	usageKnown   bool
	recorded     bool
	failed       bool
}

type cancelOnCloseBody struct {
	body   io.ReadCloser
	cancel context.CancelFunc
	once   sync.Once
}

func newPassthroughUsageBody(body io.ReadCloser, record func(usage.Event), target routing.Target, meta protocols.RequestMeta) io.ReadCloser {
	return &passthroughUsageBody{
		body:       body,
		record:     record,
		target:     target,
		meta:       meta,
		blockState: make(map[int]protocolanthropic.ContentBlock),
	}
}

func newCancelOnCloseBody(body io.ReadCloser, cancel context.CancelFunc) io.ReadCloser {
	if cancel == nil {
		return body
	}
	if body == nil {
		cancel()
		return nil
	}
	return &cancelOnCloseBody{
		body:   body,
		cancel: cancel,
	}
}

func (b *cancelOnCloseBody) Read(p []byte) (int, error) {
	return b.body.Read(p)
}

func (b *cancelOnCloseBody) Close() error {
	var err error
	b.once.Do(func() {
		err = b.body.Close()
		b.cancel()
	})
	return err
}

func (b *passthroughUsageBody) Read(p []byte) (int, error) {
	n, err := b.body.Read(p)
	if n > 0 {
		b.consume(p[:n])
	}
	if err != nil {
		if !errors.Is(err, io.EOF) {
			b.failed = true
		}
		if errors.Is(err, io.EOF) {
			b.finish()
		}
	}
	return n, err
}

func (b *passthroughUsageBody) Close() error {
	b.finish()
	return b.body.Close()
}

func (b *passthroughUsageBody) consume(chunk []byte) {
	b.lineBuf = append(b.lineBuf, chunk...)
	for {
		idx := bytes.IndexByte(b.lineBuf, '\n')
		if idx < 0 {
			return
		}
		line := string(bytes.TrimRight(b.lineBuf[:idx], "\r"))
		b.lineBuf = b.lineBuf[idx+1:]
		b.consumeLine(line)
	}
}

func (b *passthroughUsageBody) consumeLine(line string) {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" {
		b.eventName = ""
		return
	}
	if strings.HasPrefix(trimmed, "event:") {
		b.eventName = strings.TrimSpace(strings.TrimPrefix(trimmed, "event:"))
		return
	}
	if !strings.HasPrefix(trimmed, "data:") {
		return
	}

	payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
	if payload == "" || payload == "[DONE]" {
		return
	}

	switch b.meta.Protocol {
	case protocols.ProtocolOpenAI:
		switch b.meta.APIType {
		case protocols.APIChatCompletions:
			events, err := protocolopenai.StreamChunkPayloadToLLMEvents(payload)
			if err == nil {
				if debugtrace.Enabled() {
					slog.Debug("usage passthrough openai stream trace",
						"api_type", b.meta.APIType,
						"payload", debugtrace.String(payload),
					)
				}
				b.observeEvents(events)
			}
		case protocols.APIResponses:
			if event, ok := openAIResponsesUsageEvent(payload); ok {
				if debugtrace.Enabled() {
					slog.Debug("usage passthrough responses stream trace",
						"payload", debugtrace.String(payload),
						"input_tokens", event.InputTokens,
						"output_tokens", event.OutputTokens,
					)
				}
				b.observeEvents([]llm.StreamEvent{event})
			}
		}
	case protocols.ProtocolAnthropic:
		out := make(chan llm.StreamEvent, 8)
		if err := protocolanthropic.SSEEventToLLMEvents(b.eventName, payload, b.blockState, out); err == nil {
			close(out)
			events := make([]llm.StreamEvent, 0, len(out))
			for event := range out {
				events = append(events, event)
			}
			b.observeEvents(events)
		}
	}
}

func openAIResponsesUsageEvent(payload string) (llm.StreamEvent, bool) {
	var resp protocolopenai.ResponsesResponse
	if err := json.Unmarshal([]byte(payload), &resp); err != nil || resp.Usage == nil {
		return llm.StreamEvent{}, false
	}
	out := protocolopenai.ResponsesResponseToLLM(resp, []byte(payload))
	return llm.StreamEvent{
		Type:         llm.StreamEventUsage,
		InputTokens:  out.InputTokens,
		OutputTokens: out.OutputTokens,
	}, true
}

func (b *passthroughUsageBody) observeEvents(events []llm.StreamEvent) {
	for _, event := range events {
		if event.Type != llm.StreamEventUsage {
			continue
		}
		b.inputTokens = event.InputTokens
		b.outputTokens = event.OutputTokens
		b.usageKnown = true
	}
}

func (b *passthroughUsageBody) finish() {
	if b.recorded || b.failed {
		return
	}
	b.recorded = true
	b.record(usage.Event{
		Timestamp:     time.Now().UTC(),
		APIKeyID:      b.meta.APIKeyID,
		APIKeyName:    b.meta.APIKeyName,
		ProviderName:  b.target.ProviderName,
		ExternalModel: b.meta.ExternalModel,
		BackendModel:  b.target.BackendModel,
		Protocol:      b.meta.Protocol,
		APIType:       b.meta.APIType,
		InputTokens:   b.inputTokens,
		OutputTokens:  b.outputTokens,
		TotalTokens:   b.inputTokens + b.outputTokens,
		Stream:        true,
		Passthrough:   true,
		UsageKnown:    b.usageKnown,
	})
}

// IsRetryableError reports whether the error should fall through to the next target.
func IsRetryableError(err error) bool {
	if err == nil {
		return false
	}

	msg := err.Error()
	for _, code := range []string{" status 429", " status 500", " status 502", " status 503", " status 504"} {
		if strings.Contains(msg, code) {
			return true
		}
	}
	return errors.Is(err, context.DeadlineExceeded)
}

// IsRetryableStatus reports whether an upstream status should fall through to the next target.
func IsRetryableStatus(status int) bool {
	switch status {
	case http.StatusTooManyRequests, http.StatusInternalServerError, http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout:
		return true
	default:
		return false
	}
}

func IsUnsupportedPassthroughStatus(status int) bool {
	switch status {
	case http.StatusNotFound, http.StatusMethodNotAllowed, http.StatusNotImplemented:
		return true
	default:
		return false
	}
}

func (p *Policy) timeoutFor(stream bool) time.Duration {
	if stream {
		return p.cfg.StreamSetupTimeout
	}
	return p.cfg.RequestTimeout
}

func (p *Policy) allow(provider string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()

	state := p.breaker(provider)
	if state.openUntil.IsZero() || time.Now().After(state.openUntil) {
		state.openUntil = time.Time{}
		return true
	}

	slog.Warn("provider circuit open", "provider", provider, "open_until", state.openUntil)
	return false
}

func (p *Policy) markSuccess(provider string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	state := p.breaker(provider)
	state.consecutiveFailures = 0
	state.openUntil = time.Time{}
}

func (p *Policy) markFailure(provider string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	state := p.breaker(provider)
	state.consecutiveFailures++
	if state.consecutiveFailures < p.cfg.BreakerFailureThresh {
		return
	}

	state.openUntil = time.Now().Add(p.cfg.BreakerOpenFor)
	state.consecutiveFailures = 0
}

func (p *Policy) breaker(provider string) *breakerState {
	state := p.breakers[provider]
	if state == nil {
		state = &breakerState{}
		p.breakers[provider] = state
	}
	return state
}

func sanitizeChatRequest(req llm.ChatRequest, target routing.Target) llm.ChatRequest {
	if len(target.IgnoreRequestFields) == 0 {
		return req
	}
	out := req
	for _, field := range target.IgnoreRequestFields {
		switch canonicalRequestField(field) {
		case "temperature":
			out.Temperature = nil
		case "top_p":
			out.TopP = nil
		case "max_tokens":
			out.MaxTokens = 0
		}
	}
	return out
}

func rewriteRequestForTarget(body []byte, target routing.Target) ([]byte, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, fmt.Errorf("invalid json: %w", err)
	}
	payload["model"] = target.BackendModel
	for _, field := range target.IgnoreRequestFields {
		for _, alias := range requestFieldAliases(field) {
			delete(payload, alias)
		}
	}
	out, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	return out, nil
}

func canonicalRequestField(field string) string {
	switch strings.ToLower(strings.TrimSpace(field)) {
	case "max_output_tokens":
		return "max_tokens"
	default:
		return strings.ToLower(strings.TrimSpace(field))
	}
}

func requestFieldAliases(field string) []string {
	switch canonicalRequestField(field) {
	case "max_tokens":
		return []string{"max_tokens", "max_output_tokens"}
	case "temperature":
		return []string{"temperature"}
	case "top_p":
		return []string{"top_p"}
	default:
		canonical := canonicalRequestField(field)
		if canonical == "" {
			return nil
		}
		return []string{canonical}
	}
}

func closeResponseBody(scope string, body interface{ Close() error }) {
	if body == nil {
		return
	}
	if err := body.Close(); err != nil {
		slog.Warn("close response body", "scope", scope, "err", err)
	}
}
