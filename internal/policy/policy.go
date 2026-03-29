package policy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/jiayx/llmio/internal/llm"
	protocols "github.com/jiayx/llmio/internal/protocols"
	providerapi "github.com/jiayx/llmio/internal/providers/api"
	"github.com/jiayx/llmio/internal/routing"
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

	mu       sync.Mutex
	breakers map[string]*breakerState
}

// New constructs an execution policy.
func New(cfg Config, adapters map[string]providerapi.ProviderAdapter) *Policy {
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

		providerReq := req
		providerReq.Model = target.BackendModel
		attemptCtx, cancel := context.WithTimeout(ctx, p.cfg.RequestTimeout)
		resp, err := p.providers[target.ProviderName].Chat(attemptCtx, providerReq)
		cancel()
		if err == nil {
			p.markSuccess(target.ProviderName)
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

		providerReq := req
		providerReq.Model = target.BackendModel
		attemptCtx, cancel := context.WithTimeout(ctx, p.cfg.StreamSetupTimeout)
		stream, err := p.providers[target.ProviderName].ChatStream(attemptCtx, providerReq)
		cancel()
		if err == nil {
			p.markSuccess(target.ProviderName)
			return stream, nil
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

		payload, err := rewriteRequestModel(meta.Body, target.BackendModel)
		if err != nil {
			return nil, true, err
		}

		attemptCtx, cancel := context.WithTimeout(ctx, p.timeoutFor(meta.Stream))
		resp, err := forwarder.Forward(attemptCtx, meta.Protocol, meta.UpstreamPath, payload, meta.Headers)
		cancel()
		if err != nil {
			p.markFailure(target.ProviderName)
			if IsRetryableError(err) {
				continue
			}
			return nil, true, fmt.Errorf("provider %s: %w", target.ProviderName, err)
		}
		if IsRetryableStatus(resp.StatusCode) {
			p.markFailure(target.ProviderName)
			closeResponseBody("passthrough retry", resp.Body)
			continue
		}

		p.markSuccess(target.ProviderName)
		return resp, true, nil
	}

	return nil, false, nil
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

func rewriteRequestModel(body []byte, backendModel string) ([]byte, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, fmt.Errorf("invalid json: %w", err)
	}
	payload["model"] = backendModel
	out, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	return out, nil
}

func closeResponseBody(scope string, body interface{ Close() error }) {
	if body == nil {
		return
	}
	if err := body.Close(); err != nil {
		slog.Warn("close response body", "scope", scope, "err", err)
	}
}
