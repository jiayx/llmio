package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/jiayx/llmio/internal/config"
	"github.com/jiayx/llmio/internal/core"
	anthropicproto "github.com/jiayx/llmio/internal/protocols/anthropic"
	openaiproto "github.com/jiayx/llmio/internal/protocols/openai"
	"github.com/jiayx/llmio/internal/providers"
)

type chatProvider = providers.ChatProvider

type routeTarget struct {
	ProviderName string
	BackendModel string
}

type modelRoute struct {
	Targets []routeTarget
}

// Server routes compatible OpenAI and Anthropic requests to configured providers.
type Server struct {
	apiKeys   map[string]struct{}
	providers map[string]chatProvider
	models    map[string]modelRoute
}

// NewServer constructs a gateway server from config.
func NewServer(cfg *config.Config) (*Server, error) {
	s := &Server{
		apiKeys:   make(map[string]struct{}, len(cfg.APIKeys)),
		providers: make(map[string]chatProvider, len(cfg.Providers)),
		models:    make(map[string]modelRoute, len(cfg.ModelRoutes)),
	}

	for _, key := range cfg.APIKeys {
		if key != "" {
			s.apiKeys[key] = struct{}{}
		}
	}

	for _, p := range cfg.Providers {
		switch p.Type {
		case "openai-compatible":
			s.providers[p.Name] = providers.NewOpenAICompatible(p)
		case "anthropic-native":
			s.providers[p.Name] = providers.NewAnthropicNative(p)
		default:
			return nil, fmt.Errorf("unsupported provider type %q", p.Type)
		}
	}

	for _, route := range cfg.ModelRoutes {
		targets := make([]routeTarget, 0, len(route.Targets))
		for _, target := range route.Targets {
			if _, ok := s.providers[target.Provider]; !ok {
				return nil, fmt.Errorf("model route %q references unknown provider %q", route.ExternalModel, target.Provider)
			}
			targets = append(targets, routeTarget{
				ProviderName: target.Provider,
				BackendModel: target.BackendModel,
			})
		}
		s.models[route.ExternalModel] = modelRoute{Targets: targets}
	}

	return s, nil
}

// Handler returns the HTTP entrypoint for the gateway.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.handleHealth)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/models", s.handleModels)
	mux.HandleFunc("/v1/chat/completions", s.handleOpenAIChatCompletions)
	mux.HandleFunc("/chat/completions", s.handleOpenAIChatCompletions)
	mux.HandleFunc("/v1/responses", s.handleOpenAIResponses)
	mux.HandleFunc("/responses", s.handleOpenAIResponses)
	mux.HandleFunc("/anthropic/v1/messages", s.handleAnthropicMessages)
	mux.HandleFunc("/messages", s.handleAnthropicMessages)
	return s.withLogging(s.withAuth(mux))
}

func (s *Server) withAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" || len(s.apiKeys) == 0 {
			next.ServeHTTP(w, r)
			return
		}

		key := bearerToken(r.Header.Get("Authorization"))
		if key == "" {
			key = strings.TrimSpace(r.Header.Get("x-api-key"))
		}
		if _, ok := s.apiKeys[key]; !ok {
			if strings.HasPrefix(r.URL.Path, "/anthropic/") {
				writeAnthropicError(w, http.StatusUnauthorized, "authentication_error", "unauthorized")
			} else {
				writeOpenAIError(w, http.StatusUnauthorized, "unauthorized")
			}
			return
		}

		next.ServeHTTP(w, r)
	})
}

func (s *Server) withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		ww := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(ww, r)
		slog.Info("request completed",
			"method", r.Method,
			"path", r.URL.Path,
			"status", ww.status,
			"duration", time.Since(start).Truncate(time.Millisecond),
			"remote", r.RemoteAddr,
		)
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	models := make([]openaiproto.ModelInfo, 0, len(s.models))
	for external, route := range s.models {
		owner := ""
		if len(route.Targets) > 0 {
			owner = route.Targets[0].ProviderName
		}
		models = append(models, openaiproto.ModelInfo{
			ID:      external,
			Object:  "model",
			OwnedBy: owner,
		})
	}
	writeJSON(w, http.StatusOK, openaiproto.ModelsResponse{
		Object: "list",
		Data:   models,
	})
}

func (s *Server) handleOpenAIChatCompletions(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, fmt.Sprintf("read request: %v", err))
		return
	}

	var req openaiproto.ChatCompletionRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, fmt.Sprintf("invalid json: %v", err))
		return
	}

	route, ok := s.models[req.Model]
	if !ok {
		writeOpenAIError(w, http.StatusBadRequest, fmt.Sprintf("unknown model %q", req.Model))
		return
	}
	slog.Debug("gateway request decoded",
		"protocol", "openai",
		"endpoint", "chat_completions",
		"external_model", req.Model,
		"stream", req.Stream,
		"route_targets", routeTargetsForLog(route),
	)

	if resp, handled, err := s.tryOpenAIPassthrough(r.Context(), route, providers.OpenAIAPIChatCompletions, "/chat/completions", body, r.Header); handled {
		if err != nil {
			writeOpenAIError(w, http.StatusBadGateway, err.Error())
			return
		}
		writePassthroughResponse(w, resp, "openai", req.Model)
		return
	}

	chatReq := openAIRequestToCore(req)
	if req.Stream {
		s.handleOpenAIStream(w, r, route, chatReq)
		return
	}

	resp, err := s.dispatchChat(r.Context(), route, chatReq)
	if err != nil {
		writeOpenAIError(w, http.StatusBadGateway, err.Error())
		return
	}

	writeCoreResponseAsOpenAI(w, req.Model, resp)
}

func (s *Server) handleOpenAIResponses(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, fmt.Sprintf("read request: %v", err))
		return
	}

	var req openaiproto.ResponsesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, fmt.Sprintf("invalid json: %v", err))
		return
	}

	route, ok := s.models[req.Model]
	if !ok {
		writeOpenAIError(w, http.StatusBadRequest, fmt.Sprintf("unknown model %q", req.Model))
		return
	}
	slog.Debug("gateway request decoded",
		"protocol", "openai",
		"endpoint", "responses",
		"external_model", req.Model,
		"stream", req.Stream,
		"route_targets", routeTargetsForLog(route),
	)

	if resp, handled, err := s.tryOpenAIPassthrough(r.Context(), route, providers.OpenAIAPIResponses, "/responses", body, r.Header); handled {
		if err != nil {
			writeOpenAIError(w, http.StatusBadGateway, err.Error())
			return
		}
		writePassthroughResponse(w, resp, "openai", req.Model)
		return
	}

	chatReq, err := openAIResponsesRequestToCore(req)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error())
		return
	}

	if req.Stream {
		s.handleOpenAIResponsesStream(w, r, req.Model, route, chatReq)
		return
	}

	resp, err := s.dispatchChat(r.Context(), route, chatReq)
	if err != nil {
		writeOpenAIError(w, http.StatusBadGateway, err.Error())
		return
	}

	writeJSON(w, http.StatusOK, coreResponseToOpenAIResponse(req.Model, resp))
}

func (s *Server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", fmt.Sprintf("read request: %v", err))
		return
	}

	var req anthropicproto.MessagesRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", fmt.Sprintf("invalid json: %v", err))
		return
	}

	route, ok := s.models[req.Model]
	if !ok {
		writeAnthropicError(w, http.StatusBadRequest, "not_found_error", fmt.Sprintf("unknown model %q", req.Model))
		return
	}
	slog.Debug("gateway request decoded",
		"protocol", "anthropic",
		"endpoint", "messages",
		"external_model", req.Model,
		"stream", req.Stream,
		"route_targets", routeTargetsForLog(route),
	)

	if resp, handled, err := s.tryAnthropicPassthrough(r.Context(), route, providers.AnthropicAPIMessages, "/messages", body, r.Header); handled {
		if err != nil {
			writeAnthropicError(w, http.StatusBadGateway, "api_error", err.Error())
			return
		}
		writePassthroughResponse(w, resp, "anthropic", req.Model)
		return
	}

	chatReq, err := anthropicToCore(req)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	if req.Stream {
		s.handleAnthropicStream(w, r, req.Model, route, chatReq)
		return
	}

	resp, err := s.dispatchChat(r.Context(), route, chatReq)
	if err != nil {
		writeAnthropicError(w, http.StatusBadGateway, "api_error", err.Error())
		return
	}

	writeJSON(w, http.StatusOK, coreResponseToAnthropic(req.Model, resp))
}

func (s *Server) dispatchChat(ctx context.Context, route modelRoute, req core.ChatRequest) (*core.ChatResponse, error) {
	var lastErr error
	for _, target := range route.Targets {
		providerReq := req
		providerReq.Model = target.BackendModel
		slog.Debug("dispatch chat attempt",
			"provider", target.ProviderName,
			"backend_model", target.BackendModel,
			"stream", false,
		)
		resp, err := s.providers[target.ProviderName].Chat(ctx, providerReq)
		if err == nil {
			slog.Debug("dispatch chat success",
				"provider", target.ProviderName,
				"backend_model", target.BackendModel,
			)
			return resp, nil
		}
		slog.Debug("dispatch chat failed",
			"provider", target.ProviderName,
			"backend_model", target.BackendModel,
			"err", err,
		)
		lastErr = fmt.Errorf("provider %s: %w", target.ProviderName, err)
		if !retryableError(err) {
			return nil, lastErr
		}
	}
	if lastErr == nil {
		lastErr = errors.New("no provider available")
	}
	return nil, lastErr
}

func (s *Server) dispatchChatStream(ctx context.Context, route modelRoute, req core.ChatRequest) (*providers.StreamReader, error) {
	var lastErr error
	for _, target := range route.Targets {
		providerReq := req
		providerReq.Model = target.BackendModel
		slog.Debug("dispatch chat attempt",
			"provider", target.ProviderName,
			"backend_model", target.BackendModel,
			"stream", true,
		)
		stream, err := s.providers[target.ProviderName].ChatStream(ctx, providerReq)
		if err == nil {
			slog.Debug("dispatch chat success",
				"provider", target.ProviderName,
				"backend_model", target.BackendModel,
				"stream", true,
			)
			return stream, nil
		}
		slog.Debug("dispatch chat failed",
			"provider", target.ProviderName,
			"backend_model", target.BackendModel,
			"stream", true,
			"err", err,
		)
		lastErr = fmt.Errorf("provider %s: %w", target.ProviderName, err)
		if !retryableError(err) {
			return nil, lastErr
		}
	}
	if lastErr == nil {
		lastErr = errors.New("no provider available")
	}
	return nil, lastErr
}

func (s *Server) tryOpenAIPassthrough(ctx context.Context, route modelRoute, apiType, path string, body []byte, headers http.Header) (*http.Response, bool, error) {
	for _, target := range route.Targets {
		provider := s.providers[target.ProviderName]
		forwarder, ok := provider.(providers.OpenAIPassthroughProvider)
		if !ok {
			continue
		}
		if supporter, ok := provider.(providers.OpenAPITypeSupporter); ok && !supporter.SupportsOpenAIAPI(apiType) {
			slog.Debug("passthrough skipped",
				"protocol", "openai",
				"api_type", apiType,
				"path", path,
				"provider", target.ProviderName,
				"backend_model", target.BackendModel,
			)
			continue
		}
		slog.Debug("passthrough attempt",
			"protocol", "openai",
			"api_type", apiType,
			"path", path,
			"provider", target.ProviderName,
			"backend_model", target.BackendModel,
		)
		payload, err := rewriteRequestModel(body, target.BackendModel)
		if err != nil {
			return nil, true, err
		}
		resp, err := forwarder.ForwardOpenAI(ctx, path, payload, headers)
		if err != nil {
			if retryableError(err) {
				slog.Debug("passthrough retryable failure",
					"protocol", "openai",
					"api_type", apiType,
					"path", path,
					"provider", target.ProviderName,
					"backend_model", target.BackendModel,
					"err", err,
				)
				continue
			}
			return nil, true, fmt.Errorf("provider %s: %w", target.ProviderName, err)
		}
		if retryableStatus(resp.StatusCode) {
			slog.Debug("passthrough retryable status",
				"protocol", "openai",
				"api_type", apiType,
				"path", path,
				"provider", target.ProviderName,
				"backend_model", target.BackendModel,
				"status", resp.StatusCode,
			)
			closeResponseBody("openai passthrough retry", resp.Body)
			continue
		}
		slog.Debug("passthrough success",
			"protocol", "openai",
			"api_type", apiType,
			"path", path,
			"provider", target.ProviderName,
			"backend_model", target.BackendModel,
			"status", resp.StatusCode,
		)
		return resp, true, nil
	}
	return nil, false, nil
}

func (s *Server) tryAnthropicPassthrough(ctx context.Context, route modelRoute, apiType, path string, body []byte, headers http.Header) (*http.Response, bool, error) {
	for _, target := range route.Targets {
		provider := s.providers[target.ProviderName]
		forwarder, ok := provider.(providers.AnthropicPassthroughProvider)
		if !ok {
			continue
		}
		if supporter, ok := provider.(providers.AnthropicAPITypeSupporter); ok && !supporter.SupportsAnthropicAPI(apiType) {
			slog.Debug("passthrough skipped",
				"protocol", "anthropic",
				"api_type", apiType,
				"path", path,
				"provider", target.ProviderName,
				"backend_model", target.BackendModel,
			)
			continue
		}
		slog.Debug("passthrough attempt",
			"protocol", "anthropic",
			"api_type", apiType,
			"path", path,
			"provider", target.ProviderName,
			"backend_model", target.BackendModel,
		)
		payload, err := rewriteRequestModel(body, target.BackendModel)
		if err != nil {
			return nil, true, err
		}
		resp, err := forwarder.ForwardAnthropic(ctx, path, payload, headers)
		if err != nil {
			if retryableError(err) {
				slog.Debug("passthrough retryable failure",
					"protocol", "anthropic",
					"api_type", apiType,
					"path", path,
					"provider", target.ProviderName,
					"backend_model", target.BackendModel,
					"err", err,
				)
				continue
			}
			return nil, true, fmt.Errorf("provider %s: %w", target.ProviderName, err)
		}
		if retryableStatus(resp.StatusCode) {
			slog.Debug("passthrough retryable status",
				"protocol", "anthropic",
				"api_type", apiType,
				"path", path,
				"provider", target.ProviderName,
				"backend_model", target.BackendModel,
				"status", resp.StatusCode,
			)
			closeResponseBody("anthropic passthrough retry", resp.Body)
			continue
		}
		slog.Debug("passthrough success",
			"protocol", "anthropic",
			"api_type", apiType,
			"path", path,
			"provider", target.ProviderName,
			"backend_model", target.BackendModel,
			"status", resp.StatusCode,
		)
		return resp, true, nil
	}
	return nil, false, nil
}

func routeTargetsForLog(route modelRoute) []string {
	if len(route.Targets) == 0 {
		return nil
	}
	out := make([]string, 0, len(route.Targets))
	for _, target := range route.Targets {
		out = append(out, target.ProviderName+":"+target.BackendModel)
	}
	return out
}
