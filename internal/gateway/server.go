package gateway

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/jiayx/llmio/internal/config"
	"github.com/jiayx/llmio/internal/llm"
	"github.com/jiayx/llmio/internal/policy"
	protocols "github.com/jiayx/llmio/internal/protocols"
	protocolanthropic "github.com/jiayx/llmio/internal/protocols/anthropic"
	protocolopenai "github.com/jiayx/llmio/internal/protocols/openai"
	"github.com/jiayx/llmio/internal/providers"
	providerapi "github.com/jiayx/llmio/internal/providers/api"
	"github.com/jiayx/llmio/internal/routing"
)

type chatProvider = providerapi.ProviderAdapter
type routeTarget = routing.Target
type modelRoute = routing.Route

// Server routes compatible OpenAI and Anthropic requests to configured providers.
type Server struct {
	apiKeys          map[string]struct{}
	providers        map[string]chatProvider
	router           *routing.Router
	policy           *policy.Policy
	protocolAdapters []protocols.ProtocolAdapter
}

// NewServer constructs a gateway server from config.
func NewServer(cfg *config.Config) (*Server, error) {
	s := &Server{
		apiKeys:          make(map[string]struct{}, len(cfg.APIKeys)),
		providers:        make(map[string]chatProvider, len(cfg.Providers)),
		protocolAdapters: protocols.DefaultAdapters(),
	}

	for _, key := range cfg.APIKeys {
		if key != "" {
			s.apiKeys[key] = struct{}{}
		}
	}

	for _, p := range cfg.Providers {
		adapter, err := providers.NewAdapter(p)
		if err != nil {
			return nil, err
		}
		s.providers[p.Name] = adapter
	}

	router, err := routing.New(cfg, s.providers)
	if err != nil {
		return nil, err
	}
	s.router = router
	s.policy = policy.New(policy.DefaultConfig(), s.providers)

	return s, nil
}

// Handler returns the HTTP entrypoint for the gateway.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.handleHealth)
	s.registerProtocolRoutes(mux)
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
			if route, ok := s.protocolAdapterForPath(r.URL.Path); ok && route.Protocol() == protocols.ProtocolAnthropic {
				protocolanthropic.WriteError(w, http.StatusUnauthorized, "authentication_error", "unauthorized")
			} else if ok {
				protocolopenai.WriteError(w, http.StatusUnauthorized, "unauthorized")
			} else {
				http.Error(w, "unauthorized", http.StatusUnauthorized)
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
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("{\"status\":\"ok\"}\n"))
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if !allowsMethod(r.Method, http.MethodGet, http.MethodHead) {
		writeMethodNotAllowed(w, "openai", http.MethodGet)
		return
	}

	protocolopenai.WriteModels(w, s.modelInfos())
}

func (s *Server) handleOpenAIChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeMethodNotAllowed(w, "openai", http.MethodPost)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		protocolopenai.WriteError(w, http.StatusBadRequest, fmt.Sprintf("read request: %v", err))
		return
	}

	req, err := protocolopenai.DecodeChatCompletionRequest(body)
	if err != nil {
		protocolopenai.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}

	route, ok := s.resolveRoute(req.Model)
	if !ok {
		protocolopenai.WriteError(w, http.StatusBadRequest, fmt.Sprintf("unknown model %q", req.Model))
		return
	}
	slog.Debug("gateway request decoded",
		"protocol", "openai",
		"endpoint", "chat_completions",
		"external_model", req.Model,
		"stream", req.Stream,
		"route_targets", routeTargetsForLog(route),
	)

	meta, err := s.protocolRequestMetaForPath(r.URL.Path, req.Model, body, r.Header, req.Stream)
	if err != nil {
		protocolopenai.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if resp, handled, err := s.tryPassthrough(r.Context(), route, meta); handled {
		if err != nil {
			protocolopenai.WriteError(w, http.StatusBadGateway, err.Error())
			return
		}
		protocolopenai.WritePassthroughResponse(w, resp, req.Model)
		return
	}

	chatReq, err := protocolopenai.ChatCompletionRequestToLLM(req)
	if err != nil {
		protocolopenai.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.Stream {
		s.handleOpenAIStream(w, r, route, chatReq)
		return
	}

	resp, err := s.dispatchChat(r.Context(), route, chatReq)
	if err != nil {
		protocolopenai.WriteError(w, http.StatusBadGateway, err.Error())
		return
	}

	protocolopenai.WriteChatCompletionResponse(w, req.Model, resp)
}

func (s *Server) handleOpenAIResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeMethodNotAllowed(w, "openai", http.MethodPost)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		protocolopenai.WriteError(w, http.StatusBadRequest, fmt.Sprintf("read request: %v", err))
		return
	}

	req, err := protocolopenai.DecodeResponsesRequest(body)
	if err != nil {
		protocolopenai.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}

	route, ok := s.resolveRoute(req.Model)
	if !ok {
		protocolopenai.WriteError(w, http.StatusBadRequest, fmt.Sprintf("unknown model %q", req.Model))
		return
	}
	slog.Debug("gateway request decoded",
		"protocol", "openai",
		"endpoint", "responses",
		"external_model", req.Model,
		"stream", req.Stream,
		"route_targets", routeTargetsForLog(route),
	)

	meta, err := s.protocolRequestMetaForPath(r.URL.Path, req.Model, body, r.Header, req.Stream)
	if err != nil {
		protocolopenai.WriteError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if resp, handled, err := s.tryPassthrough(r.Context(), route, meta); handled {
		if err != nil {
			protocolopenai.WriteError(w, http.StatusBadGateway, err.Error())
			return
		}
		protocolopenai.WritePassthroughResponse(w, resp, req.Model)
		return
	}

	chatReq, err := protocolopenai.ResponsesRequestToLLM(req)
	if err != nil {
		protocolopenai.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}

	if req.Stream {
		s.handleOpenAIResponsesStream(w, r, req.Model, route, chatReq)
		return
	}

	resp, err := s.dispatchChat(r.Context(), route, chatReq)
	if err != nil {
		protocolopenai.WriteError(w, http.StatusBadGateway, err.Error())
		return
	}

	protocolopenai.WriteResponsesResponse(w, req.Model, resp)
}

func (s *Server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeMethodNotAllowed(w, "anthropic", http.MethodPost)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		protocolanthropic.WriteError(w, http.StatusBadRequest, "invalid_request_error", fmt.Sprintf("read request: %v", err))
		return
	}

	req, err := protocolanthropic.DecodeMessagesRequest(body)
	if err != nil {
		protocolanthropic.WriteError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	route, ok := s.resolveRoute(req.Model)
	if !ok {
		protocolanthropic.WriteError(w, http.StatusBadRequest, "not_found_error", fmt.Sprintf("unknown model %q", req.Model))
		return
	}
	slog.Debug("gateway request decoded",
		"protocol", "anthropic",
		"endpoint", "messages",
		"external_model", req.Model,
		"stream", req.Stream,
		"route_targets", routeTargetsForLog(route),
	)

	meta, err := s.protocolRequestMetaForPath(r.URL.Path, req.Model, body, r.Header, req.Stream)
	if err != nil {
		protocolanthropic.WriteError(w, http.StatusInternalServerError, "api_error", err.Error())
		return
	}
	if resp, handled, err := s.tryPassthrough(r.Context(), route, meta); handled {
		if err != nil {
			protocolanthropic.WriteError(w, http.StatusBadGateway, "api_error", err.Error())
			return
		}
		protocolanthropic.WritePassthroughResponse(w, resp, req.Model)
		return
	}

	chatReq, err := protocolanthropic.MessagesRequestToLLM(req)
	if err != nil {
		protocolanthropic.WriteError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	if req.Stream {
		s.handleAnthropicStream(w, r, req.Model, route, chatReq)
		return
	}

	resp, err := s.dispatchChat(r.Context(), route, chatReq)
	if err != nil {
		protocolanthropic.WriteError(w, http.StatusBadGateway, "api_error", err.Error())
		return
	}

	protocolanthropic.WriteMessagesResponse(w, req.Model, resp)
}

func (s *Server) dispatchChat(ctx context.Context, route modelRoute, req llm.ChatRequest) (*llm.ChatResponse, error) {
	return s.executionPolicy().ExecuteChat(ctx, route, req)
}

func (s *Server) dispatchChatStream(ctx context.Context, route modelRoute, req llm.ChatRequest) (*providerapi.StreamReader, error) {
	return s.executionPolicy().ExecuteStream(ctx, route, req)
}

func (s *Server) tryPassthrough(ctx context.Context, route modelRoute, meta protocols.RequestMeta) (*http.Response, bool, error) {
	return s.executionPolicy().ExecutePassthrough(ctx, route, meta)
}

func writeMethodNotAllowed(w http.ResponseWriter, protocol, allow string) {
	w.Header().Set("Allow", allow)
	message := fmt.Sprintf("method not allowed: use %s", allow)
	if protocol == "anthropic" {
		protocolanthropic.WriteError(w, http.StatusMethodNotAllowed, "invalid_request_error", message)
		return
	}
	protocolopenai.WriteError(w, http.StatusMethodNotAllowed, message)
}

func allowsMethod(method string, allowed ...string) bool {
	for _, candidate := range allowed {
		if method == candidate {
			return true
		}
	}
	return false
}

func (s *Server) executionPolicy() *policy.Policy {
	if s.policy == nil {
		s.policy = policy.New(policy.DefaultConfig(), s.providers)
	}
	return s.policy
}

func (s *Server) resolveRoute(externalModel string) (modelRoute, bool) {
	if s.router == nil {
		return modelRoute{}, false
	}
	return s.router.Resolve(externalModel)
}

func (s *Server) modelInfos() []routing.ModelInfo {
	if s.router == nil {
		return nil
	}
	return s.router.ModelInfos()
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

func (s *Server) registerProtocolRoutes(mux *http.ServeMux) {
	for _, adapter := range s.protocolAdaptersOrDefault() {
		for _, endpoint := range adapter.Endpoints() {
			if handler := s.handlerForEndpoint(adapter.Protocol(), endpoint.APIType); handler != nil {
				mux.HandleFunc(endpoint.InboundPath, handler)
			}
		}
	}
}

func (s *Server) handlerForEndpoint(protocol, apiType string) http.HandlerFunc {
	switch {
	case protocol == protocols.ProtocolOpenAI && apiType == protocols.APIModels:
		return s.handleModels
	case protocol == protocols.ProtocolOpenAI && apiType == protocols.APIChatCompletions:
		return s.handleOpenAIChatCompletions
	case protocol == protocols.ProtocolOpenAI && apiType == protocols.APIResponses:
		return s.handleOpenAIResponses
	case protocol == protocols.ProtocolAnthropic && apiType == protocols.APIMessages:
		return s.handleAnthropicMessages
	default:
		return nil
	}
}

func (s *Server) protocolAdaptersOrDefault() []protocols.ProtocolAdapter {
	if len(s.protocolAdapters) == 0 {
		return protocols.DefaultAdapters()
	}
	return s.protocolAdapters
}

func (s *Server) protocolAdapterForPath(path string) (protocols.ProtocolAdapter, bool) {
	adapter, _, ok := protocols.LookupEndpoint(s.protocolAdaptersOrDefault(), path)
	return adapter, ok
}

func (s *Server) protocolRequestMetaForPath(path, externalModel string, body []byte, headers http.Header, stream bool) (protocols.RequestMeta, error) {
	adapter, endpoint, ok := protocols.LookupEndpoint(s.protocolAdaptersOrDefault(), path)
	if !ok {
		return protocols.RequestMeta{}, fmt.Errorf("unsupported client path %q", path)
	}
	return adapter.Request(endpoint, externalModel, body, headers, stream), nil
}

func (s *Server) handleOpenAIStream(w http.ResponseWriter, r *http.Request, route modelRoute, req llm.ChatRequest) {
	stream, err := s.dispatchChatStream(r.Context(), route, req)
	if err != nil {
		protocolopenai.WriteError(w, http.StatusBadGateway, err.Error())
		return
	}
	protocolopenai.ServeChatCompletionStream(w, r, req.Model, stream)
}

func (s *Server) handleOpenAIResponsesStream(w http.ResponseWriter, r *http.Request, externalModel string, route modelRoute, req llm.ChatRequest) {
	stream, err := s.dispatchChatStream(r.Context(), route, req)
	if err != nil {
		protocolopenai.WriteError(w, http.StatusBadGateway, err.Error())
		return
	}
	protocolopenai.ServeResponsesStream(w, r, externalModel, stream)
}

func (s *Server) handleAnthropicStream(w http.ResponseWriter, r *http.Request, externalModel string, route modelRoute, req llm.ChatRequest) {
	stream, err := s.dispatchChatStream(r.Context(), route, req)
	if err != nil {
		protocolanthropic.WriteError(w, http.StatusBadGateway, "api_error", err.Error())
		return
	}
	protocolanthropic.ServeMessagesStream(w, r, externalModel, stream)
}
