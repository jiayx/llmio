package gateway

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"slices"
	"strings"
	"time"

	clientmeta "github.com/jiayx/llmio/internal/clients"
	anthropicclient "github.com/jiayx/llmio/internal/clients/anthropic"
	openaiclient "github.com/jiayx/llmio/internal/clients/openai"
	"github.com/jiayx/llmio/internal/config"
	"github.com/jiayx/llmio/internal/llm"
	"github.com/jiayx/llmio/internal/policy"
	"github.com/jiayx/llmio/internal/providers"
	providerapi "github.com/jiayx/llmio/internal/providers/api"
	"github.com/jiayx/llmio/internal/routing"
	transporthttp "github.com/jiayx/llmio/internal/transport/http"
)

type chatProvider = providerapi.ProviderAdapter
type routeTarget = routing.Target
type modelRoute = routing.Route

// Server routes compatible OpenAI and Anthropic requests to configured providers.
type Server struct {
	apiKeys   map[string]struct{}
	providers map[string]chatProvider
	models    map[string]modelRoute
	router    *routing.Router
	policy    *policy.Policy
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

	for _, route := range cfg.ModelRoutes {
		resolved, ok := router.Resolve(route.ExternalModel)
		if !ok {
			return nil, fmt.Errorf("model route %q not found after router build", route.ExternalModel)
		}
		s.models[route.ExternalModel] = resolved
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
	return s.withLogging(s.withAuth(mux))
}

func (s *Server) withAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" || len(s.apiKeys) == 0 {
			next.ServeHTTP(w, r)
			return
		}

		key := transporthttp.BearerToken(r.Header.Get("Authorization"))
		if key == "" {
			key = strings.TrimSpace(r.Header.Get("x-api-key"))
		}
		if _, ok := s.apiKeys[key]; !ok {
			if clientmeta.ProtocolForPath(r.URL.Path) == clientmeta.ProtocolAnthropic {
				anthropicclient.WriteError(w, http.StatusUnauthorized, "authentication_error", "unauthorized")
			} else {
				openaiclient.WriteError(w, http.StatusUnauthorized, "unauthorized")
			}
			return
		}

		next.ServeHTTP(w, r)
	})
}

func (s *Server) withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		ww := &transporthttp.StatusRecorder{ResponseWriter: w, Status: http.StatusOK}
		next.ServeHTTP(ww, r)
		slog.Info("request completed",
			"method", r.Method,
			"path", r.URL.Path,
			"status", ww.Status,
			"duration", time.Since(start).Truncate(time.Millisecond),
			"remote", r.RemoteAddr,
		)
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	transporthttp.WriteJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if !allowsMethod(r.Method, http.MethodGet, http.MethodHead) {
		writeMethodNotAllowed(w, "openai", http.MethodGet)
		return
	}

	openaiclient.WriteModels(w, s.modelInfos())
}

func (s *Server) handleOpenAIChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeMethodNotAllowed(w, "openai", http.MethodPost)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		openaiclient.WriteError(w, http.StatusBadRequest, fmt.Sprintf("read request: %v", err))
		return
	}

	req, err := openaiclient.DecodeChatCompletionRequest(body)
	if err != nil {
		openaiclient.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}

	route, ok := s.resolveRoute(req.Model)
	if !ok {
		openaiclient.WriteError(w, http.StatusBadRequest, fmt.Sprintf("unknown model %q", req.Model))
		return
	}
	slog.Debug("gateway request decoded",
		"protocol", "openai",
		"endpoint", "chat_completions",
		"external_model", req.Model,
		"stream", req.Stream,
		"route_targets", routeTargetsForLog(route),
	)

	meta := openaiclient.NewChatCompletionRequestMeta(req, body, r.Header)
	if resp, handled, err := s.tryPassthrough(r.Context(), route, meta); handled {
		if err != nil {
			openaiclient.WriteError(w, http.StatusBadGateway, err.Error())
			return
		}
		openaiclient.WritePassthroughResponse(w, resp, req.Model)
		return
	}

	chatReq, err := openaiclient.ChatCompletionRequestToLLM(req)
	if err != nil {
		openaiclient.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.Stream {
		s.handleOpenAIStream(w, r, route, chatReq)
		return
	}

	resp, err := s.dispatchChat(r.Context(), route, chatReq)
	if err != nil {
		openaiclient.WriteError(w, http.StatusBadGateway, err.Error())
		return
	}

	openaiclient.WriteChatCompletionResponse(w, req.Model, resp)
}

func (s *Server) handleOpenAIResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeMethodNotAllowed(w, "openai", http.MethodPost)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		openaiclient.WriteError(w, http.StatusBadRequest, fmt.Sprintf("read request: %v", err))
		return
	}

	req, err := openaiclient.DecodeResponsesRequest(body)
	if err != nil {
		openaiclient.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}

	route, ok := s.resolveRoute(req.Model)
	if !ok {
		openaiclient.WriteError(w, http.StatusBadRequest, fmt.Sprintf("unknown model %q", req.Model))
		return
	}
	slog.Debug("gateway request decoded",
		"protocol", "openai",
		"endpoint", "responses",
		"external_model", req.Model,
		"stream", req.Stream,
		"route_targets", routeTargetsForLog(route),
	)

	meta := openaiclient.NewResponsesRequestMeta(req, body, r.Header)
	if resp, handled, err := s.tryPassthrough(r.Context(), route, meta); handled {
		if err != nil {
			openaiclient.WriteError(w, http.StatusBadGateway, err.Error())
			return
		}
		openaiclient.WritePassthroughResponse(w, resp, req.Model)
		return
	}

	chatReq, err := openaiclient.ResponsesRequestToLLM(req)
	if err != nil {
		openaiclient.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}

	if req.Stream {
		s.handleOpenAIResponsesStream(w, r, req.Model, route, chatReq)
		return
	}

	resp, err := s.dispatchChat(r.Context(), route, chatReq)
	if err != nil {
		openaiclient.WriteError(w, http.StatusBadGateway, err.Error())
		return
	}

	openaiclient.WriteResponsesResponse(w, req.Model, resp)
}

func (s *Server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeMethodNotAllowed(w, "anthropic", http.MethodPost)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		anthropicclient.WriteError(w, http.StatusBadRequest, "invalid_request_error", fmt.Sprintf("read request: %v", err))
		return
	}

	req, err := anthropicclient.DecodeMessagesRequest(body)
	if err != nil {
		anthropicclient.WriteError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	route, ok := s.resolveRoute(req.Model)
	if !ok {
		anthropicclient.WriteError(w, http.StatusBadRequest, "not_found_error", fmt.Sprintf("unknown model %q", req.Model))
		return
	}
	slog.Debug("gateway request decoded",
		"protocol", "anthropic",
		"endpoint", "messages",
		"external_model", req.Model,
		"stream", req.Stream,
		"route_targets", routeTargetsForLog(route),
	)

	meta := anthropicclient.NewMessagesRequestMeta(req, body, r.Header)
	if resp, handled, err := s.tryPassthrough(r.Context(), route, meta); handled {
		if err != nil {
			anthropicclient.WriteError(w, http.StatusBadGateway, "api_error", err.Error())
			return
		}
		anthropicclient.WritePassthroughResponse(w, resp, req.Model)
		return
	}

	chatReq, err := anthropicclient.MessagesRequestToLLM(req)
	if err != nil {
		anthropicclient.WriteError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	if req.Stream {
		s.handleAnthropicStream(w, r, req.Model, route, chatReq)
		return
	}

	resp, err := s.dispatchChat(r.Context(), route, chatReq)
	if err != nil {
		anthropicclient.WriteError(w, http.StatusBadGateway, "api_error", err.Error())
		return
	}

	anthropicclient.WriteMessagesResponse(w, req.Model, resp)
}

func (s *Server) dispatchChat(ctx context.Context, route modelRoute, req llm.ChatRequest) (*llm.ChatResponse, error) {
	return s.executionPolicy().ExecuteChat(ctx, route, req)
}

func (s *Server) dispatchChatStream(ctx context.Context, route modelRoute, req llm.ChatRequest) (*providerapi.StreamReader, error) {
	return s.executionPolicy().ExecuteStream(ctx, route, req)
}

func (s *Server) tryPassthrough(ctx context.Context, route modelRoute, meta clientmeta.RequestMeta) (*http.Response, bool, error) {
	return s.executionPolicy().ExecutePassthrough(ctx, route, meta)
}

func writeMethodNotAllowed(w http.ResponseWriter, protocol, allow string) {
	w.Header().Set("Allow", allow)
	message := fmt.Sprintf("method not allowed: use %s", allow)
	if protocol == "anthropic" {
		anthropicclient.WriteError(w, http.StatusMethodNotAllowed, "invalid_request_error", message)
		return
	}
	openaiclient.WriteError(w, http.StatusMethodNotAllowed, message)
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
	if s.router != nil {
		if route, ok := s.router.Resolve(externalModel); ok {
			return route, true
		}
	}
	route, ok := s.models[externalModel]
	return route, ok
}

func (s *Server) modelInfos() []routing.ModelInfo {
	if s.router != nil {
		return s.router.ModelInfos()
	}

	keys := make([]string, 0, len(s.models))
	for external := range s.models {
		keys = append(keys, external)
	}
	slices.Sort(keys)

	out := make([]routing.ModelInfo, 0, len(keys))
	for _, external := range keys {
		route := s.models[external]
		owner := ""
		if len(route.Targets) > 0 {
			owner = route.Targets[0].ProviderName
		}
		out = append(out, routing.ModelInfo{
			ID:      external,
			OwnedBy: owner,
		})
	}
	return out
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

func (s *Server) handleOpenAIStream(w http.ResponseWriter, r *http.Request, route modelRoute, req llm.ChatRequest) {
	stream, err := s.dispatchChatStream(r.Context(), route, req)
	if err != nil {
		openaiclient.WriteError(w, http.StatusBadGateway, err.Error())
		return
	}
	openaiclient.ServeChatCompletionStream(w, r, req.Model, stream)
}

func (s *Server) handleOpenAIResponsesStream(w http.ResponseWriter, r *http.Request, externalModel string, route modelRoute, req llm.ChatRequest) {
	stream, err := s.dispatchChatStream(r.Context(), route, req)
	if err != nil {
		openaiclient.WriteError(w, http.StatusBadGateway, err.Error())
		return
	}
	openaiclient.ServeResponsesStream(w, r, externalModel, stream)
}

func (s *Server) handleAnthropicStream(w http.ResponseWriter, r *http.Request, externalModel string, route modelRoute, req llm.ChatRequest) {
	stream, err := s.dispatchChatStream(r.Context(), route, req)
	if err != nil {
		anthropicclient.WriteError(w, http.StatusBadGateway, "api_error", err.Error())
		return
	}
	anthropicclient.ServeMessagesStream(w, r, externalModel, stream)
}
