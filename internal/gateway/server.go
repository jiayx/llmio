package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"runtime/debug"
	"strings"
	"sync/atomic"

	"github.com/jiayx/llmio/internal/apikeys"
	"github.com/jiayx/llmio/internal/billing"
	"github.com/jiayx/llmio/internal/config"
	"github.com/jiayx/llmio/internal/llm"
	"github.com/jiayx/llmio/internal/observability"
	"github.com/jiayx/llmio/internal/policy"
	protocols "github.com/jiayx/llmio/internal/protocols"
	protocolanthropic "github.com/jiayx/llmio/internal/protocols/anthropic"
	protocolopenai "github.com/jiayx/llmio/internal/protocols/openai"
	"github.com/jiayx/llmio/internal/providers"
	providerapi "github.com/jiayx/llmio/internal/providers/api"
	"github.com/jiayx/llmio/internal/routing"
	"github.com/jiayx/llmio/internal/runtimeconfig"
)

type chatProvider = providerapi.ProviderAdapter
type routeTarget = routing.Target
type modelRoute = routing.Route

type runtimeSnapshot struct {
	providers map[string]chatProvider
	router    *routing.Router
	policy    *policy.Policy
}

// Server routes compatible OpenAI and Anthropic requests to configured providers.
type Server struct {
	adminAPIKeys     map[string]apikeys.Principal
	apiKeyStore      *apikeys.Store
	runtimeStore     *runtimeconfig.Store
	snapshot         atomic.Pointer[runtimeSnapshot]
	protocolAdapters []protocols.ProtocolAdapter
}

// NewServer constructs a gateway server from config.
func NewServer(cfg *config.Config) (*Server, error) {
	s := &Server{
		adminAPIKeys:     make(map[string]apikeys.Principal, len(cfg.AdminAPIKeys)),
		protocolAdapters: protocols.DefaultAdapters(),
	}

	for _, key := range cfg.AdminAPIKeys {
		if key != "" {
			s.adminAPIKeys[key] = apikeys.Principal{
				ID:   "admin:" + maskAPIKey(key),
				Name: "admin",
			}
		}
	}

	store, err := apikeys.Open(cfg.DatabasePath)
	if err != nil {
		return nil, err
	}
	s.apiKeyStore = store

	runtimeStore, err := runtimeconfig.Open(cfg.DatabasePath)
	if err != nil {
		return nil, err
	}
	s.runtimeStore = runtimeStore
	if err := s.applyRuntimeConfig(config.RuntimeConfig{
		Providers:   cfg.Providers,
		ModelRoutes: cfg.ModelRoutes,
		Pricing:     cfg.Pricing,
	}); err != nil {
		return nil, err
	}

	return s, nil
}

// Handler returns the HTTP entrypoint for the gateway.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.handleHealth)
	mux.HandleFunc("/admin/usage", s.handleAdminUsage)
	mux.HandleFunc("/admin/api-keys", s.handleAdminAPIKeys)
	mux.HandleFunc("/admin/api-keys/", s.handleAdminAPIKeyByID)
	mux.HandleFunc("/admin/runtime-config", s.handleAdminRuntimeConfig)
	s.registerProtocolRoutes(mux)
	return observability.Middleware(s.withRecovery(s.withAuth(mux)))
}

func (s *Server) withAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			next.ServeHTTP(w, r)
			return
		}

		key := bearerToken(r.Header.Get("Authorization"))
		if key == "" {
			key = strings.TrimSpace(r.Header.Get("x-api-key"))
		}
		if strings.HasPrefix(r.URL.Path, "/admin/") {
			principal, ok := s.adminAPIKeys[key]
			if !ok {
				writeUnauthorized(w, r.URL.Path)
				return
			}
			next.ServeHTTP(w, r.WithContext(apikeys.WithPrincipal(r.Context(), principal)))
			return
		}

		principal, ok := s.authenticateLLMKey(key)
		if !ok {
			if len(s.adminAPIKeys) == 0 && (s.apiKeyStore == nil || len(s.apiKeyStore.List()) == 0) {
				next.ServeHTTP(w, r)
				return
			}
			writeUnauthorized(w, r.URL.Path)
			return
		}
		if principal.Managed && s.apiKeyStore != nil && s.apiKeyStore.BudgetExceeded(principal.ID) {
			writeBudgetExceeded(w, r.URL.Path)
			return
		}

		next.ServeHTTP(w, r.WithContext(apikeys.WithPrincipal(r.Context(), principal)))
	})
}

func (s *Server) authenticateLLMKey(key string) (apikeys.Principal, bool) {
	if principal, ok := s.adminAPIKeys[key]; ok {
		return principal, true
	}
	if s.apiKeyStore == nil {
		return apikeys.Principal{}, false
	}
	return s.apiKeyStore.Authenticate(key)
}

func writeUnauthorized(w http.ResponseWriter, path string) {
	if adapter, _, ok := protocols.LookupEndpoint(protocols.DefaultAdapters(), path); ok && adapter.Protocol() == protocols.ProtocolAnthropic {
		protocolanthropic.WriteError(w, http.StatusUnauthorized, "authentication_error", "unauthorized")
		return
	}
	if _, _, ok := protocols.LookupEndpoint(protocols.DefaultAdapters(), path); ok {
		protocolopenai.WriteError(w, http.StatusUnauthorized, "unauthorized")
		return
	}
	writeJSON(w, http.StatusUnauthorized, map[string]any{"error": "unauthorized"})
}

func writeBudgetExceeded(w http.ResponseWriter, path string) {
	if adapter, _, ok := protocols.LookupEndpoint(protocols.DefaultAdapters(), path); ok && adapter.Protocol() == protocols.ProtocolAnthropic {
		protocolanthropic.WriteError(w, http.StatusPaymentRequired, "permission_error", "budget exceeded")
		return
	}
	if _, _, ok := protocols.LookupEndpoint(protocols.DefaultAdapters(), path); ok {
		protocolopenai.WriteError(w, http.StatusPaymentRequired, "budget exceeded")
		return
	}
	writeJSON(w, http.StatusPaymentRequired, map[string]any{"error": "budget exceeded"})
}

func writePanicError(w http.ResponseWriter, path string) {
	if adapter, _, ok := protocols.LookupEndpoint(protocols.DefaultAdapters(), path); ok && adapter.Protocol() == protocols.ProtocolAnthropic {
		protocolanthropic.WriteError(w, http.StatusInternalServerError, "api_error", "internal server error")
		return
	}
	if _, _, ok := protocols.LookupEndpoint(protocols.DefaultAdapters(), path); ok {
		protocolopenai.WriteError(w, http.StatusInternalServerError, "internal server error")
		return
	}
	writeJSON(w, http.StatusInternalServerError, map[string]any{"error": "internal server error"})
}

func maskAPIKey(key string) string {
	key = strings.TrimSpace(key)
	if len(key) <= 8 {
		return key
	}
	return key[:8]
}

func (s *Server) withRecovery(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ww := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		requestID := observability.RequestIDFromContext(r.Context())
		defer func() {
			if recovered := recover(); recovered != nil {
				slog.Error("request panic",
					"request_id", requestID,
					"method", r.Method,
					"path", r.URL.Path,
					"remote", r.RemoteAddr,
					"panic", fmt.Sprint(recovered),
					"stack", string(debug.Stack()),
				)
				if ww.wrote {
					return
				}
				writePanicError(ww, r.URL.Path)
			}
		}()
		next.ServeHTTP(ww, r)
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("{\"status\":\"ok\"}\n"))
}

func (s *Server) handleAdminAPIKeys(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, http.StatusOK, map[string]any{
			"data": s.apiKeyStore.List(),
		})
	case http.MethodPost:
		var req struct {
			Name      string   `json:"name"`
			BudgetUSD *float64 `json:"budget_usd,omitempty"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": "invalid json"})
			return
		}
		created, err := s.apiKeyStore.Create(req.Name, req.BudgetUSD)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusCreated, created)
	default:
		w.Header().Set("Allow", http.MethodGet+", "+http.MethodPost)
		writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"error": "method not allowed"})
	}
}

func (s *Server) handleAdminAPIKeyByID(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/admin/api-keys/")
	if id == "" {
		http.NotFound(w, r)
		return
	}
	if strings.HasSuffix(id, "/usage") {
		id = strings.TrimSuffix(id, "/usage")
		if id == "" || strings.Contains(id, "/") {
			http.NotFound(w, r)
			return
		}
		s.handleAdminAPIKeyUsage(w, r, id)
		return
	}
	if strings.Contains(id, "/") {
		http.NotFound(w, r)
		return
	}

	switch r.Method {
	case http.MethodGet:
		key, ok := s.apiKeyStore.Get(id)
		if !ok {
			http.NotFound(w, r)
			return
		}
		writeJSON(w, http.StatusOK, key)
	case http.MethodDelete:
		key, err := s.apiKeyStore.Disable(id)
		if err != nil {
			if err == os.ErrNotExist {
				http.NotFound(w, r)
				return
			}
			writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, key)
	default:
		w.Header().Set("Allow", http.MethodGet+", "+http.MethodDelete)
		writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"error": "method not allowed"})
	}
}

func (s *Server) handleAdminAPIKeyUsage(w http.ResponseWriter, r *http.Request, id string) {
	if !allowsMethod(r.Method, http.MethodGet) {
		w.Header().Set("Allow", http.MethodGet)
		writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"error": "method not allowed"})
		return
	}
	report, ok := s.apiKeyStore.UsageByID(id)
	if !ok {
		http.NotFound(w, r)
		return
	}
	writeJSON(w, http.StatusOK, report)
}

func (s *Server) handleAdminUsage(w http.ResponseWriter, r *http.Request) {
	if !allowsMethod(r.Method, http.MethodGet) {
		w.Header().Set("Allow", http.MethodGet)
		writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"error": "method not allowed"})
		return
	}
	data, total := s.apiKeyStore.UsageReports()
	writeJSON(w, http.StatusOK, map[string]any{
		"total": total,
		"data":  data,
	})
}

func (s *Server) handleAdminRuntimeConfig(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		if s.runtimeStore == nil {
			writeJSON(w, http.StatusInternalServerError, map[string]any{"error": "runtime config store unavailable"})
			return
		}
		doc, _, err := s.runtimeStore.Load()
		if err != nil {
			if os.IsNotExist(err) {
				writeJSON(w, http.StatusOK, config.RuntimeConfig{})
				return
			}
			writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, doc)
	case http.MethodPut:
		if s.runtimeStore == nil {
			writeJSON(w, http.StatusInternalServerError, map[string]any{"error": "runtime config store unavailable"})
			return
		}
		var doc config.RuntimeConfig
		if err := json.NewDecoder(r.Body).Decode(&doc); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": "invalid json"})
			return
		}
		cfg := &config.Config{
			DatabasePath: "llmio.db",
			Providers:    doc.Providers,
			ModelRoutes:  doc.ModelRoutes,
			Pricing:      doc.Pricing,
		}
		if len(doc.Providers) > 0 || len(doc.ModelRoutes) > 0 || len(doc.Pricing) > 0 {
			if err := config.Prepare(cfg, "."); err != nil {
				writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
				return
			}
		}
		if _, _, _, err := s.buildRuntime(doc); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
			return
		}
		if err := s.runtimeStore.Save(doc); err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
			return
		}
		if err := s.applyRuntimeConfig(doc); err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, doc)
	default:
		w.Header().Set("Allow", http.MethodGet+", "+http.MethodPut)
		writeJSON(w, http.StatusMethodNotAllowed, map[string]any{"error": "method not allowed"})
	}
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		slog.Error("write json", "status", status, "err", err)
	}
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
		"request_id", observability.RequestIDFromContext(r.Context()),
		"protocol", "openai",
		"endpoint", "chat_completions",
		"external_model", req.Model,
		"stream", req.Stream,
		"route_targets", routeTargetsForLog(route),
	)

	meta, err := s.protocolRequestMetaForPath(r.Context(), r.URL.Path, req.Model, body, r.Header, req.Stream)
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
	chatReq.SourceProtocol = protocols.ProtocolOpenAI
	chatReq.SourceAPIType = protocols.APIChatCompletions
	chatReq.RawRequestBody = append([]byte(nil), body...)
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
		"request_id", observability.RequestIDFromContext(r.Context()),
		"protocol", "openai",
		"endpoint", "responses",
		"external_model", req.Model,
		"stream", req.Stream,
		"route_targets", routeTargetsForLog(route),
	)

	meta, err := s.protocolRequestMetaForPath(r.Context(), r.URL.Path, req.Model, body, r.Header, req.Stream)
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
	chatReq.SourceProtocol = protocols.ProtocolOpenAI
	chatReq.SourceAPIType = protocols.APIResponses
	chatReq.RawRequestBody = append([]byte(nil), body...)

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
		"request_id", observability.RequestIDFromContext(r.Context()),
		"protocol", "anthropic",
		"endpoint", "messages",
		"external_model", req.Model,
		"stream", req.Stream,
		"route_targets", routeTargetsForLog(route),
	)

	meta, err := s.protocolRequestMetaForPath(r.Context(), r.URL.Path, req.Model, body, r.Header, req.Stream)
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
	chatReq.SourceProtocol = protocols.ProtocolAnthropic
	chatReq.SourceAPIType = protocols.APIMessages
	chatReq.RawRequestBody = append([]byte(nil), body...)

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
	return s.currentSnapshot().policy
}

func (s *Server) resolveRoute(externalModel string) (modelRoute, bool) {
	snapshot := s.currentSnapshot()
	if snapshot.router == nil {
		return modelRoute{}, false
	}
	return snapshot.router.Resolve(externalModel)
}

func (s *Server) modelInfos() []routing.ModelInfo {
	snapshot := s.currentSnapshot()
	if snapshot.router == nil {
		return nil
	}
	return snapshot.router.ModelInfos()
}

func (s *Server) applyRuntimeConfig(doc config.RuntimeConfig) error {
	providersMap, router, execPolicy, err := s.buildRuntime(doc)
	if err != nil {
		return err
	}

	s.snapshot.Store(&runtimeSnapshot{
		providers: providersMap,
		router:    router,
		policy:    execPolicy,
	})
	return nil
}

func (s *Server) buildRuntime(doc config.RuntimeConfig) (map[string]chatProvider, *routing.Router, *policy.Policy, error) {
	providersMap := make(map[string]chatProvider, len(doc.Providers))
	for _, p := range doc.Providers {
		adapter, err := providers.NewAdapter(p)
		if err != nil {
			return nil, nil, nil, err
		}
		providersMap[p.Name] = adapter
	}

	var (
		router *routing.Router
		err    error
	)
	if len(doc.ModelRoutes) > 0 {
		router, err = routing.New(&config.Config{
			Providers:   doc.Providers,
			ModelRoutes: doc.ModelRoutes,
			Pricing:     doc.Pricing,
		}, providersMap)
		if err != nil {
			return nil, nil, nil, err
		}
	}

	execPolicy := policy.NewWithRecorder(policy.DefaultConfig(), providersMap, billing.PricingRecorder{
		Catalog: billing.NewCatalog(doc.Pricing),
		Next:    s.apiKeyStore,
	})
	return providersMap, router, execPolicy, nil
}

func (s *Server) currentSnapshot() *runtimeSnapshot {
	if snapshot := s.snapshot.Load(); snapshot != nil {
		return snapshot
	}
	panic("gateway runtime snapshot is not initialized")
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

func (s *Server) protocolRequestMetaForPath(ctx context.Context, path, externalModel string, body []byte, headers http.Header, stream bool) (protocols.RequestMeta, error) {
	adapter, endpoint, ok := protocols.LookupEndpoint(s.protocolAdaptersOrDefault(), path)
	if !ok {
		return protocols.RequestMeta{}, fmt.Errorf("unsupported client path %q", path)
	}
	meta := adapter.Request(endpoint, externalModel, body, headers, stream)
	if principal, ok := apikeys.PrincipalFromContext(ctx); ok {
		meta.APIKeyID = principal.ID
		meta.APIKeyName = principal.Name
	}
	return meta, nil
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
