package routing

import (
	"fmt"
	"slices"

	"github.com/jiayx/llmio/internal/config"
	providerapi "github.com/jiayx/llmio/internal/providers/api"
)

// Target is one candidate backend for an external model.
type Target struct {
	ProviderName string
	BackendModel string
}

// Route maps one external model to ordered backend targets.
type Route struct {
	ExternalModel string
	Targets       []Target
}

// ModelInfo is the gateway-visible model inventory.
type ModelInfo struct {
	ID      string
	OwnedBy string
}

// Router resolves external model names to backend targets.
type Router struct {
	routes map[string]Route
}

// New constructs a static router from config.
func New(cfg *config.Config, adapters map[string]providerapi.ProviderAdapter) (*Router, error) {
	r := &Router{routes: make(map[string]Route, len(cfg.ModelRoutes))}

	for _, route := range cfg.ModelRoutes {
		targets := make([]Target, 0, len(route.Targets))
		for _, target := range route.Targets {
			if _, ok := adapters[target.Provider]; !ok {
				return nil, fmt.Errorf("model route %q references unknown provider %q", route.ExternalModel, target.Provider)
			}
			targets = append(targets, Target{
				ProviderName: target.Provider,
				BackendModel: target.BackendModel,
			})
		}
		r.routes[route.ExternalModel] = Route{
			ExternalModel: route.ExternalModel,
			Targets:       targets,
		}
	}

	return r, nil
}

// Resolve returns the configured route for an external model.
func (r *Router) Resolve(externalModel string) (Route, bool) {
	if r == nil {
		return Route{}, false
	}
	route, ok := r.routes[externalModel]
	return route, ok
}

// ModelInfos returns models in stable order for `/v1/models`.
func (r *Router) ModelInfos() []ModelInfo {
	if r == nil {
		return nil
	}

	keys := make([]string, 0, len(r.routes))
	for model := range r.routes {
		keys = append(keys, model)
	}
	slices.Sort(keys)

	out := make([]ModelInfo, 0, len(keys))
	for _, model := range keys {
		route := r.routes[model]
		owner := ""
		if len(route.Targets) > 0 {
			owner = route.Targets[0].ProviderName
		}
		out = append(out, ModelInfo{
			ID:      model,
			OwnedBy: owner,
		})
	}
	return out
}
