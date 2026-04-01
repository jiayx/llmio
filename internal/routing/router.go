package routing

import (
	"fmt"
	"slices"

	"github.com/jiayx/llmio/internal/config"
	providerapi "github.com/jiayx/llmio/internal/providers/api"
)

// Target is one candidate backend for an external model.
type Target struct {
	ProviderName        string
	BackendModel        string
	IgnoreRequestFields []string
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

// New constructs a static router from runtime config.
func New(cfg *config.RuntimeConfig, adapters map[string]providerapi.ProviderAdapter) (*Router, error) {
	r := &Router{routes: make(map[string]Route, len(cfg.Models))}
	targetsByName := make(map[string]config.TargetConfig, len(cfg.Targets))
	for _, target := range cfg.Targets {
		targetsByName[target.Name] = target
	}

	for _, model := range cfg.Models {
		targets := make([]Target, 0, len(model.Targets))
		for _, targetName := range model.Targets {
			target, ok := targetsByName[targetName]
			if !ok {
				return nil, fmt.Errorf("model %q references unknown target %q", model.Name, targetName)
			}
			if _, ok := adapters[target.Provider]; !ok {
				return nil, fmt.Errorf("target %q references unknown provider %q", target.Name, target.Provider)
			}
			targets = append(targets, Target{
				ProviderName:        target.Provider,
				BackendModel:        target.BackendModel,
				IgnoreRequestFields: append([]string(nil), target.IgnoreRequestFields...),
			})
		}
		r.routes[model.Name] = Route{
			ExternalModel: model.Name,
			Targets:       targets,
		}
	}

	return r, nil
}

// Resolve returns the configured route for an external model.
func (r *Router) Resolve(externalModel string) (Route, bool) {
	route, ok := r.routes[externalModel]
	return route, ok
}

// ModelInfos returns models in stable order for `/v1/models`.
func (r *Router) ModelInfos() []ModelInfo {
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
