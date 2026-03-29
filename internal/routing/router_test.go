package routing

import (
	"context"
	"testing"

	"github.com/jiayx/llmio/internal/config"
	"github.com/jiayx/llmio/internal/llm"
	providerapi "github.com/jiayx/llmio/internal/providers/api"
)

func TestRouterResolveAndModelInfos(t *testing.T) {
	router, err := New(&config.Config{
		ModelRoutes: []config.ModelRoute{
			{
				ExternalModel: "b-model",
				Targets:       []config.Target{{Provider: "p1", BackendModel: "backend-b"}},
			},
			{
				ExternalModel: "a-model",
				Targets:       []config.Target{{Provider: "p2", BackendModel: "backend-a"}},
			},
		},
	}, map[string]providerapi.ProviderAdapter{
		"p1": routingStubProvider{},
		"p2": routingStubProvider{},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	route, ok := router.Resolve("a-model")
	if !ok {
		t.Fatalf("Resolve() did not find model")
	}
	if len(route.Targets) != 1 || route.Targets[0].ProviderName != "p2" {
		t.Fatalf("route = %#v", route)
	}

	infos := router.ModelInfos()
	if len(infos) != 2 {
		t.Fatalf("infos = %#v", infos)
	}
	if infos[0].ID != "a-model" || infos[1].ID != "b-model" {
		t.Fatalf("infos = %#v", infos)
	}
}

type routingStubProvider struct{}

func (routingStubProvider) Name() string { return "stub" }

func (routingStubProvider) Chat(context.Context, llm.ChatRequest) (*llm.ChatResponse, error) {
	return nil, nil
}

func (routingStubProvider) ChatStream(context.Context, llm.ChatRequest) (*providerapi.StreamReader, error) {
	return nil, nil
}
