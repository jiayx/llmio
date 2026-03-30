package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Config is the top-level gateway configuration loaded from disk.
type Config struct {
	Listen       string           `json:"listen"`
	AdminAPIKeys []string         `json:"admin_api_keys"`
	DatabasePath string           `json:"database_path"`
	Providers    []ProviderConfig `json:"providers"`
	ModelRoutes  []ModelRoute     `json:"model_routes"`
}

// ProviderConfig defines a backend provider endpoint and its auth settings.
type ProviderConfig struct {
	Name              string            `json:"name"`
	Type              string            `json:"type"`
	BaseURL           string            `json:"base_url"`
	APIKey            string            `json:"api_key"`
	Headers           map[string]string `json:"headers"`
	ModelsPath        string            `json:"models_path"`
	SupportedAPITypes []string          `json:"supported_api_types"`
}

// ModelRoute maps one external model name to one or more backend targets.
type ModelRoute struct {
	ExternalModel       string   `json:"external_model"`
	Provider            string   `json:"provider"`
	BackendModel        string   `json:"backend_model"`
	IgnoreRequestFields []string `json:"ignore_request_fields,omitempty"`
	Targets             []Target `json:"targets"`
}

// Target defines a single backend provider/model pair for routing.
type Target struct {
	Provider            string   `json:"provider"`
	BackendModel        string   `json:"backend_model"`
	IgnoreRequestFields []string `json:"ignore_request_fields,omitempty"`
}

// Load reads, expands, validates, and normalizes a gateway config file.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}

	data = []byte(os.ExpandEnv(string(data)))

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("decode %s: %w", path, err)
	}

	if len(cfg.Providers) == 0 {
		return nil, fmt.Errorf("config.providers is required")
	}
	if len(cfg.ModelRoutes) == 0 {
		return nil, fmt.Errorf("config.model_routes is required")
	}

	providerNames := make(map[string]struct{}, len(cfg.Providers))
	for i := range cfg.Providers {
		p := &cfg.Providers[i]
		p.Name = strings.TrimSpace(p.Name)
		if p.Name == "" {
			return nil, fmt.Errorf("config.providers[%d].name is required", i)
		}
		if _, exists := providerNames[p.Name]; exists {
			return nil, fmt.Errorf("duplicate provider %q", p.Name)
		}
		providerNames[p.Name] = struct{}{}
		p.Type = strings.ToLower(strings.TrimSpace(p.Type))
		if p.Type == "" {
			p.Type = "openai-compatible"
		}
		p.BaseURL = strings.TrimRight(strings.TrimSpace(p.BaseURL), "/")
		if p.BaseURL == "" {
			return nil, fmt.Errorf("provider %q base_url is required", p.Name)
		}
		if p.ModelsPath == "" {
			p.ModelsPath = "/models"
		}
		apiTypes := make([]string, 0, len(p.SupportedAPITypes))
		for _, apiType := range p.SupportedAPITypes {
			apiType = strings.ToLower(strings.TrimSpace(apiType))
			if apiType == "" {
				continue
			}
			apiTypes = append(apiTypes, apiType)
		}
		p.SupportedAPITypes = apiTypes
	}

	for i := range cfg.AdminAPIKeys {
		cfg.AdminAPIKeys[i] = strings.TrimSpace(cfg.AdminAPIKeys[i])
	}
	cfg.DatabasePath = strings.TrimSpace(cfg.DatabasePath)
	if cfg.DatabasePath == "" {
		cfg.DatabasePath = filepath.Join(filepath.Dir(path), "llmio.db")
	} else if !filepath.IsAbs(cfg.DatabasePath) {
		cfg.DatabasePath = filepath.Join(filepath.Dir(path), cfg.DatabasePath)
	}

	for i := range cfg.ModelRoutes {
		route := &cfg.ModelRoutes[i]
		route.ExternalModel = strings.TrimSpace(route.ExternalModel)
		if route.ExternalModel == "" {
			return nil, fmt.Errorf("config.model_routes[%d].external_model is required", i)
		}
		if len(route.Targets) == 0 {
			route.Provider = strings.TrimSpace(route.Provider)
			route.BackendModel = strings.TrimSpace(route.BackendModel)
			route.IgnoreRequestFields = normalizeRequestFields(route.IgnoreRequestFields)
			if route.Provider == "" || route.BackendModel == "" {
				return nil, fmt.Errorf("model route %q requires provider/backend_model or targets", route.ExternalModel)
			}
			route.Targets = []Target{{
				Provider:            route.Provider,
				BackendModel:        route.BackendModel,
				IgnoreRequestFields: route.IgnoreRequestFields,
			}}
		}
		for j := range route.Targets {
			target := &route.Targets[j]
			target.Provider = strings.TrimSpace(target.Provider)
			target.BackendModel = strings.TrimSpace(target.BackendModel)
			target.IgnoreRequestFields = normalizeRequestFields(target.IgnoreRequestFields)
			if target.Provider == "" {
				return nil, fmt.Errorf("model route %q target[%d] provider is required", route.ExternalModel, j)
			}
			if target.BackendModel == "" {
				return nil, fmt.Errorf("model route %q target[%d] backend_model is required", route.ExternalModel, j)
			}
		}
	}

	externalModels := make(map[string]struct{}, len(cfg.ModelRoutes))
	for _, route := range cfg.ModelRoutes {
		if _, exists := externalModels[route.ExternalModel]; exists {
			return nil, fmt.Errorf("duplicate model route %q", route.ExternalModel)
		}
		externalModels[route.ExternalModel] = struct{}{}
	}

	return &cfg, nil
}

func normalizeRequestFields(fields []string) []string {
	if len(fields) == 0 {
		return nil
	}
	out := make([]string, 0, len(fields))
	seen := make(map[string]struct{}, len(fields))
	for _, field := range fields {
		field = strings.ToLower(strings.TrimSpace(field))
		if field == "" {
			continue
		}
		if _, ok := seen[field]; ok {
			continue
		}
		seen[field] = struct{}{}
		out = append(out, field)
	}
	return out
}

// LoadEnv loads the process .env file before any other config resolution.
func LoadEnv() error {
	return loadDotEnv(".env")
}

func loadDotEnv(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("read %s: %w", path, err)
	}

	lines := strings.Split(string(data), "\n")
	for i, raw := range lines {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "export ") {
			line = strings.TrimSpace(strings.TrimPrefix(line, "export "))
		}

		key, value, ok := strings.Cut(line, "=")
		if !ok {
			return fmt.Errorf("parse %s:%d: missing '='", path, i+1)
		}
		key = strings.TrimSpace(key)
		if key == "" {
			return fmt.Errorf("parse %s:%d: empty key", path, i+1)
		}
		if _, exists := os.LookupEnv(key); exists {
			continue
		}

		value = strings.TrimSpace(value)
		if len(value) >= 2 {
			if (value[0] == '"' && value[len(value)-1] == '"') || (value[0] == '\'' && value[len(value)-1] == '\'') {
				unquoted, unquoteErr := strconv.Unquote(`"` + strings.ReplaceAll(value[1:len(value)-1], `"`, `\"`) + `"`)
				if value[0] == '"' {
					unquoted, unquoteErr = strconv.Unquote(value)
				}
				if unquoteErr != nil {
					return fmt.Errorf("parse %s:%d: %w", path, i+1, unquoteErr)
				}
				value = unquoted
			}
		}

		if err := os.Setenv(key, value); err != nil {
			return fmt.Errorf("set %s from %s:%d: %w", key, path, i+1, err)
		}
	}

	return nil
}
