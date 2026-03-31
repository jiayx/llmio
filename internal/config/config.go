package config

import (
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
	Pricing      []PricingRule    `json:"pricing,omitempty"`
	ModelRoutes  []ModelRoute     `json:"model_routes"`
}

type RuntimeConfig struct {
	Providers   []ProviderConfig `json:"providers"`
	Pricing     []PricingRule    `json:"pricing,omitempty"`
	ModelRoutes []ModelRoute     `json:"model_routes"`
}

type PricingRule struct {
	Provider                      string  `json:"provider"`
	BackendModel                  string  `json:"backend_model"`
	Scheme                        string  `json:"scheme,omitempty"`
	InputPer1MTokens              float64 `json:"input_per_1m_tokens"`
	CachedInputPer1MTokens        float64 `json:"cached_input_per_1m_tokens,omitempty"`
	CacheReadInputPer1MTokens     float64 `json:"cache_read_input_per_1m_tokens,omitempty"`
	CacheCreationInputPer1MTokens float64 `json:"cache_creation_input_per_1m_tokens,omitempty"`
	OutputPer1MTokens             float64 `json:"output_per_1m_tokens"`
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
	ExternalModel string   `json:"external_model"`
	Targets       []Target `json:"targets"`
}

// Target defines a single backend provider/model pair for routing.
type Target struct {
	Provider            string   `json:"provider"`
	BackendModel        string   `json:"backend_model"`
	IgnoreRequestFields []string `json:"ignore_request_fields,omitempty"`
}

func ResolveProviderConfig(cfg ProviderConfig) (ProviderConfig, error) {
	cfg.APIKey = strings.TrimSpace(cfg.APIKey)
	if cfg.APIKey == "" {
		return cfg, nil
	}
	if !strings.Contains(cfg.APIKey, "${") {
		return cfg, nil
	}

	envName, ok := parseEnvReference(cfg.APIKey)
	if !ok {
		return ProviderConfig{}, fmt.Errorf("provider %q api_key must be plaintext or ${ENV_NAME}", cfg.Name)
	}
	value, exists := os.LookupEnv(envName)
	if !exists {
		return ProviderConfig{}, fmt.Errorf("provider %q api_key env %q is not set", cfg.Name, envName)
	}
	cfg.APIKey = value
	return cfg, nil
}

func Prepare(cfg *Config, baseDir string) error {
	if cfg == nil {
		return fmt.Errorf("config is required")
	}
	if err := PrepareRuntimeConfig(&RuntimeConfig{
		Providers:   cfg.Providers,
		Pricing:     cfg.Pricing,
		ModelRoutes: cfg.ModelRoutes,
	}); err != nil {
		return err
	}

	for i := range cfg.AdminAPIKeys {
		cfg.AdminAPIKeys[i] = strings.TrimSpace(cfg.AdminAPIKeys[i])
	}
	cfg.DatabasePath = strings.TrimSpace(cfg.DatabasePath)
	if cfg.DatabasePath == "" {
		cfg.DatabasePath = filepath.Join(baseDir, "llmio.db")
	} else if !filepath.IsAbs(cfg.DatabasePath) {
		cfg.DatabasePath = filepath.Join(baseDir, cfg.DatabasePath)
	}

	return nil
}

func PrepareRuntimeConfig(cfg *RuntimeConfig) error {
	if cfg == nil {
		return fmt.Errorf("runtime config is required")
	}
	if len(cfg.Providers) == 0 {
		return fmt.Errorf("config.providers is required")
	}
	if len(cfg.ModelRoutes) == 0 {
		return fmt.Errorf("config.model_routes is required")
	}

	providerNames := make(map[string]struct{}, len(cfg.Providers))
	for i := range cfg.Providers {
		p := &cfg.Providers[i]
		p.Name = strings.TrimSpace(p.Name)
		if p.Name == "" {
			return fmt.Errorf("config.providers[%d].name is required", i)
		}
		if _, exists := providerNames[p.Name]; exists {
			return fmt.Errorf("duplicate provider %q", p.Name)
		}
		providerNames[p.Name] = struct{}{}
		p.Type = strings.ToLower(strings.TrimSpace(p.Type))
		if p.Type == "" {
			p.Type = "openai-compatible"
		}
		p.BaseURL = strings.TrimRight(strings.TrimSpace(p.BaseURL), "/")
		if p.BaseURL == "" {
			return fmt.Errorf("provider %q base_url is required", p.Name)
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

	for i := range cfg.ModelRoutes {
		route := &cfg.ModelRoutes[i]
		route.ExternalModel = strings.TrimSpace(route.ExternalModel)
		if route.ExternalModel == "" {
			return fmt.Errorf("config.model_routes[%d].external_model is required", i)
		}
		if len(route.Targets) == 0 {
			return fmt.Errorf("model route %q requires at least one target", route.ExternalModel)
		}
		for j := range route.Targets {
			target := &route.Targets[j]
			target.Provider = strings.TrimSpace(target.Provider)
			target.BackendModel = strings.TrimSpace(target.BackendModel)
			target.IgnoreRequestFields = normalizeRequestFields(target.IgnoreRequestFields)
			if target.Provider == "" {
				return fmt.Errorf("model route %q target[%d] provider is required", route.ExternalModel, j)
			}
			if target.BackendModel == "" {
				return fmt.Errorf("model route %q target[%d] backend_model is required", route.ExternalModel, j)
			}
		}
	}

	for i := range cfg.Pricing {
		rule := &cfg.Pricing[i]
		rule.Provider = strings.TrimSpace(rule.Provider)
		rule.BackendModel = strings.TrimSpace(rule.BackendModel)
		rule.Scheme = strings.ToLower(strings.TrimSpace(rule.Scheme))
		if rule.Provider == "" {
			return fmt.Errorf("config.pricing[%d].provider is required", i)
		}
		if _, ok := providerNames[rule.Provider]; !ok {
			return fmt.Errorf("config.pricing[%d].provider %q is not defined", i, rule.Provider)
		}
		if rule.BackendModel == "" {
			return fmt.Errorf("config.pricing[%d].backend_model is required", i)
		}
		if rule.Scheme == "" {
			rule.Scheme = "generic"
		}
	}

	externalModels := make(map[string]struct{}, len(cfg.ModelRoutes))
	for _, route := range cfg.ModelRoutes {
		if _, exists := externalModels[route.ExternalModel]; exists {
			return fmt.Errorf("duplicate model route %q", route.ExternalModel)
		}
		externalModels[route.ExternalModel] = struct{}{}
	}
	return nil
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

func parseEnvReference(value string) (string, bool) {
	if !strings.HasPrefix(value, "${") || !strings.HasSuffix(value, "}") {
		return "", false
	}
	name := strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(value, "${"), "}"))
	if name == "" {
		return "", false
	}
	for _, r := range name {
		if (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '_' {
			continue
		}
		return "", false
	}
	return name, true
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
