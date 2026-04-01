package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// AppConfig contains process startup settings that are not hot-reloaded.
type AppConfig struct {
	Listen       string   `json:"listen"`
	AdminAPIKeys []string `json:"admin_api_keys"`
	DatabasePath string   `json:"database_path"`
}

type RuntimeConfig struct {
	Providers []ProviderConfig `json:"providers"`
	Targets   []TargetConfig   `json:"targets,omitempty"`
	Pricing   []PricingRule    `json:"pricing,omitempty"`
	Models    []ModelConfig    `json:"models,omitempty"`
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

// TargetConfig defines one named backend target bound to a provider and backend model.
type TargetConfig struct {
	Name                string   `json:"name"`
	Provider            string   `json:"provider"`
	BackendModel        string   `json:"backend_model"`
	IgnoreRequestFields []string `json:"ignore_request_fields,omitempty"`
}

// ModelConfig maps one external model name to one or more named backend targets.
type ModelConfig struct {
	Name    string   `json:"name"`
	Targets []string `json:"targets"`
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

func normalizeRuntimeConfig(cfg *RuntimeConfig) error {
	if cfg == nil {
		return fmt.Errorf("runtime config is required")
	}
	for i := range cfg.Providers {
		p := &cfg.Providers[i]
		p.Name = strings.TrimSpace(p.Name)
		p.Type = strings.ToLower(strings.TrimSpace(p.Type))
		if p.Type == "" {
			p.Type = "openai-compatible"
		}
		p.BaseURL = strings.TrimRight(strings.TrimSpace(p.BaseURL), "/")
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

	for i := range cfg.Targets {
		target := &cfg.Targets[i]
		target.Name = strings.TrimSpace(target.Name)
		target.Provider = strings.TrimSpace(target.Provider)
		target.BackendModel = strings.TrimSpace(target.BackendModel)
		target.IgnoreRequestFields = normalizeRequestFields(target.IgnoreRequestFields)
	}

	for i := range cfg.Models {
		model := &cfg.Models[i]
		model.Name = strings.TrimSpace(model.Name)
		for j := range model.Targets {
			model.Targets[j] = strings.TrimSpace(model.Targets[j])
		}
	}

	for i := range cfg.Pricing {
		rule := &cfg.Pricing[i]
		rule.Provider = strings.TrimSpace(rule.Provider)
		rule.BackendModel = strings.TrimSpace(rule.BackendModel)
		rule.Scheme = strings.ToLower(strings.TrimSpace(rule.Scheme))
		if rule.Scheme == "" {
			rule.Scheme = "generic"
		}
	}

	return nil
}

func validateRuntimeConfig(cfg *RuntimeConfig) error {
	if cfg == nil {
		return fmt.Errorf("runtime config is required")
	}
	if len(cfg.Providers) == 0 {
		return fmt.Errorf("config.providers is required")
	}
	if len(cfg.Targets) == 0 {
		return fmt.Errorf("config.targets is required")
	}
	if len(cfg.Models) == 0 {
		return fmt.Errorf("config.models is required")
	}

	providerNames := make(map[string]struct{}, len(cfg.Providers))
	for i := range cfg.Providers {
		p := &cfg.Providers[i]
		if p.Name == "" {
			return fmt.Errorf("config.providers[%d].name is required", i)
		}
		if _, exists := providerNames[p.Name]; exists {
			return fmt.Errorf("duplicate provider %q", p.Name)
		}
		providerNames[p.Name] = struct{}{}
		if p.BaseURL == "" {
			return fmt.Errorf("provider %q base_url is required", p.Name)
		}
	}

	targetNames := make(map[string]struct{}, len(cfg.Targets))
	for i := range cfg.Targets {
		target := &cfg.Targets[i]
		if target.Name == "" {
			return fmt.Errorf("config.targets[%d].name is required", i)
		}
		if _, exists := targetNames[target.Name]; exists {
			return fmt.Errorf("duplicate target %q", target.Name)
		}
		targetNames[target.Name] = struct{}{}
		if target.Provider == "" {
			return fmt.Errorf("config.targets[%d].provider is required", i)
		}
		if _, ok := providerNames[target.Provider]; !ok {
			return fmt.Errorf("config.targets[%d].provider %q is not defined", i, target.Provider)
		}
		if target.BackendModel == "" {
			return fmt.Errorf("config.targets[%d].backend_model is required", i)
		}
	}

	modelNames := make(map[string]struct{}, len(cfg.Models))
	for i := range cfg.Models {
		model := &cfg.Models[i]
		if model.Name == "" {
			return fmt.Errorf("config.models[%d].name is required", i)
		}
		if _, exists := modelNames[model.Name]; exists {
			return fmt.Errorf("duplicate model %q", model.Name)
		}
		modelNames[model.Name] = struct{}{}
		if len(model.Targets) == 0 {
			return fmt.Errorf("model %q requires at least one target", model.Name)
		}
		for j := range model.Targets {
			if model.Targets[j] == "" {
				return fmt.Errorf("model %q target[%d] is required", model.Name, j)
			}
			if _, ok := targetNames[model.Targets[j]]; !ok {
				return fmt.Errorf("model %q references unknown target %q", model.Name, model.Targets[j])
			}
		}
	}

	for i := range cfg.Pricing {
		rule := &cfg.Pricing[i]
		if rule.Provider == "" {
			return fmt.Errorf("config.pricing[%d].provider is required", i)
		}
		if _, ok := providerNames[rule.Provider]; !ok {
			return fmt.Errorf("config.pricing[%d].provider %q is not defined", i, rule.Provider)
		}
		if rule.BackendModel == "" {
			return fmt.Errorf("config.pricing[%d].backend_model is required", i)
		}
	}

	return nil
}

func PrepareRuntimeConfig(cfg *RuntimeConfig) error {
	if cfg == nil {
		return fmt.Errorf("runtime config is required")
	}
	if err := normalizeRuntimeConfig(cfg); err != nil {
		return err
	}
	return validateRuntimeConfig(cfg)
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
