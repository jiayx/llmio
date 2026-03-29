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
	Listen      string           `json:"listen"`
	APIKeys     []string         `json:"api_keys"`
	Providers   []ProviderConfig `json:"providers"`
	ModelRoutes []ModelRoute     `json:"model_routes"`
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
	Provider      string   `json:"provider"`
	BackendModel  string   `json:"backend_model"`
	Targets       []Target `json:"targets"`
}

// Target defines a single backend provider/model pair for routing.
type Target struct {
	Provider     string `json:"provider"`
	BackendModel string `json:"backend_model"`
}

// Load reads, expands, validates, and normalizes a gateway config file.
func Load(path string) (*Config, error) {
	if err := loadDotEnvForConfig(path); err != nil {
		return nil, err
	}

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

	for i := range cfg.Providers {
		p := &cfg.Providers[i]
		if p.Type == "" {
			p.Type = "openai-compatible"
		}
		if p.ModelsPath == "" {
			p.ModelsPath = "/models"
		}
		p.BaseURL = strings.TrimRight(p.BaseURL, "/")
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

	for i := range cfg.APIKeys {
		cfg.APIKeys[i] = strings.TrimSpace(cfg.APIKeys[i])
	}

	for i := range cfg.ModelRoutes {
		route := &cfg.ModelRoutes[i]
		if len(route.Targets) == 0 {
			if route.Provider == "" || route.BackendModel == "" {
				return nil, fmt.Errorf("model route %q requires provider/backend_model or targets", route.ExternalModel)
			}
			route.Targets = []Target{{
				Provider:     route.Provider,
				BackendModel: route.BackendModel,
			}}
		}
	}

	return &cfg, nil
}

func loadDotEnvForConfig(configPath string) error {
	seen := make(map[string]struct{}, 2)
	paths := []string{filepath.Join(filepath.Dir(configPath), ".env"), ".env"}
	for _, path := range paths {
		if _, ok := seen[path]; ok {
			continue
		}
		seen[path] = struct{}{}
		if err := loadDotEnv(path); err != nil {
			return err
		}
	}
	return nil
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
