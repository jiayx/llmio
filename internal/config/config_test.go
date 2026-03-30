package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadExpandsEnvFromDotEnv(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "llmio.json")

	if err := os.WriteFile(configPath, []byte(`{
		"providers": [{"name":"p","type":"openai-compatible","base_url":"https://example.com/v1","api_key":"${TEST_PROVIDER_KEY}"}],
		"model_routes": [{"external_model":"m","targets":[{"provider":"p","backend_model":"backend"}]}]
	}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	t.Setenv("TEST_PROVIDER_KEY", "")
	if err := os.Unsetenv("TEST_PROVIDER_KEY"); err != nil {
		t.Fatalf("unset env: %v", err)
	}
	t.Setenv("TEST_PROVIDER_KEY", "from-env")

	cfg, err := Load(configPath)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if got := cfg.Providers[0].APIKey; got != "from-env" {
		t.Fatalf("api_key = %q", got)
	}
}

func TestLoadDoesNotMutateEnvResolution(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "llmio.json")

	if err := os.WriteFile(configPath, []byte(`{
		"providers": [{"name":"p","type":"openai-compatible","base_url":"https://example.com/v1","api_key":"${TEST_PROVIDER_KEY}"}],
		"model_routes": [{"external_model":"m","targets":[{"provider":"p","backend_model":"backend"}]}]
	}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	t.Setenv("TEST_PROVIDER_KEY", "from-env")

	cfg, err := Load(configPath)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if got := cfg.Providers[0].APIKey; got != "from-env" {
		t.Fatalf("api_key = %q", got)
	}
}

func TestLoadNormalizesSupportedAPITypes(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "llmio.json")

	if err := os.WriteFile(configPath, []byte(`{
		"providers": [{
			"name":"p",
			"type":"openai-compatible",
			"base_url":"https://example.com/v1",
			"supported_api_types":[" Responses ","CHAT_COMPLETIONS",""]
		}],
		"model_routes": [{"external_model":"m","targets":[{"provider":"p","backend_model":"backend"}]}]
	}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg, err := Load(configPath)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	got := cfg.Providers[0].SupportedAPITypes
	if len(got) != 2 {
		t.Fatalf("supported_api_types = %#v", got)
	}
	if got[0] != "responses" || got[1] != "chat_completions" {
		t.Fatalf("supported_api_types = %#v", got)
	}
}

func TestLoadNormalizesIgnoreRequestFields(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "llmio.json")

	if err := os.WriteFile(configPath, []byte(`{
		"providers": [{"name":"p","type":"openai-compatible","base_url":"https://example.com/v1"}],
		"model_routes": [{
			"external_model":"m",
			"targets":[{
				"provider":"p",
				"backend_model":"backend",
				"ignore_request_fields":[" Temperature ","temperature","MAX_OUTPUT_TOKENS",""]
			}]
		}]
	}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg, err := Load(configPath)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	got := cfg.ModelRoutes[0].Targets[0].IgnoreRequestFields
	if len(got) != 2 {
		t.Fatalf("ignore_request_fields = %#v", got)
	}
	if got[0] != "temperature" || got[1] != "max_output_tokens" {
		t.Fatalf("ignore_request_fields = %#v", got)
	}
}

func TestLoadRejectsDuplicateProviders(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "llmio.json")

	if err := os.WriteFile(configPath, []byte(`{
		"providers": [
			{"name":"p","type":"openai-compatible","base_url":"https://example.com/v1"},
			{"name":"p","type":"anthropic-native","base_url":"https://api.anthropic.com/v1"}
		],
		"model_routes": [{"external_model":"m","targets":[{"provider":"p","backend_model":"backend"}]}]
	}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	_, err := Load(configPath)
	if err == nil || err.Error() != `duplicate provider "p"` {
		t.Fatalf("error = %v", err)
	}
}

func TestLoadRejectsDuplicateModelRoutes(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "llmio.json")

	if err := os.WriteFile(configPath, []byte(`{
		"providers": [{"name":"p","type":"openai-compatible","base_url":"https://example.com/v1"}],
		"model_routes": [
			{"external_model":"m","targets":[{"provider":"p","backend_model":"backend-a"}]},
			{"external_model":"m","targets":[{"provider":"p","backend_model":"backend-b"}]}
		]
	}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	_, err := Load(configPath)
	if err == nil || err.Error() != `duplicate model route "m"` {
		t.Fatalf("error = %v", err)
	}
}

func TestLoadDefaultsDatabasePath(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "llmio.json")

	if err := os.WriteFile(configPath, []byte(`{
		"admin_api_keys": ["admin-1"],
		"providers": [{"name":"p","type":"openai-compatible","base_url":"https://example.com/v1"}],
		"model_routes": [{"external_model":"m","targets":[{"provider":"p","backend_model":"backend"}]}]
	}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg, err := Load(configPath)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if len(cfg.AdminAPIKeys) != 1 || cfg.AdminAPIKeys[0] != "admin-1" {
		t.Fatalf("admin_api_keys = %#v", cfg.AdminAPIKeys)
	}
	if cfg.DatabasePath != filepath.Join(dir, "llmio.db") {
		t.Fatalf("database_path = %q", cfg.DatabasePath)
	}
}

func TestLoadPreservesAdminAPIKeys(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "llmio.json")

	if err := os.WriteFile(configPath, []byte(`{
		"admin_api_keys": ["admin-1", " admin-2 "],
		"providers": [{"name":"p","type":"openai-compatible","base_url":"https://example.com/v1"}],
		"model_routes": [{"external_model":"m","targets":[{"provider":"p","backend_model":"backend"}]}]
	}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	cfg, err := Load(configPath)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if len(cfg.AdminAPIKeys) != 2 || cfg.AdminAPIKeys[0] != "admin-1" || cfg.AdminAPIKeys[1] != "admin-2" {
		t.Fatalf("admin_api_keys = %#v", cfg.AdminAPIKeys)
	}
}

func TestLoadEnvReadsDotEnvFromWorkingDirectory(t *testing.T) {
	dir := t.TempDir()
	oldwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd() error = %v", err)
	}
	defer func() {
		if chdirErr := os.Chdir(oldwd); chdirErr != nil {
			t.Fatalf("restore wd: %v", chdirErr)
		}
	}()
	if err := os.Chdir(dir); err != nil {
		t.Fatalf("Chdir() error = %v", err)
	}

	if err := os.WriteFile(filepath.Join(dir, ".env"), []byte("LLMIO_CONFIG=/tmp/llmio.custom.json\nLLMIO_LOG_LEVEL=debug\n"), 0o644); err != nil {
		t.Fatalf("write .env: %v", err)
	}
	if err := os.Unsetenv("LLMIO_CONFIG"); err != nil {
		t.Fatalf("unset LLMIO_CONFIG: %v", err)
	}
	if err := os.Unsetenv("LLMIO_LOG_LEVEL"); err != nil {
		t.Fatalf("unset LLMIO_LOG_LEVEL: %v", err)
	}

	if err := LoadEnv(); err != nil {
		t.Fatalf("LoadEnv() error = %v", err)
	}
	if got := os.Getenv("LLMIO_CONFIG"); got != "/tmp/llmio.custom.json" {
		t.Fatalf("LLMIO_CONFIG = %q", got)
	}
	if got := os.Getenv("LLMIO_LOG_LEVEL"); got != "debug" {
		t.Fatalf("LLMIO_LOG_LEVEL = %q", got)
	}
}
