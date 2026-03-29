package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadExpandsEnvFromDotEnv(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "llmio.json")

	if err := os.WriteFile(filepath.Join(dir, ".env"), []byte("TEST_PROVIDER_KEY=from-dotenv\n"), 0o644); err != nil {
		t.Fatalf("write .env: %v", err)
	}
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

	cfg, err := Load(configPath)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if got := cfg.Providers[0].APIKey; got != "from-dotenv" {
		t.Fatalf("api_key = %q", got)
	}
}

func TestLoadDotEnvDoesNotOverrideExistingEnv(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "llmio.json")

	if err := os.WriteFile(filepath.Join(dir, ".env"), []byte("TEST_PROVIDER_KEY=from-dotenv\n"), 0o644); err != nil {
		t.Fatalf("write .env: %v", err)
	}
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
