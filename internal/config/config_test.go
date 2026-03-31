package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestPrepareExpandsAndNormalizesRuntimeConfig(t *testing.T) {
	cfg := &Config{
		Providers: []ProviderConfig{{
			Name:              "p",
			Type:              "openai-compatible",
			BaseURL:           "https://example.com/v1/",
			APIKey:            os.ExpandEnv("${TEST_PROVIDER_KEY}"),
			SupportedAPITypes: []string{" Responses ", "CHAT_COMPLETIONS", ""},
		}},
		ModelRoutes: []ModelRoute{{
			ExternalModel: "m",
			Targets: []Target{{
				Provider:            "p",
				BackendModel:        "backend",
				IgnoreRequestFields: []string{" Temperature ", "temperature", "MAX_OUTPUT_TOKENS", ""},
			}},
		}},
	}
	t.Setenv("TEST_PROVIDER_KEY", "from-env")
	cfg.Providers[0].APIKey = os.ExpandEnv("${TEST_PROVIDER_KEY}")

	if err := Prepare(cfg, t.TempDir()); err != nil {
		t.Fatalf("Prepare() error = %v", err)
	}
	if got := cfg.Providers[0].APIKey; got != "from-env" {
		t.Fatalf("api_key = %q", got)
	}
	if got := cfg.Providers[0].BaseURL; got != "https://example.com/v1" {
		t.Fatalf("base_url = %q", got)
	}
	if got := cfg.Providers[0].SupportedAPITypes; len(got) != 2 || got[0] != "responses" || got[1] != "chat_completions" {
		t.Fatalf("supported_api_types = %#v", got)
	}
	if got := cfg.ModelRoutes[0].Targets[0].IgnoreRequestFields; len(got) != 2 || got[0] != "temperature" || got[1] != "max_output_tokens" {
		t.Fatalf("ignore_request_fields = %#v", got)
	}
}

func TestPrepareRejectsDuplicateProviders(t *testing.T) {
	cfg := &Config{
		Providers: []ProviderConfig{
			{Name: "p", Type: "openai-compatible", BaseURL: "https://example.com/v1"},
			{Name: "p", Type: "anthropic-native", BaseURL: "https://api.anthropic.com/v1"},
		},
		ModelRoutes: []ModelRoute{{
			ExternalModel: "m",
			Targets:       []Target{{Provider: "p", BackendModel: "backend"}},
		}},
	}

	if err := Prepare(cfg, t.TempDir()); err == nil || err.Error() != `duplicate provider "p"` {
		t.Fatalf("error = %v", err)
	}
}

func TestPrepareRejectsDuplicateModelRoutes(t *testing.T) {
	cfg := &Config{
		Providers: []ProviderConfig{{Name: "p", Type: "openai-compatible", BaseURL: "https://example.com/v1"}},
		ModelRoutes: []ModelRoute{
			{ExternalModel: "m", Targets: []Target{{Provider: "p", BackendModel: "backend-a"}}},
			{ExternalModel: "m", Targets: []Target{{Provider: "p", BackendModel: "backend-b"}}},
		},
	}

	if err := Prepare(cfg, t.TempDir()); err == nil || err.Error() != `duplicate model route "m"` {
		t.Fatalf("error = %v", err)
	}
}

func TestPrepareDefaultsDatabasePathAndAdminAPIKeys(t *testing.T) {
	baseDir := t.TempDir()
	cfg := &Config{
		AdminAPIKeys: []string{"admin-1", " admin-2 "},
		Providers:    []ProviderConfig{{Name: "p", Type: "openai-compatible", BaseURL: "https://example.com/v1"}},
		ModelRoutes:  []ModelRoute{{ExternalModel: "m", Targets: []Target{{Provider: "p", BackendModel: "backend"}}}},
	}

	if err := Prepare(cfg, baseDir); err != nil {
		t.Fatalf("Prepare() error = %v", err)
	}
	if len(cfg.AdminAPIKeys) != 2 || cfg.AdminAPIKeys[0] != "admin-1" || cfg.AdminAPIKeys[1] != "admin-2" {
		t.Fatalf("admin_api_keys = %#v", cfg.AdminAPIKeys)
	}
	if cfg.DatabasePath != filepath.Join(baseDir, "llmio.db") {
		t.Fatalf("database_path = %q", cfg.DatabasePath)
	}
}

func TestResolveProviderConfigAllowsPlaintextAPIKey(t *testing.T) {
	got, err := ResolveProviderConfig(ProviderConfig{
		Name:   "p",
		APIKey: "secret-value",
	})
	if err != nil {
		t.Fatalf("ResolveProviderConfig() error = %v", err)
	}
	if got.APIKey != "secret-value" {
		t.Fatalf("api_key = %q", got.APIKey)
	}
}

func TestResolveProviderConfigExpandsEnvReference(t *testing.T) {
	t.Setenv("TEST_PROVIDER_KEY", "from-env")

	got, err := ResolveProviderConfig(ProviderConfig{
		Name:   "p",
		APIKey: "${TEST_PROVIDER_KEY}",
	})
	if err != nil {
		t.Fatalf("ResolveProviderConfig() error = %v", err)
	}
	if got.APIKey != "from-env" {
		t.Fatalf("api_key = %q", got.APIKey)
	}
}

func TestResolveProviderConfigRejectsInvalidEnvReference(t *testing.T) {
	_, err := ResolveProviderConfig(ProviderConfig{
		Name:   "p",
		APIKey: "prefix-${TEST_PROVIDER_KEY}",
	})
	if err == nil || err.Error() != `provider "p" api_key must be plaintext or ${ENV_NAME}` {
		t.Fatalf("error = %v", err)
	}
}

func TestResolveProviderConfigRejectsMissingEnvReference(t *testing.T) {
	_, err := ResolveProviderConfig(ProviderConfig{
		Name:   "p",
		APIKey: "${MISSING_PROVIDER_KEY}",
	})
	if err == nil || err.Error() != `provider "p" api_key env "MISSING_PROVIDER_KEY" is not set` {
		t.Fatalf("error = %v", err)
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

	if err := os.WriteFile(filepath.Join(dir, ".env"), []byte("LLMIO_DATABASE_PATH=/tmp/llmio.db\nLLMIO_LOG_LEVEL=debug\n"), 0o644); err != nil {
		t.Fatalf("write .env: %v", err)
	}
	if err := os.Unsetenv("LLMIO_DATABASE_PATH"); err != nil {
		t.Fatalf("unset LLMIO_DATABASE_PATH: %v", err)
	}
	if err := os.Unsetenv("LLMIO_LOG_LEVEL"); err != nil {
		t.Fatalf("unset LLMIO_LOG_LEVEL: %v", err)
	}

	if err := LoadEnv(); err != nil {
		t.Fatalf("LoadEnv() error = %v", err)
	}
	if got := os.Getenv("LLMIO_DATABASE_PATH"); got != "/tmp/llmio.db" {
		t.Fatalf("LLMIO_DATABASE_PATH = %q", got)
	}
	if got := os.Getenv("LLMIO_LOG_LEVEL"); got != "debug" {
		t.Fatalf("LLMIO_LOG_LEVEL = %q", got)
	}
}
