use std::collections::HashMap;

use async_trait::async_trait;

/// Provider name constants — used as keys in the `Providers` map.
/// Matches Go's `ProviderOpenAI`, `ProviderAnthropic`, `ProviderOllama` constants.
pub const PROVIDER_OPENAI: &str = "openai";
pub const PROVIDER_ANTHROPIC: &str = "anthropic";
pub const PROVIDER_OLLAMA: &str = "ollama";

/// Fallback max token limit for providers that require it (e.g., Anthropic).
pub const DEFAULT_MAX_TOKENS: u32 = 1024;

/// Optional parameters for a completion request.
/// Zero values mean "use provider default".
#[derive(Debug, Clone, Default)]
pub struct GenerateParams {
    pub max_tokens: u32,
}

/// Abstracts an LLM API for text completion.
///
/// Go uses an interface; Rust uses an async trait with `Box<dyn Provider>` for dynamic dispatch.
/// The `async_trait` crate is used because Rust's native async trait doesn't support
/// `dyn` dispatch without boxing futures manually.
#[async_trait]
pub trait Provider: Send + Sync {
    async fn generate(
        &self,
        model: &str,
        system_prompt: &str,
        user_prompt: &str,
        params: &GenerateParams,
    ) -> anyhow::Result<String>;
}

/// Maps provider names to their implementations.
///
/// Go uses `map[string]Provider`; Rust uses `HashMap<String, Box<dyn Provider>>`.
pub struct Providers {
    inner: HashMap<String, Box<dyn Provider>>,
}

impl Providers {
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: String, provider: Box<dyn Provider>) {
        self.inner.insert(name, provider);
    }

    /// Returns the Provider and resolved model name for a given model identifier.
    /// Provider-specific prefixes (e.g., "ollama-") are stripped from the returned model name.
    pub fn for_model<'a>(&'a self, model: &'a str) -> anyhow::Result<(&'a dyn Provider, &'a str)> {
        let (name, resolved_model) = resolve_model(model);
        let provider = self
            .inner
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("no {name} provider configured (missing API key?)"))?;
        Ok((provider.as_ref(), resolved_model))
    }
}

impl FromIterator<(String, Box<dyn Provider>)> for Providers {
    fn from_iter<T: IntoIterator<Item = (String, Box<dyn Provider>)>>(iter: T) -> Self {
        Self {
            inner: HashMap::from_iter(iter),
        }
    }
}

/// Maps a model identifier to a provider name and the model name to pass to the API.
///
/// Models prefixed with "ollama-" route to Ollama (prefix stripped),
/// "claude" prefix routes to Anthropic, all others to OpenAI.
/// Uses `str::strip_prefix` — Rust equivalent of Go's `strings.CutPrefix`.
pub fn resolve_model(model: &str) -> (&str, &str) {
    if let Some(m) = model.strip_prefix("ollama-") {
        return (PROVIDER_OLLAMA, m);
    }
    if model.starts_with("claude") {
        return (PROVIDER_ANTHROPIC, model);
    }
    (PROVIDER_OPENAI, model)
}

/// Test utilities — exposed for use by language.rs and main.rs tests.
#[cfg(test)]
pub mod tests {
    use super::*;

    /// Mock provider for testing — mirrors Go's `mockProvider` struct.
    pub struct MockProvider {
        pub response: String,
        pub error: Option<String>,
    }

    impl MockProvider {
        pub fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
                error: None,
            }
        }

        pub fn with_error(msg: &str) -> Self {
            Self {
                response: String::new(),
                error: Some(msg.to_string()),
            }
        }
    }

    #[async_trait]
    impl Provider for MockProvider {
        async fn generate(
            &self,
            _model: &str,
            _system_prompt: &str,
            _user_prompt: &str,
            _params: &GenerateParams,
        ) -> anyhow::Result<String> {
            match &self.error {
                Some(e) => Err(anyhow::anyhow!("{e}")),
                None => Ok(self.response.clone()),
            }
        }
    }

    #[test]
    fn test_resolve_model() {
        let cases = vec![
            ("gpt-4o", PROVIDER_OPENAI, "gpt-4o"),
            ("gpt-5-mini", PROVIDER_OPENAI, "gpt-5-mini"),
            (
                "claude-sonnet-4-5-20250514",
                PROVIDER_ANTHROPIC,
                "claude-sonnet-4-5-20250514",
            ),
            ("claude-haiku-4-5", PROVIDER_ANTHROPIC, "claude-haiku-4-5"),
            ("ollama-qwen3:8b", PROVIDER_OLLAMA, "qwen3:8b"),
            ("ollama-llama3", PROVIDER_OLLAMA, "llama3"),
            ("some-other-model", PROVIDER_OPENAI, "some-other-model"),
        ];

        for (model, want_provider, want_model) in cases {
            let (got_provider, got_model) = resolve_model(model);
            assert_eq!(
                got_provider, want_provider,
                "resolve_model({model:?}) provider"
            );
            assert_eq!(got_model, want_model, "resolve_model({model:?}) model");
        }
    }

    #[test]
    fn test_for_model() {
        let mock = MockProvider::new("ok");
        let providers = Providers::from_iter([(
            PROVIDER_OPENAI.to_string(),
            Box::new(mock) as Box<dyn Provider>,
        )]);

        // OpenAI model should resolve
        assert!(providers.for_model("gpt-4o").is_ok());

        // Claude model should fail (no anthropic provider registered)
        assert!(providers.for_model("claude-haiku-4-5").is_err());

        // Ollama model should fail (no ollama provider registered)
        assert!(providers.for_model("ollama-qwen3:8b").is_err());
    }
}
