use async_trait::async_trait;

use crate::provider::{GenerateParams, Provider};
use crate::provider_openai::OpenAiCompatibleProvider;

/// Implements Provider using the OpenRouter API (OpenAI-compatible).
pub struct OpenRouterProvider {
    inner: OpenAiCompatibleProvider,
}

impl OpenRouterProvider {
    pub fn new(api_key: &str) -> Self {
        Self {
            inner: OpenAiCompatibleProvider::new(
                api_key,
                "https://openrouter.ai/api/v1",
                "openrouter",
            ),
        }
    }
}

#[async_trait]
impl Provider for OpenRouterProvider {
    async fn generate(
        &self,
        model: &str,
        system_prompt: &str,
        user_prompt: &str,
        params: &GenerateParams,
    ) -> anyhow::Result<String> {
        self.inner
            .generate(model, system_prompt, user_prompt, params)
            .await
    }

    async fn health_check(&self) -> anyhow::Result<()> {
        self.inner.health_check().await
    }
}
