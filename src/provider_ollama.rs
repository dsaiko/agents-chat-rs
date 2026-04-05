use async_trait::async_trait;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::generation::chat::ChatMessage;
use ollama_rs::models::ModelOptions;
use ollama_rs::Ollama;

use crate::provider::{GenerateParams, Provider};

/// Implements Provider using the Ollama chat API.
///
/// Go uses `ollama/api.Client` with `client.Chat()`; Rust uses `ollama-rs`'s
/// `Ollama` with `send_chat_messages()`. Both use non-streaming mode.
pub struct OllamaProvider {
    client: Ollama,
}

impl OllamaProvider {
    /// Creates a new Ollama provider.
    /// Uses OLLAMA_HOST env var if set, otherwise defaults to localhost:11434.
    pub fn new() -> Self {
        // ollama-rs Ollama::default() connects to 127.0.0.1:11434.
        // For custom host, we check OLLAMA_HOST env var to match Go's behavior
        // (Go's api.ClientFromEnvironment() reads OLLAMA_HOST).
        let client = match std::env::var("OLLAMA_HOST") {
            Ok(host) => {
                // Try to parse as a full URL first, fall back to host:port
                Ollama::try_new(host).unwrap_or_default()
            }
            Err(_) => Ollama::default(),
        };

        Self { client }
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    async fn generate(
        &self,
        model: &str,
        system_prompt: &str,
        user_prompt: &str,
        params: &GenerateParams,
    ) -> anyhow::Result<String> {
        let mut messages = Vec::new();
        if !system_prompt.is_empty() {
            messages.push(ChatMessage::system(system_prompt.to_string()));
        }
        messages.push(ChatMessage::user(user_prompt.to_string()));

        let mut request = ChatMessageRequest::new(model.to_string(), messages);

        // Go sets `Options: map[string]any{"num_predict": cp.MaxTokens}`.
        // ollama-rs uses a typed `ModelOptions` builder instead.
        if params.max_tokens > 0 {
            request = request.options(
                ModelOptions::default().num_predict(params.max_tokens as i32),
            );
        }

        let response = self.client.send_chat_messages(request).await?;
        Ok(response.message.content.trim().to_string())
    }
}
