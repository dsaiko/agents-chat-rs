use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::provider::{GenerateParams, Provider, DEFAULT_MAX_TOKENS};

/// Anthropic Messages API request body.
/// No official Rust SDK exists, so we use reqwest with manual serde types.
/// This is simpler and more maintainable than depending on an unofficial crate
/// for what amounts to a single HTTP POST call.
#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

/// Anthropic Messages API response body.
#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: String,
}

/// Implements Provider using the Anthropic Messages API via raw HTTP.
///
/// Go uses the official `anthropic-sdk-go`; Rust has no official SDK.
/// We make a direct POST to `https://api.anthropic.com/v1/messages`
/// with `x-api-key` and `anthropic-version` headers.
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
}

impl AnthropicProvider {
    pub fn new(api_key: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.to_string(),
        }
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    async fn generate(
        &self,
        model: &str,
        system_prompt: &str,
        user_prompt: &str,
        params: &GenerateParams,
    ) -> anyhow::Result<String> {
        // Anthropic requires max_tokens — default to DEFAULT_MAX_TOKENS if not set.
        let max_tokens = if params.max_tokens > 0 {
            params.max_tokens
        } else {
            DEFAULT_MAX_TOKENS
        };

        let request = AnthropicRequest {
            model: model.to_string(),
            max_tokens,
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: user_prompt.to_string(),
            }],
            system: if system_prompt.is_empty() {
                None
            } else {
                Some(system_prompt.to_string())
            },
        };

        let resp = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic API error {status}: {body}");
        }

        let response: AnthropicResponse = resp.json().await?;

        // Extract text blocks from response — mirrors Go's loop over `msg.Content`.
        let text: String = response
            .content
            .iter()
            .filter(|block| block.block_type == "text")
            .map(|block| block.text.as_str())
            .collect::<Vec<_>>()
            .join("");

        Ok(text.trim().to_string())
    }
}
