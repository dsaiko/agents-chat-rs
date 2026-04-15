use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::provider::{DEFAULT_MAX_TOKENS, GenerateParams, Provider};

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<AnthropicMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
}

#[derive(Serialize)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    content: Vec<ContentBlock>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type", default)]
    block_type: String,
    #[serde(default)]
    text: String,
}

fn parse_anthropic_response(response: &AnthropicResponse) -> anyhow::Result<String> {
    match response.stop_reason.as_deref() {
        None | Some("end_turn") | Some("stop_sequence") => {}
        Some("max_tokens") => anyhow::bail!("response incomplete: stopped at max_tokens"),
        Some("refusal") => anyhow::bail!("response refused"),
        Some(reason) => anyhow::bail!("unsupported stop reason: {reason}"),
    }

    let text = response
        .content
        .iter()
        .filter(|block| block.block_type == "text")
        .map(|block| block.text.as_str())
        .collect::<Vec<_>>()
        .join("");

    let text = text.trim().to_string();
    if text.is_empty() && !response.content.is_empty() {
        anyhow::bail!("response contained no text output");
    }

    Ok(text)
}

/// Implements Provider using the Anthropic Messages API via raw HTTP.
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
        let max_tokens = if params.max_tokens > 0 {
            params.max_tokens
        } else {
            DEFAULT_MAX_TOKENS
        };

        let request = AnthropicRequest {
            model,
            max_tokens,
            messages: vec![AnthropicMessage {
                role: "user",
                content: user_prompt,
            }],
            system: (!system_prompt.is_empty()).then_some(system_prompt),
            temperature: params.temperature,
            top_p: params.top_p,
        };

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic API error {status}: {body}");
        }

        let response: AnthropicResponse = response.json().await?;
        parse_anthropic_response(&response)
    }

    async fn health_check(&self) -> anyhow::Result<()> {
        let response = self
            .client
            .get("https://api.anthropic.com/v1/models?limit=1")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("anthropic health check failed: {status}: {body}");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_anthropic_response_completed() {
        let response = AnthropicResponse {
            stop_reason: Some("end_turn".to_string()),
            content: vec![ContentBlock {
                block_type: "text".to_string(),
                text: " hello ".to_string(),
            }],
        };

        assert_eq!(parse_anthropic_response(&response).unwrap(), "hello");
    }

    #[test]
    fn test_parse_anthropic_response_refusal() {
        let response = AnthropicResponse {
            stop_reason: Some("refusal".to_string()),
            content: vec![],
        };

        assert!(parse_anthropic_response(&response).is_err());
    }

    #[test]
    fn test_parse_anthropic_response_max_tokens() {
        let response = AnthropicResponse {
            stop_reason: Some("max_tokens".to_string()),
            content: vec![],
        };

        assert!(
            parse_anthropic_response(&response)
                .unwrap_err()
                .to_string()
                .contains("max_tokens")
        );
    }

    #[test]
    fn test_parse_anthropic_response_without_text_errors() {
        let response = AnthropicResponse {
            stop_reason: Some("end_turn".to_string()),
            content: vec![ContentBlock {
                block_type: "tool_use".to_string(),
                text: String::new(),
            }],
        };

        assert!(parse_anthropic_response(&response).is_err());
    }
}
