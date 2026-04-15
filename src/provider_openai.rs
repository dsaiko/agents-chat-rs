use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::provider::{GenerateParams, Provider};

pub(crate) struct OpenAiCompatibleProvider {
    client: reqwest::Client,
    api_key: String,
    api_base: String,
    name: &'static str,
}

impl OpenAiCompatibleProvider {
    pub(crate) fn new(api_key: &str, api_base: &str, name: &'static str) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.to_string(),
            api_base: api_base.trim_end_matches('/').to_string(),
            name,
        }
    }

    pub(crate) async fn generate(
        &self,
        model: &str,
        system_prompt: &str,
        user_prompt: &str,
        params: &GenerateParams,
    ) -> anyhow::Result<String> {
        let request = ResponsesRequest {
            model,
            input: user_prompt,
            instructions: (!system_prompt.is_empty()).then_some(system_prompt),
            max_output_tokens: (params.max_tokens > 0).then_some(params.max_tokens),
            temperature: params.temperature,
            top_p: params.top_p,
        };

        let response = self
            .client
            .post(format!("{}/responses", self.api_base))
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("{status}: {body}");
        }

        let response: ResponsesResponse = response.json().await?;
        parse_openai_compatible_response(&response)
    }

    pub(crate) async fn health_check(&self) -> anyhow::Result<()> {
        let response = self
            .client
            .get(format!("{}/models", self.api_base))
            .bearer_auth(&self.api_key)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("{} health check failed: {status}: {body}", self.name);
        }

        Ok(())
    }
}

#[derive(Serialize)]
struct ResponsesRequest<'a> {
    model: &'a str,
    input: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ResponsesResponse {
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    error: Option<ResponseError>,
    #[serde(default)]
    incomplete_details: Option<IncompleteDetails>,
    #[serde(default)]
    output: Vec<ResponseOutputItem>,
}

#[derive(Debug, Deserialize)]
struct ResponseError {
    #[serde(default)]
    message: String,
}

#[derive(Debug, Deserialize)]
struct IncompleteDetails {
    #[serde(default)]
    reason: String,
}

#[derive(Debug, Deserialize)]
struct ResponseOutputItem {
    #[serde(rename = "type", default)]
    item_type: String,
    #[serde(default)]
    content: Vec<ResponseContentItem>,
}

#[derive(Debug, Deserialize)]
struct ResponseContentItem {
    #[serde(rename = "type", default)]
    item_type: String,
    #[serde(default)]
    text: String,
}

pub(crate) fn parse_openai_compatible_response(
    response: &ResponsesResponse,
) -> anyhow::Result<String> {
    match response.status.as_deref() {
        None | Some("completed") => {}
        Some("failed") => {
            if let Some(error) = &response.error
                && !error.message.is_empty()
            {
                anyhow::bail!("response failed: {}", error.message);
            }
            anyhow::bail!("response failed");
        }
        Some("incomplete") => {
            if let Some(details) = &response.incomplete_details
                && !details.reason.is_empty()
            {
                anyhow::bail!("response incomplete: {}", details.reason);
            }
            anyhow::bail!("response incomplete");
        }
        Some(status) => anyhow::bail!("unexpected response status: {status}"),
    }

    let text = response
        .output
        .iter()
        .filter(|item| item.item_type == "message")
        .flat_map(|item| item.content.iter())
        .filter(|item| item.item_type == "output_text")
        .map(|item| item.text.as_str())
        .collect::<Vec<_>>()
        .join("");

    let text = text.trim().to_string();
    if text.is_empty() && !response.output.is_empty() {
        anyhow::bail!("response contained no text output");
    }

    Ok(text)
}

/// Implements Provider using the OpenAI Responses API.
pub struct OpenAiProvider {
    inner: OpenAiCompatibleProvider,
}

impl OpenAiProvider {
    pub fn new(api_key: &str) -> Self {
        Self {
            inner: OpenAiCompatibleProvider::new(api_key, "https://api.openai.com/v1", "openai"),
        }
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_openai_response_completed() {
        let response = ResponsesResponse {
            status: Some("completed".to_string()),
            error: None,
            incomplete_details: None,
            output: vec![ResponseOutputItem {
                item_type: "message".to_string(),
                content: vec![ResponseContentItem {
                    item_type: "output_text".to_string(),
                    text: " hello ".to_string(),
                }],
            }],
        };

        assert_eq!(
            parse_openai_compatible_response(&response).unwrap(),
            "hello"
        );
    }

    #[test]
    fn test_parse_openai_response_allows_missing_status() {
        let response = ResponsesResponse {
            status: None,
            error: None,
            incomplete_details: None,
            output: vec![ResponseOutputItem {
                item_type: "message".to_string(),
                content: vec![ResponseContentItem {
                    item_type: "output_text".to_string(),
                    text: "hello".to_string(),
                }],
            }],
        };

        assert_eq!(
            parse_openai_compatible_response(&response).unwrap(),
            "hello"
        );
    }

    #[test]
    fn test_parse_openai_response_failed() {
        let response = ResponsesResponse {
            status: Some("failed".to_string()),
            error: Some(ResponseError {
                message: "boom".to_string(),
            }),
            incomplete_details: None,
            output: vec![],
        };

        assert!(
            parse_openai_compatible_response(&response)
                .unwrap_err()
                .to_string()
                .contains("boom")
        );
    }

    #[test]
    fn test_parse_openai_response_incomplete() {
        let response = ResponsesResponse {
            status: Some("incomplete".to_string()),
            error: None,
            incomplete_details: Some(IncompleteDetails {
                reason: "max_output_tokens".to_string(),
            }),
            output: vec![],
        };

        assert!(
            parse_openai_compatible_response(&response)
                .unwrap_err()
                .to_string()
                .contains("max_output_tokens")
        );
    }

    #[test]
    fn test_parse_openai_response_without_text_errors() {
        let response = ResponsesResponse {
            status: Some("completed".to_string()),
            error: None,
            incomplete_details: None,
            output: vec![ResponseOutputItem {
                item_type: "function_call".to_string(),
                content: vec![],
            }],
        };

        assert!(parse_openai_compatible_response(&response).is_err());
    }
}
