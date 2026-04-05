use async_openai::config::OpenAIConfig;
use async_openai::types::responses::{
    CreateResponse, CreateResponseArgs, InputParam, OutputItem, OutputMessageContent,
};
use async_openai::Client;
use async_trait::async_trait;

use crate::provider::{GenerateParams, Provider, DEFAULT_MAX_TOKENS};

/// Implements Provider using the OpenAI Responses API.
///
/// Go uses `openai.Client` with `client.Responses.New()`; Rust uses `async-openai`'s
/// `Client` with `client.responses().create()`. The Responses API is newer than
/// Chat Completions and is what the Go code uses.
pub struct OpenAiProvider {
    client: Client<OpenAIConfig>,
}

impl OpenAiProvider {
    pub fn new(api_key: &str) -> Self {
        let config = OpenAIConfig::new().with_api_key(api_key);
        Self {
            client: Client::with_config(config),
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
        // Build the request using derive_builder pattern.
        // Go uses `responses.ResponseNewParams` with struct fields;
        // Rust uses `CreateResponseArgs` builder for the same purpose.
        let mut builder = CreateResponseArgs::default();
        builder.model(model.to_string());
        builder.input(InputParam::Text(user_prompt.to_string()));

        if !system_prompt.is_empty() {
            builder.instructions(system_prompt.to_string());
        }

        let max_tokens = if params.max_tokens > 0 {
            params.max_tokens
        } else {
            DEFAULT_MAX_TOKENS
        };
        builder.max_output_tokens(max_tokens);

        let request: CreateResponse = builder.build()?;
        let response = self.client.responses().create(request).await?;

        // Extract text from response output items.
        // Go uses `resp.OutputText()` helper; async-openai doesn't have that convenience,
        // so we manually iterate over output items and extract text content.
        let text: String = response
            .output
            .iter()
            .filter_map(|item| {
                if let OutputItem::Message(msg) = item {
                    Some(
                        msg.content
                            .iter()
                            .filter_map(|c| {
                                if let OutputMessageContent::OutputText(t) = c {
                                    Some(t.text.as_str())
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(""),
                    )
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");

        Ok(text.trim().to_string())
    }
}
