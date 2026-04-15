use async_trait::async_trait;
use ollama_rs::Ollama;
use ollama_rs::generation::chat::request::ChatMessageRequest;
use ollama_rs::generation::chat::{ChatMessage, ChatMessageResponse};
use ollama_rs::models::ModelOptions;

use crate::provider::{GenerateParams, Provider};

pub(crate) fn parse_ollama_response(response: &ChatMessageResponse) -> anyhow::Result<String> {
    let text = response.message.content.trim().to_string();
    if text.is_empty() && !response.message.tool_calls.is_empty() {
        anyhow::bail!("response contained tool calls instead of text output");
    }
    Ok(text)
}

/// Implements Provider using the Ollama chat API.
pub struct OllamaProvider {
    client: Ollama,
}

impl OllamaProvider {
    /// Creates a new Ollama provider using `OLLAMA_HOST` when set, otherwise
    /// the default localhost endpoint.
    pub fn from_environment() -> anyhow::Result<Self> {
        let client = match std::env::var("OLLAMA_HOST") {
            Ok(host) => Ollama::try_new(host)?,
            Err(_) => Ollama::default(),
        };

        Ok(Self { client })
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
        let mut has_options = false;
        let mut options = ModelOptions::default();

        if params.max_tokens > 0 {
            has_options = true;
            options = options.num_predict(params.max_tokens as i32);
        }
        if let Some(temperature) = params.temperature {
            has_options = true;
            options = options.temperature(temperature as f32);
        }
        if let Some(top_p) = params.top_p {
            has_options = true;
            options = options.top_p(top_p as f32);
        }
        if has_options {
            request = request.options(options);
        }

        let response = self.client.send_chat_messages(request).await?;
        parse_ollama_response(&response)
    }

    async fn health_check(&self) -> anyhow::Result<()> {
        self.client.list_local_models().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ollama_rs::generation::chat::{ChatMessage, ChatMessageFinalResponseData, MessageRole};

    fn response(
        content: &str,
        tool_calls: Vec<ollama_rs::generation::tools::ToolCall>,
    ) -> ChatMessageResponse {
        ChatMessageResponse {
            model: "test".to_string(),
            created_at: "2026-01-01T00:00:00Z".to_string(),
            message: ChatMessage {
                role: MessageRole::Assistant,
                content: content.to_string(),
                tool_calls,
                images: None,
                thinking: None,
            },
            logprobs: None,
            done: true,
            final_data: Some(ChatMessageFinalResponseData {
                total_duration: 0,
                load_duration: 0,
                prompt_eval_count: 0,
                prompt_eval_duration: 0,
                eval_count: 0,
                eval_duration: 0,
            }),
        }
    }

    #[test]
    fn test_parse_ollama_response_completed() {
        let response = response(" hello ", vec![]);
        assert_eq!(parse_ollama_response(&response).unwrap(), "hello");
    }

    #[test]
    fn test_parse_ollama_response_tool_calls_error() {
        let tool_call = ollama_rs::generation::tools::ToolCall {
            function: ollama_rs::generation::tools::ToolCallFunction {
                name: "lookup".to_string(),
                arguments: serde_json::json!({}),
            },
        };
        let response = response("", vec![tool_call]);
        assert!(parse_ollama_response(&response).is_err());
    }
}
