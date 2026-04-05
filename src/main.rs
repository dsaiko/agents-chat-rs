mod demo;
mod language;
mod provider;
mod provider_anthropic;
mod provider_ollama;
mod provider_openai;

use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;

use demo::{Agent, Demo};
use language::{detect_language, Language};
use provider::{GenerateParams, Providers, PROVIDER_ANTHROPIC, PROVIDER_OLLAMA, PROVIDER_OPENAI};
use provider_anthropic::AnthropicProvider;
use provider_ollama::OllamaProvider;
use provider_openai::OpenAiProvider;

/// Maximum number of history entries to include in the prompt.
/// Older entries are truncated to stay within LLM context limits.
const MAX_HISTORY: usize = 8;

/// Multi-agent debate simulator — AI agents with different personalities
/// argue a topic, potentially using different LLM providers in the same conversation.
#[derive(Parser)]
#[command(name = "agents-chat", about = "Multi-agent AI debate simulator")]
struct Cli {
    /// Path to the demo directory (overrides DEMO_DIR env var)
    demo_dir: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    // Resolve demo directory: CLI arg takes priority, then DEMO_DIR env var.
    // Go: os.Args[1] || filepath.Join("demos", os.Getenv("DEMO_DIR"))
    let demo_dir = match cli.demo_dir {
        Some(dir) => dir,
        None => match std::env::var("DEMO_DIR") {
            Ok(dir) => PathBuf::from("demos").join(dir),
            Err(_) => anyhow::bail!("usage: agents-chat <demo-dir> or set DEMO_DIR"),
        },
    };

    let demo = Demo::load(&demo_dir)?;

    if demo.agents.len() < 2 {
        anyhow::bail!("need at least 2 agent files");
    }

    let providers = init_providers();

    // Verify all agents have a working provider
    for agent in &demo.agents {
        if let Err(e) = providers.for_model(&agent.model) {
            anyhow::bail!("agent {} (model {}): {e}", agent.name, agent.model);
        }
    }

    // Go uses context.WithTimeout(15 minutes); Rust uses tokio::time::timeout.
    tokio::time::timeout(Duration::from_secs(900), run_debate(&providers, &demo))
        .await
        .map_err(|_| anyhow::anyhow!("debate timed out after 15 minutes"))??;

    Ok(())
}

/// Runs the complete debate session — language detection, then rounds of agent responses.
async fn run_debate(providers: &Providers, demo: &Demo) -> anyhow::Result<()> {
    let separator = "\u{2500}".repeat(60);

    println!("{separator}");
    println!("  Language detection ({})....", demo.agents[0].model);
    let lang = detect_language(providers, &demo.agents[0].model, &demo.question).await;
    println!("  {}", lang.language);
    println!("  {}", demo.question);
    println!("{separator}");

    let mut history = vec![format!("{}: {}", lang.moderator, demo.question)];

    for i in 1..=demo.rounds {
        println!("\n\u{2500}\u{2500} {} \u{2500}\u{2500}", lang.round(i, demo.rounds));
        for agent in &demo.agents {
            let reply = run_agent(providers, &lang, agent, &history).await?;
            let indented = format!("    {}", reply.replace('\n', "\n    "));
            println!("\n  [{}] ({})\n{indented}", agent.name, agent.model);
            history.push(format!("{}: {reply}", agent.name));
        }
    }

    println!("\n{separator}");
    Ok(())
}

/// Sends the conversation history to the agent's LLM provider and returns the reply.
async fn run_agent(
    providers: &Providers,
    lang: &Language,
    agent: &Agent,
    history: &[String],
) -> anyhow::Result<String> {
    let (provider, model) = providers.for_model(&agent.model)?;

    let prompt = build_prompt(lang, history);
    let text = provider
        .generate(
            model,
            agent.instructions.trim(),
            &prompt,
            &GenerateParams {
                max_tokens: agent.max_tokens,
            },
        )
        .await?;

    if text.is_empty() {
        Ok(lang.empty_reply.clone())
    } else {
        Ok(text)
    }
}

/// Constructs the user prompt from conversation history,
/// keeping only the last MAX_HISTORY entries to stay within context limits.
fn build_prompt(lang: &Language, history: &[String]) -> String {
    let start = if history.len() > MAX_HISTORY {
        history.len() - MAX_HISTORY
    } else {
        0
    };

    let mut result = String::new();
    result.push_str(&lang.conversation_pre);
    result.push_str("\n\n");
    for line in &history[start..] {
        result.push_str(line);
        result.push('\n');
    }
    result.push('\n');
    result.push_str(&lang.conversation_post);
    result
}

/// Creates and returns providers based on available API keys and local services.
///
/// Go checks env vars and creates provider instances for each available key.
/// Rust does the same — providers are only added if the corresponding API key exists.
fn init_providers() -> Providers {
    let mut providers = Providers::new();

    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        providers.insert(
            PROVIDER_OPENAI.to_string(),
            Box::new(OpenAiProvider::new(&key)),
        );
    }

    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        providers.insert(
            PROVIDER_ANTHROPIC.to_string(),
            Box::new(AnthropicProvider::new(&key)),
        );
    }

    // Ollama doesn't require an API key — it reads OLLAMA_HOST from env.
    // Go uses api.ClientFromEnvironment() which succeeds if Ollama is reachable.
    // We always register the Ollama provider; connection errors happen at generate() time.
    providers.insert(
        PROVIDER_OLLAMA.to_string(),
        Box::new(OllamaProvider::new()),
    );

    providers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::default_language;
    use crate::provider::tests::MockProvider;

    #[tokio::test]
    async fn test_run_agent_with_mock() {
        let providers = Providers::from_iter([(
            PROVIDER_OPENAI.to_string(),
            Box::new(MockProvider::new("Hello from mock")) as Box<dyn crate::provider::Provider>,
        )]);

        let lang = default_language();
        let agent = Agent {
            name: "Test".to_string(),
            model: "gpt-4o".to_string(),
            max_tokens: 0,
            instructions: "Be helpful.".to_string(),
        };
        let history = vec!["Moderator: Test question".to_string()];

        let reply = run_agent(&providers, &lang, &agent, &history).await.unwrap();
        assert_eq!(reply, "Hello from mock");
    }

    #[tokio::test]
    async fn test_run_agent_with_ollama_model() {
        let providers = Providers::from_iter([(
            PROVIDER_OLLAMA.to_string(),
            Box::new(MockProvider::new("Hello from Ollama")) as Box<dyn crate::provider::Provider>,
        )]);

        let lang = default_language();
        let agent = Agent {
            name: "Test".to_string(),
            model: "ollama-qwen3:8b".to_string(),
            max_tokens: 0,
            instructions: "Be helpful.".to_string(),
        };
        let history = vec!["Moderator: Test question".to_string()];

        let reply = run_agent(&providers, &lang, &agent, &history).await.unwrap();
        assert_eq!(reply, "Hello from Ollama");
    }

    #[tokio::test]
    async fn test_run_agent_empty_reply() {
        let providers = Providers::from_iter([(
            PROVIDER_OPENAI.to_string(),
            Box::new(MockProvider::new("")) as Box<dyn crate::provider::Provider>,
        )]);

        let lang = default_language();
        let agent = Agent {
            name: "Test".to_string(),
            model: "gpt-4o".to_string(),
            max_tokens: 0,
            instructions: "Be helpful.".to_string(),
        };
        let history = vec!["Moderator: Hi".to_string()];

        let reply = run_agent(&providers, &lang, &agent, &history).await.unwrap();
        assert_eq!(reply, lang.empty_reply);
    }

    #[tokio::test]
    async fn test_run_agent_provider_error() {
        let providers = Providers::from_iter([(
            PROVIDER_OPENAI.to_string(),
            Box::new(MockProvider::with_error("API error")) as Box<dyn crate::provider::Provider>,
        )]);

        let lang = default_language();
        let agent = Agent {
            name: "Test".to_string(),
            model: "gpt-4o".to_string(),
            max_tokens: 0,
            instructions: "Be helpful.".to_string(),
        };
        let history = vec!["Moderator: Hi".to_string()];

        assert!(run_agent(&providers, &lang, &agent, &history).await.is_err());
    }

    #[test]
    fn test_build_prompt() {
        let lang = default_language();
        let history = vec![
            "Moderator: Topic".to_string(),
            "Agent A: Reply 1".to_string(),
        ];

        let prompt = build_prompt(&lang, &history);
        assert!(prompt.contains(&lang.conversation_pre));
        assert!(prompt.contains(&lang.conversation_post));
        assert!(prompt.contains("Moderator: Topic"));
        assert!(prompt.contains("Agent A: Reply 1"));
    }

    #[test]
    fn test_build_prompt_truncation() {
        let lang = default_language();

        // Create 10 history entries — only last 8 should appear
        let history: Vec<String> = (0..10).map(|i| format!("Entry {i}")).collect();

        let prompt = build_prompt(&lang, &history);

        assert!(!prompt.contains("Entry 0"), "should not contain Entry 0 (truncated)");
        assert!(!prompt.contains("Entry 1"), "should not contain Entry 1 (truncated)");
        assert!(prompt.contains("Entry 2"), "should contain Entry 2");
        assert!(prompt.contains("Entry 9"), "should contain Entry 9");
    }
}
