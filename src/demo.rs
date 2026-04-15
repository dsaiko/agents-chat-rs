use std::fs;
use std::path::Path;

use serde::Deserialize;

/// Default number of debate rounds when not specified.
pub const DEFAULT_ROUNDS: usize = 5;

/// A single debate participant with its LLM configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Agent {
    pub name: String,
    pub model: String,
    pub max_tokens: u32,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub prompt: String,
}

/// Configuration for a debate session loaded from a single YAML file.
#[derive(Debug, Clone)]
pub struct Demo {
    pub question: String,
    pub rounds: usize,
    pub agents: Vec<Agent>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDemo {
    #[serde(default)]
    question: String,
    rounds: Option<i64>,
    #[serde(default)]
    agents: Vec<RawAgent>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAgent {
    #[serde(default)]
    name: String,
    #[serde(default)]
    model: String,
    max_tokens: Option<i64>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    #[serde(default)]
    prompt: String,
}

impl Demo {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;

        let raw: RawDemo = serde_yaml::from_str(&content)
            .map_err(|e| anyhow::anyhow!("parsing {}: {e}", path.display()))?;

        Self::from_raw(raw, &path.display().to_string())
    }

    fn from_raw(raw: RawDemo, filename: &str) -> anyhow::Result<Self> {
        let question = raw.question.trim().to_string();
        if question.is_empty() {
            anyhow::bail!("missing 'question' in {filename}");
        }

        let rounds = match raw.rounds {
            None | Some(0) => DEFAULT_ROUNDS,
            Some(value) => {
                if value < 0 {
                    anyhow::bail!("invalid 'rounds' in {filename}: must be 0 or greater");
                }
                usize::try_from(value)
                    .map_err(|_| anyhow::anyhow!("invalid 'rounds' in {filename}: out of range"))?
            }
        };

        if raw.agents.is_empty() {
            anyhow::bail!("missing 'agents' in {filename}");
        }

        let agents = raw
            .agents
            .into_iter()
            .map(|agent| Agent::from_raw(agent, filename))
            .collect::<anyhow::Result<Vec<_>>>()?;

        Ok(Self {
            question,
            rounds,
            agents,
        })
    }
}

impl Agent {
    fn from_raw(raw: RawAgent, filename: &str) -> anyhow::Result<Self> {
        let name = raw.name.trim().to_string();
        if name.is_empty() {
            anyhow::bail!("missing 'name' in {filename}");
        }

        let model = raw.model.trim().to_string();
        if model.is_empty() {
            anyhow::bail!("missing 'model' in {filename}");
        }

        let max_tokens = match raw.max_tokens {
            None => 0,
            Some(value) => {
                if value < 0 {
                    anyhow::bail!("invalid 'max_tokens' in {filename}: must be 0 or greater");
                }
                u32::try_from(value).map_err(|_| {
                    anyhow::anyhow!("invalid 'max_tokens' in {filename}: out of range")
                })?
            }
        };

        validate_optional_range("temperature", raw.temperature, 0.0, 2.0, filename)?;
        validate_optional_range("top_p", raw.top_p, 0.0, 1.0, filename)?;

        Ok(Self {
            name,
            model,
            max_tokens,
            temperature: raw.temperature,
            top_p: raw.top_p,
            prompt: raw.prompt.trim().to_string(),
        })
    }
}

fn validate_optional_range(
    field: &str,
    value: Option<f64>,
    min: f64,
    max: f64,
    filename: &str,
) -> anyhow::Result<()> {
    if let Some(value) = value
        && (!value.is_finite() || value < min || value > max)
    {
        anyhow::bail!("invalid '{field}' in {filename}: must be between {min} and {max}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_demo(dir: &Path, filename: &str, content: &str) -> std::path::PathBuf {
        let path = dir.join(filename);
        fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn test_load_single_file_demo() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_demo(
            dir.path(),
            "demo.yaml",
            r#"
question: What is Go?
agents:
  - name: Alpha
    model: gpt-4o
    prompt: Be concise.
  - name: Beta
    model: gpt-5-mini
    prompt: Be critical.
"#,
        );

        let demo = Demo::load(&path).unwrap();

        assert_eq!(demo.question, "What is Go?");
        assert_eq!(demo.agents.len(), 2);
        assert_eq!(demo.agents[0].name, "Alpha");
        assert_eq!(demo.agents[0].model, "gpt-4o");
        assert_eq!(demo.agents[0].prompt, "Be concise.");
        assert_eq!(demo.agents[1].name, "Beta");
        assert_eq!(demo.agents[1].model, "gpt-5-mini");
        assert_eq!(demo.agents[1].prompt, "Be critical.");
    }

    #[test]
    fn test_load_preserves_agent_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_demo(
            dir.path(),
            "demo.yaml",
            r#"
question: Order test
agents:
  - name: Zeta
    model: gpt-4o
    prompt: First in file.
  - name: Alpha
    model: gpt-4o
    prompt: Second in file.
"#,
        );

        let demo = Demo::load(&path).unwrap();
        assert_eq!(demo.agents[0].name, "Zeta");
        assert_eq!(demo.agents[0].model, "gpt-4o");
        assert_eq!(demo.agents[0].prompt, "First in file.");
        assert_eq!(demo.agents[1].name, "Alpha");
        assert_eq!(demo.agents[1].model, "gpt-4o");
        assert_eq!(demo.agents[1].prompt, "Second in file.");
    }

    #[test]
    fn test_load_with_rounds_and_all_agent_params() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_demo(
            dir.path(),
            "demo.yaml",
            r#"
rounds: 3
question: Test topic
agents:
  - name: A
    model: gpt-4o
    max_tokens: 2048
    temperature: 0.9
    top_p: 0.95
    prompt: Be helpful.
"#,
        );

        let demo = Demo::load(&path).unwrap();
        let agent = &demo.agents[0];

        assert_eq!(demo.rounds, 3);
        assert_eq!(agent.max_tokens, 2048);
        assert_eq!(agent.temperature, Some(0.9));
        assert_eq!(agent.top_p, Some(0.95));
    }

    #[test]
    fn test_load_default_rounds() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_demo(
            dir.path(),
            "demo.yaml",
            r#"
question: Plain question
agents:
  - name: A
    model: gpt-4o
    prompt: Hi
"#,
        );

        let demo = Demo::load(&path).unwrap();
        assert_eq!(demo.rounds, DEFAULT_ROUNDS);
    }

    #[test]
    fn test_load_block_scalars() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_demo(
            dir.path(),
            "demo.yaml",
            r#"
question: |
  Multi-line
  question
agents:
  - name: A
    model: gpt-4o
    prompt: |
      line one
      line two
"#,
        );

        let demo = Demo::load(&path).unwrap();
        assert_eq!(demo.question, "Multi-line\nquestion");
        assert_eq!(demo.agents[0].prompt, "line one\nline two");
    }

    #[test]
    fn test_load_rejects_instructions_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_demo(
            dir.path(),
            "demo.yaml",
            r#"
question: Test
agents:
  - name: A
    model: gpt-4o
    instructions: Hi
"#,
        );

        assert!(Demo::load(&path).is_err());
    }

    #[test]
    fn test_load_rejects_negative_rounds() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_demo(
            dir.path(),
            "demo.yaml",
            r#"
rounds: -1
question: Test topic
agents:
  - name: A
    model: gpt-4o
    prompt: Hi
"#,
        );

        assert!(Demo::load(&path).is_err());
    }

    #[test]
    fn test_load_rejects_negative_max_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_demo(
            dir.path(),
            "demo.yaml",
            r#"
question: Test
agents:
  - name: A
    model: gpt-4o
    max_tokens: -1
    prompt: Hi
"#,
        );

        assert!(Demo::load(&path).is_err());
    }

    #[test]
    fn test_load_rejects_out_of_range_temperature() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_demo(
            dir.path(),
            "demo.yaml",
            r#"
question: Test
agents:
  - name: A
    model: gpt-4o
    temperature: 3
    prompt: Hi
"#,
        );

        assert!(Demo::load(&path).is_err());
    }

    #[test]
    fn test_load_rejects_out_of_range_top_p() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_demo(
            dir.path(),
            "demo.yaml",
            r#"
question: Test
agents:
  - name: A
    model: gpt-4o
    top_p: 1.1
    prompt: Hi
"#,
        );

        assert!(Demo::load(&path).is_err());
    }

    #[test]
    fn test_load_rejects_nan_temperature() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_demo(
            dir.path(),
            "demo.yaml",
            r#"
question: Test
agents:
  - name: A
    model: gpt-4o
    temperature: .nan
    prompt: Hi
"#,
        );

        assert!(Demo::load(&path).is_err());
    }

    #[test]
    fn test_validate_optional_range_rejects_infinity() {
        assert!(
            validate_optional_range("temperature", Some(f64::INFINITY), 0.0, 2.0, "demo.yaml")
                .is_err()
        );
    }

    #[test]
    fn test_load_missing_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_demo(
            dir.path(),
            "demo.yaml",
            r#"
question: Topic
agents:
  - model: gpt-4o
    prompt: Hi
"#,
        );

        assert!(Demo::load(&path).is_err());
    }

    #[test]
    fn test_load_missing_model() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_demo(
            dir.path(),
            "demo.yaml",
            r#"
question: Topic
agents:
  - name: A
    prompt: Hi
"#,
        );

        assert!(Demo::load(&path).is_err());
    }

    #[test]
    fn test_load_missing_question() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_demo(
            dir.path(),
            "demo.yaml",
            r#"
agents:
  - name: A
    model: gpt-4o
    prompt: Hi
"#,
        );

        assert!(Demo::load(&path).is_err());
    }

    #[test]
    fn test_load_empty_question() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_demo(
            dir.path(),
            "demo.yaml",
            r#"
question:
agents:
  - name: A
    model: gpt-4o
    prompt: Hi
"#,
        );

        assert!(Demo::load(&path).is_err());
    }

    #[test]
    fn test_load_missing_agents() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_demo(dir.path(), "demo.yaml", "question: Topic\n");

        assert!(Demo::load(&path).is_err());
    }

    #[test]
    fn test_load_rejects_unknown_top_level_field() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_demo(
            dir.path(),
            "demo.yaml",
            r#"
question: Topic
surprise: true
agents:
  - name: A
    model: gpt-4o
    prompt: Hi
"#,
        );

        assert!(Demo::load(&path).is_err());
    }

    #[test]
    fn test_load_rejects_unknown_agent_field() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_demo(
            dir.path(),
            "demo.yaml",
            r#"
question: Topic
agents:
  - name: A
    model: gpt-4o
    mood: loud
    prompt: Hi
"#,
        );

        assert!(Demo::load(&path).is_err());
    }
}
