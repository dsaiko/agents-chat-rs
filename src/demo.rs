use std::fs;
use std::path::Path;

/// Default number of debate rounds when not specified in Question.md frontmatter.
pub const DEFAULT_ROUNDS: usize = 5;

/// A single debate participant with its LLM configuration.
/// Mirrors Go's `Agent` struct — fields map 1:1.
#[derive(Debug, Clone, PartialEq)]
pub struct Agent {
    pub name: String,
    pub model: String,
    pub max_tokens: u32,
    pub instructions: String,
}

/// Configuration for a debate session loaded from a directory of markdown files.
/// Mirrors Go's `Demo` struct.
#[derive(Debug, Clone)]
pub struct Demo {
    pub question: String,
    pub rounds: usize,
    pub agents: Vec<Agent>,
}

impl Demo {
    /// Loads a demo from a directory containing `Question.md` and agent `.md` files.
    ///
    /// `Question.md` may have optional frontmatter with `rounds` (defaults to 5).
    /// All other `.md` files are parsed as agents (must have `name` and `model` in frontmatter).
    /// Agents are sorted alphabetically by name.
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let question_path = dir.join("Question.md");
        let question_data = fs::read_to_string(&question_path)
            .map_err(|e| anyhow::anyhow!("reading Question.md: {e}"))?;

        let (question, rounds) = match parse_frontmatter(&question_data) {
            Ok((fm, body)) => {
                let rounds = match fm.get("rounds") {
                    Some(v) if !v.is_empty() => v
                        .parse::<usize>()
                        .map_err(|e| anyhow::anyhow!("invalid 'rounds' in Question.md: {e}"))?,
                    _ => DEFAULT_ROUNDS,
                };
                (body, rounds)
            }
            // No frontmatter — treat entire file as question text, default rounds.
            Err(_) => (question_data.trim().to_string(), DEFAULT_ROUNDS),
        };

        let entries = fs::read_dir(dir)
            .map_err(|e| anyhow::anyhow!("reading directory: {e}"))?;

        let mut agents = Vec::new();
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            if entry.file_type()?.is_dir()
                || !name_str.ends_with(".md")
                || name_str.eq_ignore_ascii_case("Question.md")
            {
                continue;
            }

            let data = fs::read_to_string(entry.path())
                .map_err(|e| anyhow::anyhow!("reading {name_str}: {e}"))?;

            let agent = parse_agent_file(&data)
                .map_err(|e| anyhow::anyhow!("parsing {name_str}: {e}"))?;

            agents.push(agent);
        }

        agents.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(Demo {
            question,
            rounds,
            agents,
        })
    }
}

/// Parses a markdown file with frontmatter containing `name` and `model` fields.
/// The markdown body becomes the agent's system prompt / instructions.
pub fn parse_agent_file(content: &str) -> anyhow::Result<Agent> {
    let (fm, body) = parse_frontmatter(content)?;

    let name = fm
        .get("name")
        .filter(|v| !v.is_empty())
        .ok_or_else(|| anyhow::anyhow!("missing 'name' in frontmatter"))?
        .clone();

    let model = fm
        .get("model")
        .filter(|v| !v.is_empty())
        .ok_or_else(|| anyhow::anyhow!("missing 'model' in frontmatter"))?
        .clone();

    let max_tokens = match fm.get("max_tokens") {
        Some(v) if !v.is_empty() => v
            .parse::<u32>()
            .map_err(|e| anyhow::anyhow!("invalid 'max_tokens' in frontmatter: {e}"))?,
        _ => 0,
    };

    Ok(Agent {
        name,
        model,
        max_tokens,
        instructions: body,
    })
}

/// Parses YAML-like frontmatter delimited by `---` lines.
/// Returns a map of key-value pairs and the body text after the closing delimiter.
///
/// Go used `strings.Cut` for splitting; Rust uses `str::split_once` — same semantics.
pub fn parse_frontmatter(content: &str) -> anyhow::Result<(std::collections::HashMap<String, String>, String)> {
    let content = content.trim();
    if !content.starts_with("---") {
        anyhow::bail!("missing frontmatter");
    }

    let after_open = &content[3..];
    let (frontmatter, rest) = after_open
        .split_once("---")
        .ok_or_else(|| anyhow::anyhow!("missing closing frontmatter delimiter"))?;

    let mut fields = std::collections::HashMap::new();
    for line in frontmatter.trim().lines() {
        if let Some((k, v)) = line.trim().split_once(':') {
            fields.insert(k.trim().to_string(), v.trim().to_string());
        }
    }

    let body = rest.trim().to_string();
    Ok((fields, body))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_parse_frontmatter() {
        let cases = vec![
            (
                "valid frontmatter with body",
                "---\nname: Test\nmodel: gpt-4o\n---\nHello world",
                Some((vec![("name", "Test"), ("model", "gpt-4o")], "Hello world")),
            ),
            (
                "frontmatter only, no body",
                "---\nkey: value\n---",
                Some((vec![("key", "value")], "")),
            ),
            (
                "whitespace around values",
                "---\n  name :  Agent A  \n---\nBody text",
                Some((vec![("name", "Agent A")], "Body text")),
            ),
            (
                "missing opening delimiter",
                "name: Test\n---\nBody",
                None,
            ),
            (
                "missing closing delimiter",
                "---\nname: Test\nBody",
                None,
            ),
            ("empty input", "", None),
        ];

        for (name, input, expected) in cases {
            let result = parse_frontmatter(input);
            match expected {
                None => {
                    assert!(result.is_err(), "{name}: expected error, got Ok");
                }
                Some((expected_fields, expected_body)) => {
                    let (fields, body) = result.unwrap_or_else(|e| panic!("{name}: unexpected error: {e}"));
                    assert_eq!(body, expected_body, "{name}: body mismatch");
                    for (k, v) in expected_fields {
                        assert_eq!(
                            fields.get(k).map(|s| s.as_str()),
                            Some(v),
                            "{name}: fields[{k}] mismatch"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_parse_agent_file() {
        let cases: Vec<(&str, &str, Option<Agent>)> = vec![
            (
                "valid agent",
                "---\nname: Agent A\nmodel: gpt-4o\n---\nYou are helpful.",
                Some(Agent {
                    name: "Agent A".to_string(),
                    model: "gpt-4o".to_string(),
                    max_tokens: 0,
                    instructions: "You are helpful.".to_string(),
                }),
            ),
            ("missing name", "---\nmodel: gpt-4o\n---\nInstructions", None),
            ("missing model", "---\nname: Agent A\n---\nInstructions", None),
            ("invalid frontmatter", "no frontmatter here", None),
        ];

        for (name, input, expected) in cases {
            let result = parse_agent_file(input);
            match expected {
                None => {
                    assert!(result.is_err(), "{name}: expected error, got Ok");
                }
                Some(want) => {
                    let got = result.unwrap_or_else(|e| panic!("{name}: unexpected error: {e}"));
                    assert_eq!(got, want, "{name}: agent mismatch");
                }
            }
        }
    }

    #[test]
    fn test_load() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        fs::write(p.join("Question.md"), "What is Go?").unwrap();
        fs::write(
            p.join("AgentA.md"),
            "---\nname: Alpha\nmodel: gpt-4o\n---\nBe concise.",
        )
        .unwrap();
        fs::write(
            p.join("AgentB.md"),
            "---\nname: Beta\nmodel: gpt-5-mini\n---\nBe critical.",
        )
        .unwrap();

        let demo = Demo::load(p).unwrap();

        assert_eq!(demo.question, "What is Go?");
        assert_eq!(demo.agents.len(), 2);

        // Agents should be sorted by name
        assert_eq!(demo.agents[0].name, "Alpha");
        assert_eq!(demo.agents[0].model, "gpt-4o");
        assert_eq!(demo.agents[1].name, "Beta");
        assert_eq!(demo.agents[1].model, "gpt-5-mini");
    }

    #[test]
    fn test_load_with_rounds() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        fs::write(p.join("Question.md"), "---\nrounds: 3\n---\nTest topic").unwrap();
        fs::write(p.join("AgentA.md"), "---\nname: A\nmodel: gpt-4o\n---\nHi").unwrap();
        fs::write(p.join("AgentB.md"), "---\nname: B\nmodel: gpt-4o\n---\nHi").unwrap();

        let demo = Demo::load(p).unwrap();
        assert_eq!(demo.rounds, 3);
        assert_eq!(demo.question, "Test topic");
    }

    #[test]
    fn test_load_default_rounds() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        // No frontmatter — should default to 5 rounds
        fs::write(p.join("Question.md"), "Plain question").unwrap();
        fs::write(p.join("AgentA.md"), "---\nname: A\nmodel: gpt-4o\n---\nHi").unwrap();
        fs::write(p.join("AgentB.md"), "---\nname: B\nmodel: gpt-4o\n---\nHi").unwrap();

        let demo = Demo::load(p).unwrap();
        assert_eq!(demo.rounds, 5);
    }

    #[test]
    fn test_load_invalid_rounds() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        fs::write(p.join("Question.md"), "---\nrounds: abc\n---\nTopic").unwrap();
        fs::write(p.join("AgentA.md"), "---\nname: A\nmodel: gpt-4o\n---\nHi").unwrap();

        assert!(Demo::load(p).is_err());
    }

    #[test]
    fn test_load_missing_question() {
        let dir = tempfile::tempdir().unwrap();
        assert!(Demo::load(dir.path()).is_err());
    }

    #[test]
    fn test_load_skips_non_md_files() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        fs::write(p.join("Question.md"), "Topic").unwrap();
        fs::write(p.join("AgentA.md"), "---\nname: A\nmodel: gpt-4o\n---\nHi").unwrap();
        fs::write(p.join("AgentB.md"), "---\nname: B\nmodel: gpt-4o\n---\nHi").unwrap();
        fs::write(p.join("notes.txt"), "should be ignored").unwrap();

        let demo = Demo::load(p).unwrap();
        assert_eq!(demo.agents.len(), 2, "txt file should be ignored");
    }
}
