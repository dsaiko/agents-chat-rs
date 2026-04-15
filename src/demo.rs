use std::collections::HashMap;
use std::fs;
use std::path::Path;

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
    pub instructions: String,
}

/// Configuration for a debate session loaded from a demo directory.
#[derive(Debug, Clone)]
pub struct Demo {
    pub question: String,
    pub rounds: usize,
    pub agents: Vec<Agent>,
}

impl Demo {
    /// Loads a demo from a directory. Go-style YAML takes precedence; legacy
    /// Markdown demos remain supported for backward compatibility.
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        if dir.join("question.yaml").exists() {
            return load_yaml_demo(dir);
        }
        if dir.join("Question.md").exists() {
            return load_markdown_demo(dir);
        }

        anyhow::bail!("reading question.yaml: No such file or directory")
    }
}

fn load_yaml_demo(dir: &Path) -> anyhow::Result<Demo> {
    let question_data = fs::read_to_string(dir.join("question.yaml"))
        .map_err(|e| anyhow::anyhow!("reading question.yaml: {e}"))?;
    let fields = parse_yaml_mapping(&question_data)
        .map_err(|e| anyhow::anyhow!("parsing question.yaml: {e}"))?;

    let question = fields
        .get("question")
        .map(|value| value.trim().to_string())
        .unwrap_or_default();
    if question.is_empty() {
        anyhow::bail!("missing 'question' in question.yaml");
    }

    let rounds = match fields.get("rounds") {
        Some(value) if !value.trim().is_empty() => {
            let rounds = parse_yaml_i64(value)
                .map_err(|e| anyhow::anyhow!("invalid 'rounds' in question.yaml: {e}"))?;
            if rounds < 0 {
                anyhow::bail!("invalid 'rounds': must be 0 or greater");
            }
            if rounds == 0 {
                DEFAULT_ROUNDS
            } else {
                usize::try_from(rounds).map_err(|_| {
                    anyhow::anyhow!("invalid 'rounds' in question.yaml: out of range")
                })?
            }
        }
        _ => DEFAULT_ROUNDS,
    };

    let entries = fs::read_dir(dir).map_err(|e| anyhow::anyhow!("reading directory: {e}"))?;
    let mut agents = Vec::new();
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        if entry.file_type()?.is_dir()
            || !name_str.ends_with(".yaml")
            || name_str.eq_ignore_ascii_case("question.yaml")
        {
            continue;
        }

        let data = fs::read_to_string(entry.path())
            .map_err(|e| anyhow::anyhow!("reading {name_str}: {e}"))?;
        let agent = parse_yaml_agent(&data, &name_str)?;
        agents.push(agent);
    }

    agents.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(Demo {
        question,
        rounds,
        agents,
    })
}

fn parse_yaml_agent(content: &str, filename: &str) -> anyhow::Result<Agent> {
    let fields =
        parse_yaml_mapping(content).map_err(|e| anyhow::anyhow!("parsing {filename}: {e}"))?;

    let name = fields
        .get("name")
        .map(|value| value.trim().to_string())
        .unwrap_or_default();
    if name.is_empty() {
        anyhow::bail!("missing 'name' in {filename}");
    }

    let model = fields
        .get("model")
        .map(|value| value.trim().to_string())
        .unwrap_or_default();
    if model.is_empty() {
        anyhow::bail!("missing 'model' in {filename}");
    }

    let max_tokens = match fields.get("max_tokens") {
        Some(value) if !value.trim().is_empty() => {
            let parsed = parse_yaml_i64(value)
                .map_err(|e| anyhow::anyhow!("invalid 'max_tokens' in {filename}: {e}"))?;
            if parsed < 0 {
                anyhow::bail!("invalid 'max_tokens' in {filename}: must be 0 or greater");
            }
            u32::try_from(parsed)
                .map_err(|_| anyhow::anyhow!("invalid 'max_tokens' in {filename}: out of range"))?
        }
        _ => 0,
    };

    let temperature =
        parse_optional_yaml_float(fields.get("temperature"), "temperature", filename)?;
    let top_p = parse_optional_yaml_float(fields.get("top_p"), "top_p", filename)?;
    validate_optional_range("temperature", temperature, 0.0, 2.0, filename)?;
    validate_optional_range("top_p", top_p, 0.0, 1.0, filename)?;

    Ok(Agent {
        name,
        model,
        max_tokens,
        temperature,
        top_p,
        instructions: fields
            .get("instructions")
            .map(|value| value.trim().to_string())
            .unwrap_or_default(),
    })
}

fn load_markdown_demo(dir: &Path) -> anyhow::Result<Demo> {
    let question_data = fs::read_to_string(dir.join("Question.md"))
        .map_err(|e| anyhow::anyhow!("reading Question.md: {e}"))?;

    let (question, rounds) = match parse_frontmatter(&question_data) {
        Ok((fields, body)) => {
            let rounds = match fields.get("rounds") {
                Some(value) if !value.trim().is_empty() => {
                    let rounds = parse_yaml_i64(value)
                        .map_err(|e| anyhow::anyhow!("invalid 'rounds' in Question.md: {e}"))?;
                    if rounds < 0 {
                        anyhow::bail!("invalid 'rounds': must be 0 or greater");
                    }
                    if rounds == 0 {
                        DEFAULT_ROUNDS
                    } else {
                        usize::try_from(rounds).map_err(|_| {
                            anyhow::anyhow!("invalid 'rounds' in Question.md: out of range")
                        })?
                    }
                }
                _ => DEFAULT_ROUNDS,
            };
            (body.trim().to_string(), rounds)
        }
        Err(_) => (question_data.trim().to_string(), DEFAULT_ROUNDS),
    };

    if question.is_empty() {
        anyhow::bail!("missing 'question' in Question.md");
    }

    let entries = fs::read_dir(dir).map_err(|e| anyhow::anyhow!("reading directory: {e}"))?;
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
        let agent = parse_markdown_agent(&data, &name_str)?;
        agents.push(agent);
    }

    agents.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(Demo {
        question,
        rounds,
        agents,
    })
}

fn parse_markdown_agent(content: &str, filename: &str) -> anyhow::Result<Agent> {
    let (fields, body) = parse_frontmatter(content)?;

    let name = fields
        .get("name")
        .map(|value| value.trim().to_string())
        .unwrap_or_default();
    if name.is_empty() {
        anyhow::bail!("missing 'name' in {filename}");
    }

    let model = fields
        .get("model")
        .map(|value| value.trim().to_string())
        .unwrap_or_default();
    if model.is_empty() {
        anyhow::bail!("missing 'model' in {filename}");
    }

    let max_tokens = match fields.get("max_tokens") {
        Some(value) if !value.trim().is_empty() => {
            let parsed = parse_yaml_i64(value)
                .map_err(|e| anyhow::anyhow!("invalid 'max_tokens' in {filename}: {e}"))?;
            if parsed < 0 {
                anyhow::bail!("invalid 'max_tokens' in {filename}: must be 0 or greater");
            }
            u32::try_from(parsed)
                .map_err(|_| anyhow::anyhow!("invalid 'max_tokens' in {filename}: out of range"))?
        }
        _ => 0,
    };

    let temperature =
        parse_optional_yaml_float(fields.get("temperature"), "temperature", filename)?;
    let top_p = parse_optional_yaml_float(fields.get("top_p"), "top_p", filename)?;
    validate_optional_range("temperature", temperature, 0.0, 2.0, filename)?;
    validate_optional_range("top_p", top_p, 0.0, 1.0, filename)?;

    Ok(Agent {
        name,
        model,
        max_tokens,
        temperature,
        top_p,
        instructions: body.trim().to_string(),
    })
}

fn parse_optional_yaml_float(
    value: Option<&String>,
    field: &str,
    filename: &str,
) -> anyhow::Result<Option<f64>> {
    match value {
        Some(value) if !value.trim().is_empty() => parse_yaml_float(value)
            .map(Some)
            .map_err(|e| anyhow::anyhow!("invalid '{field}' in {filename}: {e}")),
        _ => Ok(None),
    }
}

fn validate_optional_range(
    field: &str,
    value: Option<f64>,
    min: f64,
    max: f64,
    filename: &str,
) -> anyhow::Result<()> {
    if let Some(value) = value {
        if !value.is_finite() || value < min || value > max {
            anyhow::bail!("invalid '{field}' in {filename}: must be between {min} and {max}");
        }
    }
    Ok(())
}

fn parse_yaml_i64(value: &str) -> anyhow::Result<i64> {
    value.trim().parse::<i64>().map_err(Into::into)
}

fn parse_yaml_float(value: &str) -> anyhow::Result<f64> {
    let normalized = match value.trim().to_ascii_lowercase().as_str() {
        ".nan" | "nan" => "NaN".to_string(),
        ".inf" | "+.inf" | "inf" | "+inf" | "infinity" | "+infinity" => "inf".to_string(),
        "-.inf" | "-inf" | "-infinity" => "-inf".to_string(),
        other => other.to_string(),
    };

    normalized.parse::<f64>().map_err(Into::into)
}

fn parse_yaml_mapping(content: &str) -> anyhow::Result<HashMap<String, String>> {
    let lines: Vec<&str> = content.lines().collect();
    let mut fields = HashMap::new();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            i += 1;
            continue;
        }

        let indent = leading_spaces(line);
        if indent != 0 {
            anyhow::bail!("unsupported indentation on line {}", i + 1);
        }

        let (raw_key, raw_value) = line
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("invalid YAML mapping on line {}", i + 1))?;
        let key = raw_key.trim();
        let value = raw_value.trim_start();

        if value == "|" {
            i += 1;
            let mut block_lines = Vec::new();
            let mut min_indent = usize::MAX;

            while i < lines.len() {
                let block_line = lines[i];
                if block_line.trim().is_empty() {
                    block_lines.push(block_line);
                    i += 1;
                    continue;
                }

                let block_indent = leading_spaces(block_line);
                if block_indent == 0 {
                    break;
                }

                min_indent = min_indent.min(block_indent);
                block_lines.push(block_line);
                i += 1;
            }

            let strip_indent = if min_indent == usize::MAX {
                0
            } else {
                min_indent
            };
            let mut value = String::new();
            for (idx, block_line) in block_lines.iter().enumerate() {
                if idx > 0 {
                    value.push('\n');
                }
                if block_line.trim().is_empty() {
                    continue;
                }
                value.push_str(&block_line[strip_indent..]);
            }

            fields.insert(key.to_string(), value);
            continue;
        }

        fields.insert(key.to_string(), value.trim().to_string());
        i += 1;
    }

    Ok(fields)
}

fn leading_spaces(value: &str) -> usize {
    value.as_bytes().iter().take_while(|b| **b == b' ').count()
}

fn parse_frontmatter(content: &str) -> anyhow::Result<(HashMap<String, String>, String)> {
    let content = content.trim();
    if !content.starts_with("---") {
        anyhow::bail!("missing frontmatter");
    }

    let after_open = &content[3..];
    let (frontmatter, rest) = after_open
        .split_once("---")
        .ok_or_else(|| anyhow::anyhow!("missing closing frontmatter delimiter"))?;

    let mut fields = HashMap::new();
    for line in frontmatter.trim().lines() {
        if let Some((k, v)) = line.trim().split_once(':') {
            fields.insert(k.trim().to_string(), v.trim().to_string());
        }
    }

    Ok((fields, rest.trim().to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        fs::write(p.join("question.yaml"), "question: What is Go?\n").unwrap();
        fs::write(
            p.join("AgentA.yaml"),
            "name: Alpha\nmodel: gpt-4o\ninstructions: Be concise.\n",
        )
        .unwrap();
        fs::write(
            p.join("AgentB.yaml"),
            "name: Beta\nmodel: gpt-5-mini\ninstructions: Be critical.\n",
        )
        .unwrap();

        let demo = Demo::load(p).unwrap();

        assert_eq!(demo.question, "What is Go?");
        assert_eq!(demo.agents.len(), 2);
        assert_eq!(demo.agents[0].name, "Alpha");
        assert_eq!(demo.agents[0].model, "gpt-4o");
        assert_eq!(demo.agents[1].name, "Beta");
        assert_eq!(demo.agents[1].model, "gpt-5-mini");
    }

    #[test]
    fn test_load_with_rounds() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        fs::write(p.join("question.yaml"), "rounds: 3\nquestion: Test topic\n").unwrap();
        fs::write(
            p.join("AgentA.yaml"),
            "name: A\nmodel: gpt-4o\ninstructions: Hi\n",
        )
        .unwrap();
        fs::write(
            p.join("AgentB.yaml"),
            "name: B\nmodel: gpt-4o\ninstructions: Hi\n",
        )
        .unwrap();

        let demo = Demo::load(p).unwrap();

        assert_eq!(demo.rounds, 3);
        assert_eq!(demo.question, "Test topic");
    }

    #[test]
    fn test_load_rejects_negative_rounds() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        fs::write(
            p.join("question.yaml"),
            "rounds: -1\nquestion: Test topic\n",
        )
        .unwrap();
        fs::write(
            p.join("AgentA.yaml"),
            "name: A\nmodel: gpt-4o\ninstructions: Hi\n",
        )
        .unwrap();

        assert!(Demo::load(p).is_err());
    }

    #[test]
    fn test_load_default_rounds() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        fs::write(p.join("question.yaml"), "question: Plain question\n").unwrap();
        fs::write(
            p.join("AgentA.yaml"),
            "name: A\nmodel: gpt-4o\ninstructions: Hi\n",
        )
        .unwrap();
        fs::write(
            p.join("AgentB.yaml"),
            "name: B\nmodel: gpt-4o\ninstructions: Hi\n",
        )
        .unwrap();

        let demo = Demo::load(p).unwrap();
        assert_eq!(demo.rounds, DEFAULT_ROUNDS);
    }

    #[test]
    fn test_load_with_all_agent_params() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        fs::write(p.join("question.yaml"), "question: Test\n").unwrap();
        fs::write(
            p.join("AgentA.yaml"),
            "name: A\nmodel: gpt-4o\nmax_tokens: 2048\ntemperature: 0.9\ntop_p: 0.95\ninstructions: Be helpful.\n",
        )
        .unwrap();

        let demo = Demo::load(p).unwrap();
        let agent = &demo.agents[0];
        assert_eq!(agent.max_tokens, 2048);
        assert_eq!(agent.temperature, Some(0.9));
        assert_eq!(agent.top_p, Some(0.95));
    }

    #[test]
    fn test_load_temperature_zero() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        fs::write(p.join("question.yaml"), "question: Test\n").unwrap();
        fs::write(
            p.join("AgentA.yaml"),
            "name: A\nmodel: gpt-4o\ntemperature: 0\ninstructions: Hi\n",
        )
        .unwrap();

        let demo = Demo::load(p).unwrap();
        assert_eq!(demo.agents[0].temperature, Some(0.0));
    }

    #[test]
    fn test_load_temperature_unset() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        fs::write(p.join("question.yaml"), "question: Test\n").unwrap();
        fs::write(
            p.join("AgentA.yaml"),
            "name: A\nmodel: gpt-4o\ninstructions: Hi\n",
        )
        .unwrap();

        let demo = Demo::load(p).unwrap();
        assert_eq!(demo.agents[0].temperature, None);
        assert_eq!(demo.agents[0].top_p, None);
    }

    #[test]
    fn test_load_rejects_negative_max_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        fs::write(p.join("question.yaml"), "question: Test\n").unwrap();
        fs::write(
            p.join("AgentA.yaml"),
            "name: A\nmodel: gpt-4o\nmax_tokens: -1\ninstructions: Hi\n",
        )
        .unwrap();

        assert!(Demo::load(p).is_err());
    }

    #[test]
    fn test_load_rejects_out_of_range_temperature() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        fs::write(p.join("question.yaml"), "question: Test\n").unwrap();
        fs::write(
            p.join("AgentA.yaml"),
            "name: A\nmodel: gpt-4o\ntemperature: 3\ninstructions: Hi\n",
        )
        .unwrap();

        assert!(Demo::load(p).is_err());
    }

    #[test]
    fn test_load_rejects_out_of_range_top_p() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        fs::write(p.join("question.yaml"), "question: Test\n").unwrap();
        fs::write(
            p.join("AgentA.yaml"),
            "name: A\nmodel: gpt-4o\ntop_p: 1.1\ninstructions: Hi\n",
        )
        .unwrap();

        assert!(Demo::load(p).is_err());
    }

    #[test]
    fn test_load_rejects_nan_temperature() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        fs::write(p.join("question.yaml"), "question: Test\n").unwrap();
        fs::write(
            p.join("AgentA.yaml"),
            "name: A\nmodel: gpt-4o\ntemperature: .nan\ninstructions: Hi\n",
        )
        .unwrap();

        assert!(Demo::load(p).is_err());
    }

    #[test]
    fn test_validate_optional_range_rejects_infinity() {
        assert!(
            validate_optional_range("temperature", Some(f64::INFINITY), 0.0, 2.0, "AgentA.yaml")
                .is_err()
        );
    }

    #[test]
    fn test_load_missing_name() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        fs::write(p.join("question.yaml"), "question: Topic\n").unwrap();
        fs::write(p.join("AgentA.yaml"), "model: gpt-4o\ninstructions: Hi\n").unwrap();

        assert!(Demo::load(p).is_err());
    }

    #[test]
    fn test_load_missing_model() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        fs::write(p.join("question.yaml"), "question: Topic\n").unwrap();
        fs::write(p.join("AgentA.yaml"), "name: A\ninstructions: Hi\n").unwrap();

        assert!(Demo::load(p).is_err());
    }

    #[test]
    fn test_load_missing_question() {
        let dir = tempfile::tempdir().unwrap();
        assert!(Demo::load(dir.path()).is_err());
    }

    #[test]
    fn test_load_empty_question() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        fs::write(p.join("question.yaml"), "rounds: 3\n").unwrap();
        fs::write(
            p.join("AgentA.yaml"),
            "name: A\nmodel: gpt-4o\ninstructions: Hi\n",
        )
        .unwrap();

        assert!(Demo::load(p).is_err());
    }

    #[test]
    fn test_load_skips_non_yaml_files() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        fs::write(p.join("question.yaml"), "question: Topic\n").unwrap();
        fs::write(
            p.join("AgentA.yaml"),
            "name: A\nmodel: gpt-4o\ninstructions: Hi\n",
        )
        .unwrap();
        fs::write(
            p.join("AgentB.yaml"),
            "name: B\nmodel: gpt-4o\ninstructions: Hi\n",
        )
        .unwrap();
        fs::write(p.join("notes.txt"), "should be ignored").unwrap();
        fs::write(p.join("readme.md"), "should also be ignored").unwrap();

        let demo = Demo::load(p).unwrap();
        assert_eq!(demo.agents.len(), 2);
    }

    #[test]
    fn test_parse_yaml_mapping_block_scalar() {
        let fields = parse_yaml_mapping(
            "name: Alice\ninstructions: |\n  line one\n  line two\nmodel: gpt-4o\n",
        )
        .unwrap();

        assert_eq!(fields.get("instructions").unwrap(), "line one\nline two");
    }

    #[test]
    fn test_load_legacy_markdown_demo() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        fs::write(
            p.join("Question.md"),
            "---\nrounds: 2\n---\nLegacy question?",
        )
        .unwrap();
        fs::write(
            p.join("AgentA.md"),
            "---\nname: Legacy\nmodel: gpt-4o\n---\nBe concise.",
        )
        .unwrap();
        fs::write(
            p.join("AgentB.md"),
            "---\nname: Other\nmodel: claude-sonnet-4-6\n---\nBe critical.",
        )
        .unwrap();

        let demo = Demo::load(p).unwrap();
        assert_eq!(demo.rounds, 2);
        assert_eq!(demo.question, "Legacy question?");
        assert_eq!(demo.agents.len(), 2);
    }
}
