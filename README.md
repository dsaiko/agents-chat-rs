# agents-chat

A multi-agent AI debate simulator written in Rust. Two or more AI agents with different personalities argue a topic across multiple rounds, optionally using different LLM providers in the same conversation.

## Features

- **Multiple LLM providers** — OpenAI, Anthropic (Claude), and Ollama in a single debate
- **Automatic language detection** — UI strings are translated to match the question's language
- **Configurable debates** — define agents, models, personalities, and round count via simple Markdown files
- **Mixed-provider debates** — e.g. GPT vs Claude arguing the same topic
- **Conversation history** — agents see prior exchanges (last 8 entries) for coherent multi-turn debate
- **15-minute timeout** — prevents runaway sessions

## Requirements

- Rust 2024 edition (1.85+)
- At least one LLM provider configured:
  - **OpenAI** — set `OPENAI_API_KEY`
  - **Anthropic** — set `ANTHROPIC_API_KEY`
  - **Ollama** — running locally (reads `OLLAMA_HOST`, defaults to `localhost:11434`)

## Installation

```bash
cargo build --release
```

## Usage

```bash
# Run a demo by path
agents-chat demos/flat_earth_en

# Or set DEMO_DIR (resolved relative to demos/)
DEMO_DIR=flat_earth_en agents-chat
```

### Environment Variables

| Variable            | Description                          |
|---------------------|--------------------------------------|
| `OPENAI_API_KEY`    | OpenAI API key                       |
| `ANTHROPIC_API_KEY` | Anthropic API key                    |
| `OLLAMA_HOST`       | Ollama server address                |
| `DEMO_DIR`          | Default demo directory (under `demos/`) |
| `RUST_LOG`          | Logging level (e.g. `debug`, `info`) |

## Demo Structure

Each demo is a directory containing Markdown files:

```
demos/flat_earth_en/
├── Question.md      # The debate topic
├── AgentA.md        # First agent definition
└── AgentB.md        # Second agent definition
```

### Question.md

Optional frontmatter with `rounds` (defaults to 5). The body is the debate question.

```markdown
---
rounds: 5
---
Is the Earth flat or round? Defend your position.
```

### Agent Files

Each agent file has frontmatter with `name`, `model`, and optional `max_tokens`. The body is the agent's system prompt.

```markdown
---
name: Alice
model: gpt-5.3-chat-latest
---
You are Alice, a passionate flat Earth believer. You are loud,
confrontational, and absolutely convinced the Earth is flat.
Keep it to 2-3 punchy sentences max.
```

### Model Routing

Models are routed to providers by prefix:

| Prefix      | Provider  | Example                          |
|-------------|-----------|----------------------------------|
| `claude*`   | Anthropic | `claude-sonnet-4-6`              |
| `ollama-*`  | Ollama    | `ollama-qwen3:8b` (prefix stripped) |
| *(default)* | OpenAI    | `gpt-5.3-chat-latest`           |

## Included Demos

| Demo                   | Language   | Providers           |
|------------------------|------------|---------------------|
| `flat_earth_en`        | English    | OpenAI + Anthropic  |
| `flat_earth_cz`        | Czech      | OpenAI + Anthropic  |
| `flat_earth_de`        | German     | OpenAI + Anthropic  |
| `flat_earth_es`        | Spanish    | OpenAI + Anthropic  |
| `flat_earth_fr`        | French     | OpenAI + Anthropic  |
| `flat_earth_jp`        | Japanese   | OpenAI + Anthropic  |
| `flat_earth_pt`        | Portuguese | OpenAI + Anthropic  |
| `flat_earth_cn`        | Chinese    | OpenAI + Anthropic  |
| `flat_earth_ollama_en` | English    | Ollama              |
| `flat_earth_ollama_cz` | Czech      | Ollama              |
| `better_times_cz`      | Czech      | OpenAI + Anthropic  |

## Running Tests

```bash
cargo test
```

## License

MIT
