# agents-chat

A multi-agent AI debate simulator written in Rust. Two or more AI agents with different personalities argue a topic across multiple rounds, optionally using different LLM providers in the same conversation.

## Features

- **Multi-provider support** — OpenAI, Anthropic (Claude), OpenRouter, and local Ollama models can be mixed in the same debate
- **Automatic language detection and translation** — the first agent's model detects the question language and translates UI strings dynamically
- **YAML-based configuration** — agents, personalities, and questions can be defined as `.yaml` files in the same format as the Go version
- **Per-agent model selection** — each agent can use a different model and provider
- **Demo scenarios** — bundled demos cover OpenAI, Anthropic, Ollama, and OpenRouter combinations
- **Legacy Markdown compatibility** — existing `.md` demos still load, but YAML is the primary format

## Requirements

- Rust 2024 edition (1.85+)
- At least one provider configured:
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `OPENROUTER_API_KEY`
  - local Ollama instance (`OLLAMA_HOST` optional, defaults to localhost)

## Installation

```bash
cargo build --release
```

## Usage

```bash
# Run a demo by path
cargo run -- demos/flat_earth_en

# Or set DEMO_DIR (resolved relative to demos/)
DEMO_DIR=flat_earth_en cargo run
```

## Configuration

Each demo is a directory under `demos/` containing YAML files.

### `question.yaml`

```yaml
rounds: 5
question: Is the Earth flat or round? Defend your position.
```

| Field | Description | Default |
|-------|-------------|---------|
| `rounds` | Number of debate rounds | `5` |

### Agent files

```yaml
name: Alice
model: gpt-5.3-chat-latest
max_tokens: 2048
temperature: 0.9
top_p: 0.95
instructions: |
  You are Alice, a passionate flat Earth believer.
  You are loud, confrontational, and absolutely convinced the Earth is flat.
```

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Display name of the agent |
| `model` | yes | Model ID that determines provider routing |
| `max_tokens` | no | Maximum response tokens (`0` means provider default; Anthropic falls back to `1024`) |
| `temperature` | no | Sampling temperature (`0` is valid and distinct from unset) |
| `top_p` | no | Nucleus sampling threshold |
| `instructions` | no | System prompt for the agent |

### Model routing

| Prefix | Provider | Example |
|--------|----------|---------|
| `ollama/` | Ollama | `ollama/qwen3:8b` |
| `openrouter/` | OpenRouter | `openrouter/qwen/qwen3.6-plus:free` |
| `claude*` | Anthropic | `claude-sonnet-4-6` |
| default | OpenAI | `gpt-5.3-chat-latest` |

Legacy `ollama-` model prefixes are still accepted for older Markdown demos.

## Included demos

- `flat_earth_en`
- `flat_earth_cz`
- `flat_earth_de`
- `flat_earth_es`
- `flat_earth_fr`
- `flat_earth_jp`
- `flat_earth_pt`
- `flat_earth_cn`
- `flat_earth_ollama_en`
- `flat_earth_ollama_cz`
- `flat_earth_openrouter_cz`
- `better_times_cz`

## Tests

```bash
cargo test
```

## License

MIT
