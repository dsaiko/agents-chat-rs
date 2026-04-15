use serde::{Deserialize, Serialize};

use crate::provider::{GenerateParams, Providers};

/// Localized UI strings for a specific language.
///
/// Mirrors Go's `Language` struct — serde tags match the Go JSON tags exactly,
/// ensuring the LLM detection prompt and response format are compatible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Language {
    pub moderator: String,
    pub round_format: String,
    pub empty_reply: String,
    pub conversation_pre: String,
    pub conversation_post: String,
    /// The `detected_language` JSON key matches Go's struct tag.
    #[serde(rename = "detected_language")]
    pub language: String,
}

impl Language {
    /// Formats the round header using the localized format string.
    ///
    /// Go uses `fmt.Sprintf(l.RoundFormat, i, total)` with a runtime format string.
    /// Rust's `format!` requires compile-time literals, so we use manual `replacen`
    /// to substitute the two `%d` placeholders — same semantics as Go's Sprintf for this case.
    pub fn round(&self, i: usize, total: usize) -> String {
        self.round_format
            .replacen("%d", &i.to_string(), 1)
            .replacen("%d", &total.to_string(), 1)
    }
}

/// English defaults used as fallback when language detection fails.
pub fn default_language() -> Language {
    Language {
        moderator: "Moderator".to_string(),
        round_format: "Round %d/%d".to_string(),
        empty_reply: "(empty reply)".to_string(),
        conversation_pre: "Conversation so far:".to_string(),
        conversation_post:
            "Reply as the next participant of the debate. Write nothing beyond your reply."
                .to_string(),
        language: "Detected language: English".to_string(),
    }
}

/// Uses an LLM to detect the language of text and translate UI strings.
/// Falls back to English defaults on any failure — matching Go's behavior exactly.
pub async fn detect_language(providers: &Providers, model: &str, text: &str) -> Language {
    let (provider, resolved_model) = match providers.for_model(model) {
        Ok(v) => v,
        Err(_) => return default_language(),
    };

    let originals = serde_json::to_string_pretty(&default_language()).unwrap_or_default();

    let prompt = format!(
        r#"Detect the language of the text below and translate these UI strings into that language.
Reply with ONLY a valid JSON object — no markdown, no code fences, no commentary.

English originals:
{originals}

IMPORTANT:
- Keep the two %d/%d placeholders exactly as-is in "round_format"
- In "detected_language", translate the phrase and use the actual language name

Text:
{text}"#
    );

    let result = match provider
        .generate(
            resolved_model,
            "",
            &prompt,
            &GenerateParams {
                max_tokens: 0,
                temperature: None,
                top_p: None,
            },
        )
        .await
    {
        Ok(v) => v,
        Err(_) => return default_language(),
    };

    parse_language_json(&result)
}

/// Extracts a Language from an LLM response, stripping markdown fences if present.
pub fn parse_language_json(raw: &str) -> Language {
    let mut s = raw.trim().to_string();

    // Strip markdown code fences if present (e.g., ```json ... ```)
    if s.starts_with("```") {
        if let Some(idx) = s[3..].find('\n') {
            s = s[3 + idx + 1..].to_string();
        }
        if let Some(idx) = s.rfind("```") {
            s = s[..idx].to_string();
        }
        s = s.trim().to_string();
    }

    let lang: Language = match serde_json::from_str(&s) {
        Ok(v) => v,
        Err(_) => return default_language(),
    };

    let lang = Language {
        moderator: lang.moderator.trim().to_string(),
        round_format: lang.round_format.trim().to_string(),
        empty_reply: lang.empty_reply.trim().to_string(),
        conversation_pre: lang.conversation_pre.trim().to_string(),
        conversation_post: lang.conversation_post.trim().to_string(),
        language: lang.language.trim().to_string(),
    };

    if lang.moderator.is_empty()
        || lang.empty_reply.is_empty()
        || lang.conversation_pre.is_empty()
        || lang.conversation_post.is_empty()
        || lang.language.is_empty()
        || !valid_round_format(&lang.round_format)
    {
        return default_language();
    }

    lang
}

fn valid_round_format(format: &str) -> bool {
    if format.is_empty() {
        return false;
    }

    let bytes = format.as_bytes();
    let mut placeholders = 0;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'%' {
            i += 1;
            continue;
        }
        i += 1;
        if i >= bytes.len() {
            return false;
        }
        match bytes[i] {
            b'%' => {}
            b'd' => placeholders += 1,
            _ => return false,
        }
        i += 1;
    }

    placeholders == 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_format() {
        let lang = default_language();
        assert_eq!(lang.round(2, 5), "Round 2/5");
    }

    #[test]
    fn test_parse_language_json_plain() {
        let json = r#"{"moderator":"Mod","round_format":"R %d/%d","empty_reply":"(e)","conversation_pre":"Pre:","conversation_post":"Post.","detected_language":"Lang: Test"}"#;
        let lang = parse_language_json(json);
        assert_eq!(lang.moderator, "Mod");
    }

    #[test]
    fn test_parse_language_json_fenced() {
        let json = r#"{"moderator":"Mod","round_format":"R %d/%d","empty_reply":"(e)","conversation_pre":"Pre:","conversation_post":"Post.","detected_language":"Lang: Test"}"#;
        let fenced = format!("```json\n{json}\n```");
        let lang = parse_language_json(&fenced);
        assert_eq!(lang.moderator, "Mod");
    }

    #[test]
    fn test_parse_language_json_missing_format_placeholder() {
        let json = r#"{"moderator":"M","round_format":"Runde","empty_reply":"(e)","conversation_pre":"P:","conversation_post":"P.","detected_language":"L"}"#;
        let lang = parse_language_json(json);
        assert_eq!(lang.round_format, default_language().round_format);
    }

    #[test]
    fn test_parse_language_json_partial_defaults() {
        let json = r#"{"moderator":"Mod","round_format":"R %d/%d"}"#;
        let lang = parse_language_json(json);
        assert_eq!(lang, default_language());
    }

    #[test]
    fn test_parse_language_json_invalid() {
        let lang = parse_language_json("garbage");
        assert_eq!(lang, default_language());
    }

    #[tokio::test]
    async fn test_detect_language_with_mock() {
        use crate::provider::tests::MockProvider;

        let cz_json = r#"{"moderator":"Moderátor","round_format":"Kolo %d/%d","empty_reply":"(prázdná odpověď)","conversation_pre":"Dosavadní konverzace:","conversation_post":"Odpověz jako další účastník debaty. Nenapiš nic navíc mimo svou repliku.","detected_language":"Detekovaný jazyk: Čeština"}"#;
        let providers = Providers::from_iter([(
            "openai".to_string(),
            Box::new(MockProvider::new(cz_json)) as Box<dyn crate::provider::Provider>,
        )]);

        let lang = detect_language(&providers, "gpt-4o", "Nějaký český text").await;
        assert_eq!(lang.moderator, "Moderátor");
        assert_eq!(lang.language, "Detekovaný jazyk: Čeština");
        assert_eq!(lang.round(1, 3), "Kolo 1/3");
    }

    #[tokio::test]
    async fn test_detect_language_fallback() {
        use crate::provider::tests::MockProvider;

        // Invalid JSON → fallback to English
        let providers = Providers::from_iter([(
            "openai".to_string(),
            Box::new(MockProvider::new("not json at all")) as Box<dyn crate::provider::Provider>,
        )]);

        let lang = detect_language(&providers, "gpt-4o", "Some text").await;
        assert_eq!(lang.language, default_language().language);
    }

    #[tokio::test]
    async fn test_detect_language_provider_error() {
        use crate::provider::tests::MockProvider;

        let providers = Providers::from_iter([(
            "openai".to_string(),
            Box::new(MockProvider::with_error("fail")) as Box<dyn crate::provider::Provider>,
        )]);

        let lang = detect_language(&providers, "gpt-4o", "Some text").await;
        assert_eq!(lang.language, default_language().language);
    }

    #[tokio::test]
    async fn test_detect_language_no_provider() {
        let providers = Providers::new();
        let lang = detect_language(&providers, "gpt-4o", "Some text").await;
        assert_eq!(lang.language, default_language().language);
    }
}
