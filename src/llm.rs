use std::io::Write;

use anyhow::{Context, Result};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};

const SYSTEM_PROMPT: &str = r#"
You are a rubber duck. Listen to the user's problem and help them solve it by asking questions.
You are not allowed to answer the user's question directly.
"#;

#[derive(Debug, Clone)]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

impl MessageRole {
    fn to_str(&self) -> &str {
        match self {
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::System => "system",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
}

impl Message {
    fn system(content: &str) -> Self {
        Self {
            role: MessageRole::System,
            content: content.to_string(),
        }
    }

    fn user(content: &str) -> Self {
        Self {
            role: MessageRole::User,
            content: content.to_string(),
        }
    }

    fn assistant(content: &str) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.to_string(),
        }
    }

    fn format(&self) -> String {
        format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>\n",
            self.role.to_str(),
            self.content
        )
    }
}

pub struct Llm {
    backend: LlamaBackend,
    model: LlamaModel,
    output_stream: Box<dyn Write>,
    messages: Vec<Message>,
}

impl Llm {
    pub fn new(model_path: &str, output_stream: Box<dyn Write>) -> Result<Self> {
        let backend = LlamaBackend::init()?;
        let params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, model_path, &params)
            .with_context(|| "unable to load model")?;

        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));

        Ok(Self {
            backend,
            model,
            output_stream,
            messages: vec![Message::system(SYSTEM_PROMPT)],
        })
    }

    pub fn chat(&mut self, message_content: &str) -> Result<()> {
        let user_message = Message::user(message_content);
        let tokens_list = self.format_prompt(user_message.clone())?;
        let assistant_message = self.generate(&tokens_list, 256)?;

        self.messages.push(user_message.clone());
        self.messages.push(assistant_message.clone());

        Ok(())
    }

    fn format_prompt(&self, user_message: Message) -> Result<Vec<LlamaToken>> {
        let chat_history = self
            .messages
            .iter()
            .chain(std::iter::once(&user_message))
            .map(|m| m.format())
            .collect::<Vec<String>>()
            .join("");

        let tokens = self
            .model
            .str_to_token(
                &format!(
                    "<|begin_of_text|>{chat_history}<|start_header_id|>assistant<|end_header_id|>\n\n"
                ),
                AddBos::Always,
            )
            .with_context(|| "failed to tokenize")?;

        Ok(tokens)
    }

    fn build_batch(input_tokens: &[LlamaToken]) -> Result<LlamaBatch> {
        let mut batch = LlamaBatch::new(2048, 1);

        let last_index = input_tokens.len() as i32 - 1;
        for (i, token) in (0_i32..).zip(input_tokens.into_iter()) {
            // llama_decode will output logits only for the last token of the prompt
            let is_last = i == last_index;
            batch
                .add(*token, i, &[0], is_last)
                .with_context(|| "failed to add token")?;
        }

        Ok(batch)
    }

    fn generate(
        &mut self,
        input_tokens: &[LlamaToken],
        max_response_length: i32,
    ) -> Result<Message> {
        let mut ctx = self
            .model
            .new_context(&self.backend, LlamaContextParams::default())
            .with_context(|| "unable to create the llama_context")?;

        let mut batch = Self::build_batch(input_tokens)?;
        ctx.decode(&mut batch)
            .with_context(|| "llama_decode() failed")?;

        let start_token_idx = batch.n_tokens();
        let end_token_idx = start_token_idx + max_response_length;

        let mut sampler = LlamaSampler::greedy();
        let mut output = Vec::new();

        for token_idx in start_token_idx..end_token_idx {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if token == self.model.token_eos() {
                break;
            }

            let output_bytes = self
                .model
                .token_to_bytes(token, Special::Tokenize)
                .with_context(|| "failed to token to bytes")?;

            self.output_stream.write_all(&output_bytes)?;
            self.output_stream.flush()?;
            output.extend_from_slice(&output_bytes);

            batch.clear();
            batch
                .add(token, token_idx, &[0], true)
                .with_context(|| "failed to add token")?;

            ctx.decode(&mut batch).with_context(|| "failed to eval")?;
        }

        let output_string =
            String::from_utf8(output).with_context(|| "failed to convert token to string")?;
        Ok(Message::assistant(&output_string))
    }
}

impl Drop for Llm {
    fn drop(&mut self) {
        send_logs_to_tracing(LogOptions::default());
    }
}

#[cfg(test)]
mod tests {
    use crate::test_buffer::TestBuffer;
    use serial_test::serial;

    use super::*;

    #[test]
    #[serial]
    fn test_llm() {
        let output = TestBuffer::new();
        let mut llm = Llm::new(
            "models/bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_0.gguf",
            Box::new(output.clone()),
        )
        .unwrap();

        let input = "Hello\n";
        llm.chat(&input).unwrap();

        assert!(output.get_string_content().len() > 0);
    }
}
