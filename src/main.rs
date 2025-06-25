mod llm;
#[cfg(test)]
mod test_buffer;

use anyhow::Result;
use std::env;
use std::io::{self, Write};

fn main() -> Result<()> {
    // Get command line arguments
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <models_directory>", args[0]);
        std::process::exit(1);
    }

    let models_dir = &args[1];

    let mut llm = llm::Llm::new(
        format!(
            "{models_dir}/bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_0.gguf"
        )
        .as_str(),
        Box::new(std::io::stdout()),
    )?;

    println!("Enter your text (press Enter to send, Ctrl+C to exit):");

    loop {
        // Print prompt
        print!("> ");
        io::stdout().flush()?;

        // Read input from stdin
        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(_) => {
                let input = input.trim();

                // Skip empty inputs
                if input.is_empty() {
                    continue;
                }

                // Send to LLM
                llm.chat(input)?;
                println!();
            }
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                return Err(e.into());
            }
        }
    }
}
