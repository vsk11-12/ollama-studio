# Ollama Studio

Ollama Studio is a modern, responsive desktop application built in Rust using the `eframe` and `egui` framework designed to provide a seamless interface for interacting with local Large Language Models (LLMs) via Ollama[cite: 1].

---

## ✨ Key Features

* **Multi-Tab Chat Workspace**: Manage multiple conversation sessions simultaneously, with full support for creating, renaming, and closing chat tabs[cite: 1].
* **Real-Time Streaming**: Experience live token streaming with built-in generation tracking and the ability to stop generation or unload models on the fly[cite: 1].
* **File & Vision Attachments**: Upload images (automatically encoded to base64 for vision-enabled models like `llama3.2-vision`) or text files to include their contents directly within your prompt context[cite: 1].
* **System & Performance Statistics**: Monitor session metrics including prompt tokens, completion tokens, total token throughput, active memory/VRAM usage of loaded models, and a complete list of installed local models[cite: 1].
* **Comprehensive Settings & Preferences**: Easily toggle between Dark and Light theme modes, adjust UI zoom scaling, choose between Proportional and Monospace font families, customize base font sizes, and configure custom Ollama backend endpoints[cite: 1].
* **Markdown Rendering**: Rich Markdown rendering for assistant messages, complete with formatting support, code blocks, and a quick-copy button[cite: 1].
* **Message Editing**: Edit past user queries and re-submit them to truncate and branch out your chat history[cite: 1].
* **Automatic Persistence**: Chat sessions are automatically saved locally via asynchronous and synchronous routines into a `chats.json` file[cite: 1].

---

## 🛠️ Tech Stack

* **Language**: Rust[cite: 1]
* **GUI Framework**: `eframe` / `egui`[cite: 1]
* **Markdown Viewer**: `egui_commonmark`[cite: 1]
* **Serialization**: `serde` and `serde_json`[cite: 1]
* **HTTP Client**: `reqwest` (blocking client)[cite: 1]
* **File Dialogs**: `rfd`[cite: 1]
* **Encoding**: `base64`[cite: 1]

---

## 🚀 Getting Started

### Prerequisites

1. Ensure you have [Rust and Cargo](https://www.rust-lang.org/) installed on your system[cite: 1].
2. Make sure [Ollama](https://ollama.com/) is installed and running locally on its default port (`http://127.0.0.1:11434`)[cite: 1].

### Building and Running

Clone the project repository and run the application using Cargo[cite: 1]:

```bash
cargo run --release
