# Ollama Studio 🦙

A lightweight, native cross-platform desktop GUI client for interacting with local LLMs via [Ollama](https://ollama.com). Built with Rust and `egui`.

---

## 📸 Overview

- **Native Speed:** Minimal memory usage and near-instant startup time.
- **Privacy First:** 100% local execution. No external servers or API keys required.
- **Cross-Platform:** Works seamlessly on Arch Linux and Windows.

---

## ✨ Features

- **Model Selection:** Automatically detects and lets you switch between installed local Ollama models.
- **Chat Management:** Create, delete, and persist chat sessions across app restarts.
- **System Integration:** Includes native `.desktop` launcher and app icon for Linux desktops.
- **Zero Config:** Auto-detects local Ollama instance running on default port `11434`.

---

## 📋 Prerequisites

1. Install and start [Ollama](https://ollama.com).
2. Pull at least one local LLM model:

```bash
ollama run llama3
🪟 Windows (.exe)
    Go to the Releases section.
    Download the latest ollama-studio.exe.
    Double-click ollama-studio.exe to run.

🛠️ Build from Source
Prerequisites
    Rust Toolchain

# Clone repository
git clone [https://github.com/vsk11-12/ollama-studio-app.git](https://github.com/vsk11-12/ollama-studio-app.git)
cd ollama-studio-app

# Build native release binary
cargo build --release

Linux executable: target/release/ollama-studio

Windows target cross-compile:
  cargo build --target x86_64-pc-windows-gnu --release


📁 Repository Structure
    src/ — Application source code (Rust + egui).
    ollama-studio.desktop — Linux application launcher entry.
    ollama-studio.svg — Application scalable vector icon.
    PKGBUILD — Arch Linux package build script.
    Cargo.toml — Rust dependencies and manifest.

🤝 Contributing
Contributions, issues, and feature requests are welcome!
    Fork the project.
    Create your feature branch (git checkout -b feature/AmazingFeature).
    Commit```markdown

🌟 Overview

Ollama Studio provides a clean, responsive desktop interface for your local LLMs. It runs completely offline on your machine without relying on external cloud APIs or heavy web runtimes like Electron.
✨ Key Features
    🚀 Native Performance: Built in Rust using egui for instant startup and low RAM usage.
    🔒 Private & Offline: All chat histories and data remain strictly on your local device.
    🐧 Arch Linux Ready: Native system integration including .desktop launcher, SVG icon, and PKGBUILD support.
    🪟 Windows Binary & Installer: Supports standalone .exe execution and .msi installations.
    💾 Chat Persistence: Automatically saves and reloads your previous chat sessions.

📋 Prerequisites
Ensure Ollama is installed and running on your system before launching Ollama Studio:
Bash
# Verify Ollama is active
ollama list

📄 License
Distributed under the MIT License. See LICENSE for details

## 🙏 Acknowledgements

Special thanks to the open-source projects and teams that made this possible:

- **[Ollama Team](https://ollama.com):** For creating an amazing framework and API for running local LLMs easily.
- **[emilk & egui Contributors](https://github.com/emilk/egui):** For building a fast, lightweight immediate mode GUI library for Rust.
- **[Rust Community](https://www.rust-lang.org/):** For providing an extraordinary language ecosystem and toolchain.







