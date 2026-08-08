# Ollama Studio 🦙

A fast, native cross-platform desktop GUI client for interacting with local LLMs via [Ollama](https://ollama.com). Built with Rust and `egui`.

---

## 🌟 Overview

**Ollama Studio** provides a clean, responsive desktop interface for your local LLMs. It runs completely offline on your machine without relying on external cloud APIs or heavy web runtimes like Electron.

---

## ✨ Key Features

- **🚀 Native Performance:** Built in Rust using `egui` for instant startup and low RAM usage.
- **🔒 Private & Offline:** All chat histories and data remain strictly on your local device.
- **🐧 Arch Linux Ready:** Native system integration including `.desktop` launcher, SVG icon, and `PKGBUILD` support.
- **🪟 Windows Support:** Standalone `.exe` execution and `.msi` installer support.
- **💾 Chat Persistence:** Automatically saves and reloads your previous chat sessions.
- **⚡ Zero Config:** Auto-detects local Ollama instances running on the default port (`11434`).

---

## 📋 Prerequisites

Ensure **Ollama** is installed and running on your system before launching Ollama Studio:

1. Download and install [Ollama](https://ollama.com).
2. Start Ollama and verify it's running:
   ```bash
   ollama list
   ```
3. Pull at least one local LLM model:
   ```bash
   ollama run llama3
   ```

---

## 🚀 Installation & Setup

### 1. Arch Linux (`PKGBUILD`)

Build and install Ollama Studio system-wide using `makepkg`:

```bash
# Clone repository
git clone https://github.com/vsk11-12/ollama-studio-app.git
cd ollama-studio-app

# Compile and install package
makepkg -sic
```

*This automatically places the binary in `/usr/bin/ollama-studio` and adds the application entry to your desktop launcher menu.*

---

### 2. Windows Executable

1. Navigate to the [Releases](https://github.com/vsk11-12/ollama-studio-app/releases) section.
2. Download the latest `ollama-studio.exe` file.
3. Double-click `ollama-studio.exe` to run.

---

### 3. Build from Source

Requirements: [Rust Toolchain](https://rustup.rs/) installed.

```bash
# Clone repository
git clone https://github.com/vsk11-12/ollama-studio-app.git
cd ollama-studio-app

# Build native release binary
cargo build --release
```

- **Linux binary location:** `target/release/ollama-studio`
- **Windows target cross-compile:**
  ```bash
  cargo build --target x86_64-pc-windows-gnu --release
  ```

---

## 📂 Repository Structure

```text
ollama-studio-app/
├── src/                    # Rust source files
├── ollama-studio.desktop   # Linux desktop launcher specification
├── ollama-studio.svg       # Application icon
├── PKGBUILD                # Arch Linux package build script
├── Cargo.toml              # Rust dependencies and crate configuration
└── README.md               # Project documentation
```

---

## 🙏 Acknowledgements

Special thanks to the open-source projects and teams that made this possible:

- **[Ollama Team](https://ollama.com):** For creating an amazing framework and API for running local LLMs easily.
- **[emilk & egui Contributors](https://github.com/emilk/egui):** For building a fast, lightweight immediate mode GUI library for Rust.
- **[Rust Community](https://www.rust-lang.org/):** For providing an extraordinary language ecosystem and toolchain.

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add NewFeature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
