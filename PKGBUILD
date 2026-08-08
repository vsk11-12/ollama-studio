# Maintainer: vsk11-12 <your-email@example.com>
pkgname=ollama-studio
pkgver=0.1.0
pkgrel=1
pkgdesc="Desktop GUI client for Ollama LLMs"
arch=('x86_64')
url="https://github.com/vsk11-12/ollama-studio-app"
license=('MIT')
depends=('hicolor-icon-theme')
makedepends=('cargo' 'git')
source=("git+$url.git")
sha256sums=('SKIP')

build() {
  cd "$pkgname-app"
  cargo build --release --locked
}

package() {
  cd "$pkgname-app"

  # Binary
  install -Dm755 "target/release/ollama-studio" "$pkgdir/usr/bin/ollama-studio"

  # Desktop entry
  install -Dm644 "ollama-studio.desktop" "$pkgdir/usr/share/applications/ollama-studio.desktop"

  # Icon
  install -Dm644 "ollama-studio.svg" "$pkgdir/usr/share/icons/hicolor/scalable/apps/ollama-studio.svg"
}
