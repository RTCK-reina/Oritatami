#!/usr/bin/env bash
# Oritatami — macOS .app bundle builder
# Usage: cd Native && ./build-app.sh [--release]
set -euo pipefail

CONF="${1:---debug}"
[[ "$CONF" == "--release" ]] && SWIFT_CONF="release" || SWIFT_CONF="debug"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
APP_NAME="Oritatami"
BUILD_DIR="$SCRIPT_DIR/.build/$SWIFT_CONF"

echo "→ Swift build ($SWIFT_CONF)..."
cd "$SCRIPT_DIR"
# Auto-select Xcode over Command Line Tools (needed for SwiftUI macros)
if [ -d "/Applications/Xcode.app" ]; then
  export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
elif [ -d "/Applications/Xcode-beta.app" ]; then
  export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
fi
swift build -c "$SWIFT_CONF"

echo "→ Frontend build..."
cd "$REPO_ROOT/frontend"
if [[ ! -d node_modules ]]; then npm ci; fi
npm run build

echo "→ Assembling .app bundle..."
APP_BUNDLE="$SCRIPT_DIR/$APP_NAME.app"
CONTENTS="$APP_BUNDLE/Contents"
MACOS="$CONTENTS/MacOS"
RESOURCES="$CONTENTS/Resources"

rm -rf "$APP_BUNDLE"
mkdir -p "$MACOS" "$RESOURCES"

# Executable
cp "$BUILD_DIR/$APP_NAME" "$MACOS/$APP_NAME"

# Embed built frontend into Resources so backend can serve it
mkdir -p "$RESOURCES/frontend_dist"
cp -R "$REPO_ROOT/frontend/dist/." "$RESOURCES/frontend_dist/"

# Info.plist
cat > "$CONTENTS/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>    <string>app.rtck.oritatami</string>
    <key>CFBundleName</key>          <string>Oritatami</string>
    <key>CFBundleDisplayName</key>   <string>Oritatami</string>
    <key>CFBundleVersion</key>       <string>0.3.0</string>
    <key>CFBundleShortVersionString</key> <string>0.3.0</string>
    <key>CFBundleExecutable</key>    <string>$APP_NAME</string>
    <key>CFBundlePackageType</key>   <string>APPL</string>
    <key>NSHighResolutionCapable</key> <true/>
    <key>LSMinimumSystemVersion</key> <string>14.0</string>
    <key>NSAppTransportSecurity</key>
    <dict>
        <key>NSAllowsLocalNetworking</key> <true/>
    </dict>
</dict>
</plist>
PLIST

echo "✓ Built: $APP_BUNDLE"
echo "  Run:   open '$APP_BUNDLE'"
