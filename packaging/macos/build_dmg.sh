#!/usr/bin/env bash
set -euo pipefail

APP_NAME="NeuraBreak"
VERSION="1.0.0"
DIST_DIR="packaging/macos/dist"
DMG_NAME="${APP_NAME}-${VERSION}-macOS.dmg"
DMG_OUT="${DIST_DIR}/${DMG_NAME}"
APP_PATH="${DIST_DIR}/${APP_NAME}.app"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  NeuraBreak macOS DMG Builder"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Verify .app exists
if [ ! -d "${APP_PATH}" ]; then
    echo "❌ ERROR: .app not found at ${APP_PATH}"
    echo "   Run pyinstaller first!"
    exit 1
fi

# Remove old DMG
rm -f "${DMG_OUT}"

echo "Building DMG..."
create-dmg \
    --volname "${APP_NAME}" \
    --window-pos 200 120 \
    --window-size 600 400 \
    --icon-size 100 \
    --icon "${APP_NAME}.app" 175 190 \
    --hide-extension "${APP_NAME}.app" \
    --app-drop-link 425 190 \
    --no-internet-enable \
    "${DMG_OUT}" \
    "${DIST_DIR}/"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ DMG created: ${DMG_OUT}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  To test: open '${DMG_OUT}'"
