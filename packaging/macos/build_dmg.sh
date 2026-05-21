#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

APP_NAME="NeuraBreak"
# Read version from pyproject.toml (single source of truth)
VERSION=$(grep -m1 '^version' "${REPO_ROOT}/pyproject.toml" | sed 's/.*"\(.*\)".*/\1/')
DIST_DIR="${REPO_ROOT}/dist"
DMG_NAME="${APP_NAME}-${VERSION}-macOS.dmg"
DMG_OUT="${DIST_DIR}/${DMG_NAME}"
APP_PATH="${DIST_DIR}/${APP_NAME}.app"
ENTITLEMENTS="${SCRIPT_DIR}/entitlements.plist"

# Optional signing / notarization
# Set CODESIGN_IDENTITY to "Developer ID Application: Your Name (TEAMID)"
# to enable deep signing.  Leave unset (or empty) to skip (dev / unsigned CI).
CODESIGN_IDENTITY="${CODESIGN_IDENTITY:-}"

# Set NOTARYTOOL_PROFILE to the keychain profile created once with:
#   xcrun notarytool store-credentials "notarytool-profile" \
#     --apple-id you@example.com --team-id TEAMID --password APP_SPECIFIC_PWD
# Leave unset (or empty) to skip notarization.
NOTARYTOOL_PROFILE="${NOTARYTOOL_PROFILE:-}"


echo "  NeuraBreak macOS DMG Builder  v${VERSION}"

# Verify .app exists
if [ ! -d "${APP_PATH}" ]; then
    echo " ERROR: .app not found at ${APP_PATH}"
    echo "   Run 'python packaging/macos/build.py' first!"
    exit 1
fi

# Code sign the .app
if [ -n "${CODESIGN_IDENTITY}" ]; then
    echo ""
    echo "Signing .app..."
    echo "  Identity:     ${CODESIGN_IDENTITY}"
    echo "  Entitlements: ${ENTITLEMENTS}"
    codesign \
        --deep --force --verify --verbose \
        --sign "${CODESIGN_IDENTITY}" \
        --options runtime \
        --entitlements "${ENTITLEMENTS}" \
        "${APP_PATH}"
    echo "Code signing complete."
else
    echo ""
    echo "  CODESIGN_IDENTITY not set — skipping code signing."
    echo "  (Distribution builds must set CODESIGN_IDENTITY.)"
fi

# Build DMG
# Remove old DMG
rm -f "${DMG_OUT}"

echo ""
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

# Notarize and staple
if [ -n "${NOTARYTOOL_PROFILE}" ]; then
    echo ""
    echo "Submitting DMG for notarization (profile: ${NOTARYTOOL_PROFILE})..."
    xcrun notarytool submit "${DMG_OUT}" \
        --keychain-profile "${NOTARYTOOL_PROFILE}" \
        --wait
    echo "Stapling notarization ticket to DMG..."
    xcrun stapler staple "${DMG_OUT}"
    echo "Notarization complete."
else
    echo ""
    echo "  NOTARYTOOL_PROFILE not set — skipping notarization."
    echo "  (Distribution builds must set NOTARYTOOL_PROFILE; unsigned DMGs are"
    echo "   blocked by Gatekeeper on download.)"
fi

echo ""
echo "   DMG created: ${DMG_OUT}"
echo ""
echo "  To test: open '${DMG_OUT}'"
