#!/usr/bin/env bash
# Build a .deb package for NeuraBreak.
# Run from repo root: bash packaging/linux/build_deb.sh
#
# Prerequisites:
#   - Python 3.11+ with uv
#   - dpkg-deb (part of dpkg, standard on Debian/Ubuntu)
#   - fakeroot

set -euo pipefail

VERSION="${VERSION:-0.1.0}"
ARCH="${ARCH:-amd64}"
PACKAGE_NAME="neurabreak_${VERSION}_${ARCH}"
DEST="dist/deb/${PACKAGE_NAME}"

echo "Building NeuraBreak ${VERSION} .deb for ${ARCH}..."

# Install app into staging area
mkdir -p "${DEST}/usr/lib/neurabreak"
mkdir -p "${DEST}/usr/bin"
mkdir -p "${DEST}/usr/share/applications"
mkdir -p "${DEST}/usr/share/pixmaps"
mkdir -p "${DEST}/DEBIAN"

# Install Python package to staging with pip
uv run pip install --target="${DEST}/usr/lib/neurabreak/lib" \
    --no-deps \
    "pyside6>=6.6" pydantic pydantic-settings structlog .

# Launcher script
cat > "${DEST}/usr/bin/neurabreak" << 'EOF'
#!/usr/bin/env bash
export PYTHONPATH="/usr/lib/neurabreak/lib:$PYTHONPATH"
exec python3 -m neurabreak "$@"
EOF
chmod +x "${DEST}/usr/bin/neurabreak"

# Desktop entry
cp packaging/linux/neurabreak.desktop "${DEST}/usr/share/applications/"

# DEBIAN control file
INSTALLED_SIZE=$(du -sk "${DEST}/usr" | cut -f1)

cat > "${DEST}/DEBIAN/control" << EOF
Package: neurabreak
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: ${ARCH}
Installed-Size: ${INSTALLED_SIZE}
Depends: python3 (>= 3.11), libxcb-cursor0, libgl1
Recommends: python3-pyside6
Maintainer: NeuraBreak Contributors <hello@neurabreak.app>
Homepage: https://github.com/abhijeetnishal/neurabreak
Description: AI-powered break & posture guardian
 NeuraBreak sits in your system tray and monitors your posture via webcam,
 reminding you to take breaks and correcting bad habits before they become pain.
EOF

# Build the package
mkdir -p dist/deb
fakeroot dpkg-deb --build "${DEST}" "dist/deb/${PACKAGE_NAME}.deb"

echo ""
echo "Built: dist/deb/${PACKAGE_NAME}.deb"
echo "Install with: sudo dpkg -i dist/deb/${PACKAGE_NAME}.deb"
