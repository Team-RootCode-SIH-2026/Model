#!/usr/bin/env bash
set -e

PYTHON=""

for candidate in python3.13 python3.12 python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Python 3.11, 3.12 or 3.13 is required."
    echo "On macOS with Homebrew, you can run:"
    echo "  brew install python@3.13"
    exit 1
fi

VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

echo "Using Python $VERSION"

rm -rf .venv
"$PYTHON" -m venv .venv

./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

echo
echo "Setup complete."
echo
echo "Run the app with:"
echo "  .venv/bin/python app.py"
echo
