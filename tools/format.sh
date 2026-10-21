#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

if [ "$#" -eq 0 ]; then
  cd "${REPO_ROOT}"
  python -m ruff check --fix .
  python -m ruff format .
  python -m ruff check .
else
  python -m ruff check --fix "$@"
  python -m ruff format "$@"
  python -m ruff check "$@"
fi
