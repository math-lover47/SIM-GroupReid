#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/../SIM_code_only}"

mkdir -p "${OUT_DIR}"
rm -rf "${OUT_DIR:?}"/*

rsync -a \
  --exclude '/datasets/' \
  --exclude '/checkpoints/' \
  --exclude '/logs/' \
  --exclude '/notebooks/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  "${ROOT_DIR}/" "${OUT_DIR}/"

echo "Created code-only bundle at: ${OUT_DIR}"
