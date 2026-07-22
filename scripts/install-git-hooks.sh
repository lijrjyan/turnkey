#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
git -C "$repository_root" config core.hooksPath .githooks
echo "hooks=.githooks"
