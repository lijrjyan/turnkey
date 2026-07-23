# Third-party sources

This directory records external code that must be pinned for a method reproduction.

Use one of two forms:

- Prefer a Git submodule under `third_party/submodules/`.
- Otherwise keep the clone external and pin its commit in `references/LOCK.json`.

Keep patches minimal under `third_party/patches/`. Every dependency must record its upstream URL, commit or version, and license. Turnkey's Apache-2.0 license does not replace third-party terms.
