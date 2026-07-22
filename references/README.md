# 参考资料本地缓存（references）

该目录用于缓存论文 PDF 与上游仓库（用于复现/对照）。

- 清单：`references/manifest.yaml`
- 锁文件：`references/LOCK.json`

默认不纳入 git（见 `.gitignore`）：
- `references/papers/`
- `references/repos/`

## 拉取全部

```bash
turnkey-refs fetch --all
```

该命令会根据 manifest 下载论文，并将上游仓库检出到 manifest 指定的 detached commit，
随后在 `references/LOCK.json` 中记录 requested/resolved commit。

## 安全提示

部分上游仓库可能包含越狱 prompt 或敏感内容，请仅用于安全研究与内部分析。
