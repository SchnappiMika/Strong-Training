# 归档范围与数据完整性

## 纳入的内容

保留整个 FanHua 的训练脚本、配置、实验目录、报告、参考代码快照和有用数据；补充原 WemiEnv、本次教师来源 flappybird_rl、两篇提供的论文。原文件夹层级与实验目录名保持，便于追溯已有记录。

普通 Git 文件包含代码、配置、小型 JSON/CSV、文档和图片。模型、原始协议、更新明细、参考大数据和 ZIP 等重型内容保存在 Release 完整包。`ARCHIVE_MANIFEST.json` 中每个文件都有相对路径、字节数、SHA256 和 `git` 或 `release` 位置。

## 排除的内容

- 嵌套 `.git`：发布上游源码快照，不嵌入原仓库对象、登录信息或另一段 Git 历史。
- `__pycache__`、`.pyc`、`.venv`、`node_modules`：可再生成的编译缓存和已安装依赖。
- `browser_profile`、`chrome_profile`：HTML 检查过程中生成的浏览器用户目录；截图、QA 脚本和结果仍保留。
- `project.private.config.json`：开发者工具本机私有设置。共享 `project.config.json` 保留，开发者按本机需要重新设置本地调试。
- `.env`、私钥及凭据文件：不上传访问凭证；如果内容扫描发现实际令牌，构建停止，不静默改写文件。
- 本地 `FanHua/publishing/` 构建目录：防止把发布目录递归打包到自身。构建工具副本放在仓库 `tools/build_release.py`。
- 其他工作区目录中的课程练习、BottleJump 训练等：不属于此次双游戏共享项目。

具体排除文件列表写入清单，便于检查哪些内容没有纳入。原始训练产物 `.pt`、CSV、JSON 和 `.jsonl` 不因上传而清空或改写。

## 为发布新增或改动的文件

仓库根 README、FanHua 新版 README、兼容依赖范围、目录索引、校验清单和迁移工具属于发布整理。原 FanHua README 另存 `docs/HISTORICAL_FANHUA_README.md`，保留其当时描述。

文献和第三方资源保存在私有项目中用于学习与备份。项目未为第三方资源重新指定许可证。原实验记录中的本机路径作为历史证据保留；可迁移工具生成新运行目录，不修改历史检查点。

## 完整性校验

`tools/verify_archive.py --full` 对照每个归档文件的字节数和 SHA256，并校验最终两游戏读取同一模型、评估前后参数与更新计数不变。它只使用 Python 标准库，不启动真实训练、评估或客户端。

完整 ZIP 校验通过后发布为私有仓库 Release 附件。`SHA256SUMS.txt` 校验 ZIP 与最终模型两个下载文件。普通 Git 跟踪的轻型文件与 Release 包中的对应文件一致。
