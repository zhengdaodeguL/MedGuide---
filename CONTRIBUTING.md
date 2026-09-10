# 贡献指南

感谢参与 MedGuide。提交变更前，请先确认修改符合医疗安全边界、隐私要求和现有工程约定。

## 开发准备

1. 使用 Python 3.11+、Node.js 20+ 和 PowerShell 7+。
2. 从 `.env.template` 创建 `.env`，只在未跟踪的 `.env` 中填写凭据。
3. 创建 Python 环境并安装后端依赖：

   ```powershell
   $ErrorActionPreference = 'Stop'
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
   ```

4. 安装前端依赖：进入 `frontend` 后执行 `npm ci`。

不得提交真实患者资料、访问令牌、Cookie、数据库快照、运行日志或供应商连接地址。

## 提交要求

- 保持变更聚焦，并为行为变化添加回归测试。
- 医疗风险、身份认证、会话隔离和外部数据发送相关变更必须说明威胁边界。
- 新增接口需要更新 README 和对应测试。
- 面向用户的文案不得承诺诊断、处方或疗效。

提交前执行：

```powershell
$ErrorActionPreference = 'Stop'
.\.venv\Scripts\python.exe scripts\check_version.py
.\.venv\Scripts\python.exe -m pytest -q backend\tests
.\.venv\Scripts\python.exe scripts\evaluate.py
Push-Location frontend
npm run build
Pop-Location
docker compose config --quiet
```

## 版本号

版本号只有一处真源：`backend/app/version.py` 的 `__version__`。发布新版本时同步修改
`frontend/package.json` 的 `version` 和 `CHANGELOG.md` 顶部新增的 `## [x.y.z]` 标题，
然后运行 `scripts/check_version.py` 确认三处一致；CI 会在不一致时直接失败。
打 tag 用 `v` 前缀（例如 `v0.1.0`），与 CHANGELOG 底部的链接保持一致。

## Pull Request

PR 描述应包含问题背景、实现范围、验证证据和仍未覆盖的集成环境。涉及界面的改动应附桌面和移动端截图；涉及医疗安全的改动应附正例、反例和边界用例。
