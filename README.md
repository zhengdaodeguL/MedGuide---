# MedGuide

MedGuide 是面向健康信息整理与就医指引的可审计服务。系统以 LangChain/LangGraph 编排多轮信息采集，在风险筛查之后执行意图路由、知识检索、只读结构化查询和安全审查，回答始终保留引用、风险等级与会话摘要。

> 医疗安全边界：MedGuide 只整理健康信息、提示就医紧迫性、推荐就诊方向并解释常见药品与检查资料；不能替代执业医生完成诊断、处方、停药或急救处置。出现胸痛、呼吸困难、意识改变、突发单侧无力、大量出血或严重过敏反应时，应立即联系当地急救服务或前往急诊。

## 能力

- 多轮咨询状态：年龄、性别、主诉、持续时间、伴随症状、既往史和风险历史
- 节点化工作流：输入规范化、要素抽取、意图识别、风险筛查、检索、SQL、生成、安全审查、摘要
- 混合 RAG：查询改写、知识库路由、BM25 召回、Milvus 向量召回、融合排序、实体一致性、去重和上下文压缩
- 知识域：疾病、药品、检查检验、科室和服务边界 FAQ
- NL2SQL：表/字段白名单、单条 SELECT、参数绑定、索引友好的前缀匹配和 `LIMIT <= 50`
- 账户体系：用户名唯一、PBKDF2 密码哈希、HttpOnly 会话 Cookie、会话归属校验和注销
- SSE：节点进度、最终答案、错误事件与请求幂等恢复
- 可审计指标：检索命中、引用覆盖、风险比例、结构化查询、反馈和响应耗时

## 快速部署

### 1. 准备配置

```powershell
Copy-Item .env.template .env
```

生产模式至少填写 `MEDGUIDE_DOMAIN`、`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`、`REDIS_URL` 和 `MEDGUIDE_AUTH_DSN`。`MEDGUIDE_DOMAIN` 必须是已经解析到部署主机的公网域名；内置 Caddy 网关会自动申请并续期 TLS 证书。知识后端默认是 `milvus`，此时还必须填写 `MILVUS_URI`；若只使用经过审核并生成索引的词法知识库，请将 `MEDGUIDE_KNOWLEDGE_BACKEND` 明确设为 `bm25`。供应商地址填写到 `/v1` 基础路径，SDK 会调用标准 `/v1/chat/completions`：

```text
OPENAI_BASE_URL=<OpenAI-compatible API base URL>
OPENAI_MODEL=<supported model identifier>
```

`MEDGUIDE_COOKIE_SECURE=true` 和 `MEDGUIDE_REQUIRE_HTTPS=true` 是生产默认值。二者必须成对使用；不要把 `.env`、患者信息或数据库密码提交到仓库。

### 2. 启动服务

```powershell
docker compose up --build -d
docker compose ps
```

基础文件默认开启 Secure Cookie 和 HTTPS 门禁。Caddy 是唯一发布到主机的入口，自动把 HTTP 重定向到 HTTPS，并将 UI 与 `/api` 保持在同一 Origin；API 和 UI 容器只暴露在 Compose 网络内。浏览器使用账户 Cookie，不读取或转发服务令牌。

API 容器会在进程启动前按知识后端准备索引。`bm25` 模式根据镜像内经过审核的 `data/knowledge/catalog.json` 验证运行卷快照：格式、Embedding 契约和目录 generation 均匹配的 v2 快照会直接复用，旧格式、损坏快照或目录内容变化会触发原子重建。`milvus` 模式执行完整入库，所有向量批次成功后才原子发布同 generation 的 BM25 快照并清理旧向量代。任一步失败都会阻止 API 启动，UI 也不会提前接收流量。

### 3. 首次使用

打开前端地址，先创建用户名和密码。用户名长度为 3–64 个字符，允许中文等 Unicode 文字、数字、`.`、`_`、`-`；密码长度为 8–128 个字符。创建账户后即可建立整理会话，刷新页面会恢复登录状态。

### Windows 本地启动

完成 `.env` 和依赖配置后，可在 PowerShell 7 中运行 `./scripts/start-api.ps1`。若 Python 无法直连供应商而 Windows 已启用静态系统代理，使用 `./scripts/start-api.ps1 -UseSystemProxy`；它只为当前 API 进程导入已有代理，不修改系统设置。其他环境可按需设置 `HTTP_PROXY`、`HTTPS_PROXY` 和 `NO_PROXY`。容器中的 `localhost` 指容器本身，代理地址应由部署者按实际网络配置。

`OPENAI_MAX_TOKENS` 默认1600，用于兼容含推理额度的模型；回复仍由提示词约束为简洁的健康信息。供应商返回空内容或截断内容时，服务不会把它当作成功回答。

## 配置说明

| 配置 | 作用 |
| --- | --- |
| `MEDGUIDE_MODE` | 生产部署设为 `production` |
| `MEDGUIDE_DOMAIN` | 已解析到部署主机的公网域名；TLS 网关启动时校验，缺失或非法则失败关闭 |
| `MEDGUIDE_HTTP_PORT` / `MEDGUIDE_HTTPS_PORT` | TLS 网关对外发布端口，默认 80/443 |
| `MEDGUIDE_CORS_ORIGINS` | 受控跨源部署的精确 Origin 列表；同源部署保持为空 |
| `MEDGUIDE_READINESS_CACHE_SECONDS` | 外部依赖就绪结果的短 TTL，用于合并并发探针 |
| `MEDGUIDE_READINESS_TIMEOUT_SECONDS` | 每个请求等待就绪检查的总预算，默认 5 秒；超时返回 503，并发请求复用同一次检查 |
| `OPENAI_BASE_URL` / `OPENAI_MODEL` | OpenAI 兼容 Chat Completions 服务与模型 |
| `OPENAI_API_KEY` | 仅由 API 容器读取 |
| `REDIS_URL` | 多 worker 会话、幂等、TTL 和指标存储 |
| `MILVUS_URI` | `milvus` 知识后端的向量检索服务地址；使用默认后端时必填，并完成 Embedding 契约校验 |
| `MYSQL_DSN` | 可选的只读业务数据源；配置后启用排班、库存和检查价格查询 |
| `MEDGUIDE_AUTH_DSN` | 生产共享认证数据库地址；生产默认 `MEDGUIDE_AUTH_BACKEND=mysql` 时必填 |
| `MEDGUIDE_API_TOKENS` | 逗号分隔的服务间凭据；使用密码学随机值，浏览器账户不得使用 |
| `MEDGUIDE_AUTH_DB_PATH` | 仅供显式 SQLite 隔离模式使用的账户数据库路径；生产不会回退到该路径 |
| `MEDGUIDE_COOKIE_SECURE` | 是否为认证 Cookie 设置 `Secure` |
| `MEDGUIDE_REQUIRE_HTTPS` | UI 入口是否拒绝未由 TLS 网关转发的请求；需与 Cookie 策略保持一致 |
| `MEDGUIDE_NETWORK_SUBNET` | Compose 隔离网络及 UI/API 可信代理的唯一网段来源；显式空值或漂移会在容器入口失败关闭 |

浏览器部署只支持 UI 与 `/api` 同源；不要为前端构建或运行时设置独立 API Origin。若在 Caddy 之前增加组织网关，必须保留 HTTPS、Cookie、CSP、真实客户端地址和缓存边界。未配置 Milvus 或 MySQL 时，服务会明确报告能力未启用，不会把测试夹具当作生产数据。配置任一外部适配器后，健康检查会验证其真实连通性；失败时整理接口返回 503，而不是静默降级。

## HTTP 接口

- `POST /api/auth/register`：创建账户并设置会话 Cookie
- `POST /api/auth/login`：验证账户并设置会话 Cookie
- `POST /api/auth/logout`：撤销当前会话
- `GET /api/auth/me`：读取当前登录用户名
- `POST /api/sessions`、`GET /api/sessions/{id}`：创建和读取归属会话
- `POST /api/chat?stream=true`：SSE 整理；生产请求必须携带 `request_id`
- `GET /api/sessions/{id}/requests/{request_id}`：断线后的幂等结果恢复
- `POST /api/search`、`POST /api/query`：受保护的知识检索与只读查询
- `POST /api/feedback`：按助手回答 `message_id` 记录反馈
- `GET /api/health`、`GET /api/ready`：服务健康与编排就绪探针

生产模式关闭 Swagger、ReDoc 和 OpenAPI 文件；外层网关仍应负责 HTTPS、组织级身份、审计和公网访问控制。

## 知识入库

知识目录位于 `data/knowledge/catalog.json`。每条外部医学资料都保存发布机构、来源日期和可点击的 HTTPS 原文地址；MedGuide 自有安全边界与就诊方向映射会在来源名称中明确区分，不能冒充上游机构结论。

> **仓库自带的目录是有限的健康信息摘要集。** 当前含15条资料，覆盖范围有限，尚未经过医学审核，不能直接作为生产医学知识库。上线前必须替换为你在所在地域取得合法授权的语料，并由具备资质的医学审核人确认内容、地域化急救口径和复审周期。脚本不会自动联网抓取来源，`source_url` 只是引用地址。

替换语料后需要重建 BM25 快照；不重建时 `/api/ready` 会返回 503 并给出缺失原因，服务不会带着空知识库对外宣称就绪：

```powershell
$env:MEDGUIDE_KNOWLEDGE_BACKEND = "bm25"
.\.venv\Scripts\python.exe scripts\build_bm25_index.py --if-stale
```

默认 `milvus` 后端会在每次 API 容器启动前使用同一组 `EMBEDDING_MODEL`、`EMBEDDING_VERSION` 和 `EMBEDDING_DIMENSION` 完成向量入库，并在所有向量批次成功后原子发布 BM25 快照。以下命令可用于受控的手动重新入库；`bm25` 后端不会调用外部 Embedding 或改写 Milvus：

```powershell
docker compose run --rm --entrypoint python medguide-api -m app.ingestion
```

collection 必须使用显式 `INT64 id`、关闭 `auto_id`，并保存 `chunk_id`、`document_id`、`text`、`title`、`category`、`source`、`updated_at`、`source_url`、`corpus_generation` 以及 Embedding 契约属性。可点击来源仅接受目录中已审核机构的精确 HTTPS 主机名，且不得包含用户信息、片段、非常规端口或查询凭据。供应商不支持 Embedding 时保持 BM25-only，并在健康状态中明确显示。

## 质量验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q backend\tests
.\.venv\Scripts\python.exe scripts\evaluate.py
Set-Location frontend
npm run build
Set-Location ..
docker compose config --quiet
```

上线前应自行覆盖账户注册/登录/注销、会话归属、断线恢复、风险分流、引用正确性、移动端布局、真实模型响应和目标基础设施连通性。单元测试、离线评测和构建成功只覆盖本地逻辑，不能替代供应商与网关的实机验收。

## 项目文档

- [架构与数据流](docs/architecture.md)
- [设计规范](docs/design.md)
- [开发与安全规范](docs/development.md)
- [评测与质量门槛](docs/evaluation.md)

## 参与项目

提交代码前请阅读 [贡献指南](CONTRIBUTING.md) 和 [安全策略](SECURITY.md)。安全漏洞应通过 GitHub Private vulnerability reporting 私下提交，不要在公开 Issue 中附带患者信息、凭据或完整日志。

参与讨论和提交变更前，请遵守[行为准则](CODE_OF_CONDUCT.md)。

项目代码采用 [MIT License](LICENSE)。知识内容的来源、改编与许可见 [知识内容说明](data/knowledge/README.md)。
