# MedGuide 开发规范

## 目录

```text
backend/app/       FastAPI 接口、领域服务与基础设施适配器
backend/tests/     单元、接口、回归与安全测试
frontend/src/      React 健康信息整理工作台
data/knowledge/    经过来源审核的知识条目
docs/              设计、架构、运行和质量说明
```

## 分层

- API 层只负责 HTTP、认证、SSE 和错误映射。
- 领域层负责状态模型、意图、风险、检索、SQL 安全和工作流节点。
- 基础设施层负责 OpenAI、Milvus、Redis、MySQL 的连接、探针、重试和关闭。
- ingestion 负责清洗、脱敏、切片、版本和 generation 发布；切片不得跨来源版本边界。
- 每个外部依赖必须有明确的不可用状态；不得把夹具、进程内缓存或 hash 向量伪装成生产能力。

## 工作流约束

节点顺序固定为：`normalize → extract_profile → intent → risk → retrieve → structured_query → draft → safety_review → finalize`。

节点接收并返回可序列化状态，副作用必须可观测且可测试。风险为 `high` 时，生成节点只能输出就医指引与急救提示，不得输出诊断、处方或停药建议。

## API 契约

- `POST /api/auth/register`、`POST /api/auth/login`、`POST /api/auth/logout`、`GET /api/auth/me`
- `POST /api/sessions`、`GET /api/sessions/{session_id}`
- `POST /api/chat`，`stream=true` 返回 `text/event-stream`
- `POST /api/search`、`POST /api/query`
- `GET /api/sessions/{session_id}/requests/{request_id}`
- `POST /api/feedback`、`GET /api/metrics`
- `GET /api/health`、`GET /api/ready`

业务接口在生产模式要求有效账户 Cookie 或明确的服务间 Bearer/API Key；服务间主体不得读取或覆盖用户 Cookie 会话。

## 安全底线

- SQL 只允许单条 `SELECT`，表和字段必须在白名单内，参数绑定，强制追加 `LIMIT <= 50`。
- 用户输入、日志、知识条目和测试夹具不得包含真实患者隐私。
- 外部模型只接收最小化、脱敏后的临床摘要；异常日志不得记录密钥、Cookie、原文或完整路径中的敏感字段。
- Cookie 使用 `HttpOnly`、`SameSite=Strict`，HTTPS 部署启用 `Secure`。
- 所有会话读取、反馈和请求恢复都必须校验 owner；未知 owner 统一返回 404。

## 验证要求

后端修改至少补充针对性 pytest；前端修改至少执行 TypeScript/Vite 构建；涉及工作流、认证、部署或安全边界的改动必须补充端到端或契约回归。
