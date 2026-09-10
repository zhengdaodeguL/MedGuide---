# MedGuide 架构与数据流

```text
浏览器工作台
    │ HTTPS / REST / SSE + HttpOnly Cookie
    ▼
Caddy TLS 入口（唯一主机发布端口）
    │
    ▼
Nginx 同源 UI（静态前端、CSP、限流、/api 反向代理）
    │ Compose 隔离网络
    ▼
FastAPI API（认证、归属、幂等、协议门禁）
    │
    ▼
LangGraph 状态图
normalize → extract_profile → intent → risk
    ├──────────────────────────────┐
    ▼                              ▼
混合 RAG                         NL2SQL Guard
Query Rewrite                    只读 / 表字段白名单 / LIMIT
意图路由                         MySQL（配置后启用）
BM25 + Milvus 召回
融合排序、实体一致性、去重压缩
    │                              │
    └──────────────┬───────────────┘
                   ▼
draft → safety_review → finalize
                   │
       引用、风险、摘要、SSE 事件

Redis：会话、幂等、TTL、反馈与共享指标
OpenAI 兼容服务：安全探针与有引用生成
```

## 状态与安全

`WorkflowState` 是节点之间唯一的数据契约，包含会话画像、意图、风险、引用、结构化结果、答案来源、请求结果和耗时。风险节点先于检索和生成；`high` 会短路普通 RAG 与结构化查询，答案只保留就医指引。生产请求必须经过用户名归属校验，并以 `request_id + fingerprint` 防止断线重试重复推进。

外部模型只接收经过白名单压缩的临床摘要和脱敏引用上下文；原始输入不进入 Embedding 或生成请求。认证数据库只保存密码哈希、盐和会话 Token 哈希。

## 运行能力

| 能力 | 生产行为 |
| --- | --- |
| LLM | OpenAI 兼容 Chat Completions；无有效探针时 readiness 失败 |
| 向量检索 | 配置 Milvus 与匹配 Embedding 契约后启用；否则明确为 BM25-only |
| 关键词检索 | BM25 快照；generation 切换原子发布 |
| 业务数据 | 配置只读 MySQL 后启用；未配置不读取夹具数据 |
| 会话与指标 | Redis；带 TTL、CAS、请求幂等和反馈去重 |
| 账户 | 生产使用共享 MySQL 认证库；用户名唯一主键，HttpOnly/Secure/SameSite Cookie |

## 部署边界

Compose 只由 Caddy 发布 80/443，API 与 UI 容器不映射主机端口。Caddy 负责证书申请、HTTP 到 HTTPS 跳转和 HSTS，Nginx 保持 UI 与 `/api` 同源并执行 CSP、认证入口限流和缓存边界。若在 Caddy 前增加组织网关，必须同步收紧代理 CIDR、保留真实客户端地址并验证完整转发链。生产 API 不暴露交互式文档；`/api/health` 与 `/api/ready` 仅供容器内部受控探针使用。
