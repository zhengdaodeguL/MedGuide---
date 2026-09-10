# 知识内容与许可

代码采用仓库根目录的 MIT License。该许可不替代外部资料各自的许可、署名要求和地域限制。

`catalog.json` 当前含15条健康信息摘要，并非完整临床知识库。中文整理、删节、就诊方向映射和急救号码地域化均由 MedGuide 项目负责；这些改编未经来源机构或医学专业人员审核，也不代表其认可或背书。部署者需要完成医学审核、来源复核、地域适配和复审计划。

## OGL 内容

标记 `license: OGL-3.0` 的条目包含依据公开网页文字整理的中文改编。

**Contains public sector information licensed under the Open Government Licence v3.0.**

- [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)
- [原站点使用条款](https://www.nhs.uk/our-policies/terms-and-conditions/)
- [不允许再利用的内容清单](https://www.nhs.uk/our-policies/terms-and-conditions/content-not-licensed-for-re-use/)

只使用文字信息，不包含标识、图片、视频、嵌入式工具或第三方受限内容。界面的“查看原文”用于核对原始页面，不应解释为来源机构对中文摘要的审核。转载本目录的改编内容时应保留上述英文声明和许可链接。

2026-09-10 复核并整理了成人发热、成人布洛芬、氯雷他定、咳嗽、头痛、腹泻呕吐及尿路感染相关页面。各条目保存原文地址；`updated_at` 是页面标注的复核日期，`collected_at` 是本项目读取日期，二者均不代表本项目医学审核日期。较早的感冒及就诊方向映射条目将尚未重新核验的来源日期明确标为待复核。

## 其他条目

血常规、影像检查、远程咨询与急救摘要保留其原文链接和来源说明。它们仅是项目编写的短摘要，不包含第三方百科、药品数据库、图片或完整文章的转载许可。部署或扩充这些内容前，应逐页核验授权与当前内容；不得将本站源码 MIT 许可作为转载外部内容的依据。

MedGuide 自有的服务边界规则适用代码仓库许可。任何新增、替换或修订目录内容后，都需要重建 BM25 索引，并重新验证相关检索与风险分流。
