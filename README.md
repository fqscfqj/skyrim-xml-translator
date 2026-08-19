# Skyrim XML Translator Agent

## 简介

这是一个面向 Skyrim 模组文本的本地化工具，提供 PyQt6 图形界面、OpenAI 兼容 LLM 翻译、术语库检索和 XML/MCM 文件处理能力。

当前版本的 RAG 流程以关键词拆解、向量召回和关键词加权为主，召回强度主要由以下参数控制：

- `rag.short_term_max_results`
- `rag.long_term_max_results`
- `rag.short_term_max_chars`
- `rag.min_vector_score`

近期新增的配置项：

| 配置键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `cache.cache_ttl_hours` | float | 0 | 翻译缓存 TTL（小时），0=永不过期 |
| `general.prompt_cache_warmup_enabled` | bool | true | 串行完成前两个真实工作单元，再放开并发，使 DeepSeek V4 能识别并落盘公共前缀；不增加 API 请求 |
| `rag.format_extra_retries` | int | 2 | 格式错误时的额外重试次数 |
| `rag.latin_ratio_threshold` | float | 2.0 | 拉丁字符比例阈值（α > CJK × 阈值时触发未翻译告警），值越大容忍度越高 |

### 质量检查与重试

翻译管线在 LLM 响应后执行三层质量检查：

1. **未翻译检测**：空译文、原文与译文一致（含标识符/专有名词豁免）、containment、拉丁字符比例超标
2. **格式保护**：XML 标签、占位符数量/类型必须与原文匹配
3. **占位符残留**：检测译文末尾残留的拉丁单字母（如 `% s`→`%s`）

重试策略：
- 网络层重试由 `src/llm/retry.py` 处理（速率限制/服务端错误/连接错误/超时，各自独立的退避策略）
- 质量层重试由 `Translator.translate_text()` 处理：最多 `max_retries` 次（默认 2），格式错误额外 +`format_extra_retries` 次
- 重试提示根据问题类型自动选择模板（未翻译/格式错误/片段保留）

## 功能特点

- 图形用户界面：提供翻译任务、术语管理、配置管理和 RAG 调试可视化。
- 智能翻译：接入 OpenAI 兼容接口，支持主翻译模型、搜索模型和后备搜索模型；内含多层质量检查（未翻译检测、格式保护、占位符校验）和智能重试。
- RAG 术语库：支持 CSV 导入、手动维护术语、向量索引重建和术语一致性检索。
- 文件处理：支持 Skyrim XML 文件和 MCM 文本文件。
- 多线程：翻译和向量化均支持并发执行。
- 可配置：支持模型参数、Embedding 参数、提示词风格、语言选项、缓存 TTL、质量检查阈值和重试策略。
- 翻译缓存：跨会话持久化，支持 TTL 过期和缓存质量守卫（自动拒绝低质量缓存条目）。
- 格式保护：XML 标签、占位符在翻译前后通过哨兵机制保护，避免 LLM 破坏结构。
- 中英混合容忍：quality checker 在译文同时包含 CJK 和拉丁字符时降低 severity，避免误拒合理混合翻译。

## 安装

1. 安装 Python 3.8 或更高版本。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖：

- `PyQt6`
- `openai`
- `lxml`
- `numpy`
- `scikit-learn`
- `pyinstaller`

## 使用说明

启动程序：

```bash
python main.py
```

基本流程：

1. 在“设置”页填写 LLM、搜索模型和 Embedding 配置并保存。
2. 在“术语管理”页导入或维护术语，必要时重建索引。
3. 在“翻译任务”页加载 XML 或 MCM 文件，执行整页或选中翻译。
4. 翻译完成后保存或另存为输出文件。

## 仓库结构

- `main.py`：桌面程序入口。
- `src/gui_main.py`：主界面和交互逻辑。
- `src/config/manager.py`、`src/config/schema.py`：配置加载、迁移和 schema。
- `src/llm/client.py`、`src/llm/retry.py`、`src/llm/cost_tracker.py`：LLM 客户端、重试和计费统计。
- `src/rag/engine.py`、`src/rag/search.py`、`src/rag/vector_store.py`、`src/rag/keyword_extractor.py`、`src/rag/glossary_manager.py`：RAG 流程与索引管理。
- `src/translation/translator.py` 及同目录辅助模块：翻译主流程、提示词构建、响应解析和质量检查。
- `src/xml_processor.py`：XML 读写。
- `src/mcm_processor.py`：MCM 文本读写。
- `src/prompt/prompt_manager.py`：提示词模板加载。
- `prompts/`：可编辑的提示词模板。
- `locales/`：界面文案本地化。
- `assets/`：图标和图片资源。

运行期默认使用以下文件路径：

- `config.json`：用户配置文件，首次运行时自动生成。
- `glossary/glossary.json`：术语表。
- `glossary/vector_index.npy` 和 `glossary/terms_index.json`：向量索引。
- `cache/`：翻译缓存和向量缓存。

## 打包为 Windows 可执行文件

当前受支持的构建入口是 `build_exe.py`。

`build_exe.py` 会在存在时包含以下资源：

- `locales/`
- `prompts/`
- `config.json` 或 `config.example.json`
- `assets/`

默认不会把运行期生成的术语索引、缓存或日志目录打进包内；这些内容应由用户在运行后生成或放置。

快速打包：

```powershell
python build_exe.py --onefile --windowed
```

或直接运行：

```powershell
python build_exe.py --onedir --console
```

## 日志说明

- 应用会尝试把常规日志写入 `general.log_file` 指定路径，默认是 `logs/app.log`。
- 当默认路径不可写时，会回退到用户目录或临时目录下的应用日志位置。
- 崩溃日志会单独写入 `crash.log`，也会采用同样的回退策略。
- 使用 `--console` 打包或直接以脚本方式运行时，更容易观察启动期错误。

## 注意事项

- 首次使用前请确认 API Key、Base URL 和模型名称配置正确。
- RAG 功能依赖 Embedding 服务；若向量索引不存在，可在软件内重建。
- `prompts/` 下的 JSON 模板是可编辑资源，修改后会在运行期自动重载。

## DeepSeek V4 提示词设计

项目内置提示词按 DeepSeek V4 的“首请求轨迹敏感”特性组织，但不依赖某个思维链首句：

- **稳定任务锚点**：公共 system prompt 从单一交付目标开始，使用“锁定约束 → 必要消歧 → 目标语重组 → 一次核验”的最小闭环；只分析会改变译文的真实歧义，不复述任务、原文或规则。核验后立即提交最终 JSON，而不是在内部形成答案后直接停止生成。
- **输出与推理解耦**：最终响应只允许约定 JSON，不要求模型展示逐步推理、列出备选或复述规则。思考模式仍可在 API 的 `reasoning_content` 中工作，但不会污染可解析结果。
- **动态上下文后置**：稳定规则位于前缀，文体、术语和原文按稳定到动态的顺序排列；原文被明确标成待处理数据，避免其中的对话或伪指令改变任务轨迹。
- **局部修正重试**：质量重试只重做错误涉及的判断，保留已正确的语义与格式，不再笼统要求从头完整推演。
- **低复杂度任务短路**：关键词提取只做一次实体候选判定，去重后立即返回；关键词与纯 JSON 重封装均完全遵循用户配置的模型参数，不在运行时覆盖思考、深度、采样或输出长度。

这些策略参考了社区项目 [dsh-anchored-standard](https://github.com/xiaobright/dsh-anchored-standard) 的“最小首轮条件、延迟动态上下文”实验。该项目也明确指出，`We need` 等词法轨迹不等同于能力提升，独立复现的能力收益仍不确定；因此本项目以 JSON 合规率、格式/语义检查、重试率、延迟和 token 用量作为实际验收指标，而不检测或诱导固定思维链措辞。实验边界可参阅其[工具面剂量研究](https://github.com/0liveiraaa/DeepseekCotexplorations/tree/main/contributions/xiaobright-v4-tool-surface-dose-response)。

## 成本优化建议

- **国产模型前缀缓存**：DeepSeek 的上下文硬盘缓存会自动运行，只命中从第 0 个 token 开始相同的完整缓存前缀单元；无需发送厂商专属参数。程序将稳定的核心规则放在请求开头，把术语和原文等动态内容放在末尾，并默认串行完成前两个真实工作单元，让 DeepSeek 识别并落盘公共前缀，第三条起再释放其余并发。详见 [DeepSeek 上下文硬盘缓存](https://api-docs.deepseek.com/zh-cn/guides/kv_cache)。
- **缓存命中可观测**：程序同时识别 DeepSeek 的 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` 和常见 OpenAI 兼容服务的 `prompt_tokens_details.cached_tokens`，翻译任务结束时会在日志中输出本次输入 token 缓存命中率。
- **关键词提取使用更便宜的模型**：`llm_search` 配置项用于 RAG 关键词提取，这是一个结构化的低复杂度任务，不需要高端模型。建议将 `llm_search.model` 配置为轻量模型，而 `llm.model` 保留高质量模型用于翻译。关键词提取约占总 LLM 调用量的 50%，使用更便宜的模型可以在不影响翻译质量的前提下降低成本。
- **统一关键词提取链路**：所有启用 RAG 的原文都会先经过关键词提取，再进入术语检索；若关键词提取结果为空或服务暂时不可用，仍会回退到本地规则兜底，避免完全失去术语召回。
- **短文本批处理保持可选**：批处理能进一步减少请求数，但可能改变模型处理单条文本时的注意力分配，因此默认仍关闭；质量优先场景无需启用。

## 多供应商思考深度控制

设置页为翻译、关键词和兜底模型分别提供三个统一控件：思考控制协议、思考模式和思考深度。协议默认根据 Base URL 与模型名自动识别，也可手动锁定。客户端会转换成供应商实际使用的请求结构：

| GUI 协议 | 请求映射 |
|---|---|
| OpenAI / Meta / 标准兼容 | 顶层 `reasoning_effort` |
| DeepSeek | `thinking.type` + 顶层 `reasoning_effort` |
| DashScope / Qwen | `enable_thinking` + `reasoning_effort` |
| OpenRouter | 统一 `reasoning.effort` |
| Claude 深度 / 自适应 | `output_config.effort`；显式开启时使用 `thinking.type=adaptive` |
| Gemini OpenAI 兼容层 | 顶层 `reasoning_effort` |

项目不设置思考 Token 预算，也不会把统一深度暗中转换为固定 Token 上限；模型按提示词约束完成任务。对于不能关闭思考的 Gemini/Claude 型号，“关闭”会映射为该模型支持的最低档。若 API 以 400/422 明确拒绝思考参数，程序会记录警告并仅重试一次供应商默认设置，避免因为可选参数导致整条翻译失败。

配置文件仍保留旧的 `enable_thinking` / `reasoning_effort`，现有配置无需迁移：

```json
{
  "llm": {
    "parameters": {
      "reasoning_protocol": "auto",
      "enable_thinking": true,
      "reasoning_effort": "high"
    }
  }
}
```

供应商支持的档位并不相同。DeepSeek V4 原生使用 `high/max`，兼容值会映射到这两档，且思考模式下采样参数不生效；Muse Spark 的有效最高档是 `high`，`xhigh` 与其等效；Gemini 的 OpenAI 兼容层直接接受 `reasoning_effort`，但部分模型不能关闭思考；Claude 新模型已从固定 `budget_tokens` 迁移到 effort / 自适应思考。参阅 [DeepSeek 思考模式](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/)、[OpenAI API 参数](https://platform.openai.com/docs/api-reference)、[Meta Muse Spark reasoning cookbook](https://github.com/meta-models/meta-model-cookbook/blob/main/01_api_fundamentals/06_reasoning_tokens.ipynb)、[Gemini OpenAI 兼容说明](https://ai.google.dev/gemini-api/docs/openai)、[Claude effort 说明](https://platform.claude.com/docs/en/build-with-claude/effort)、[DashScope 参数参考](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)及 [OpenRouter reasoning 统一参数](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)。

Claude 协议选项用于能传递 Anthropic 原生思考字段的 OpenAI 兼容网关；本项目使用 OpenAI SDK，不会把 Anthropic 的原生 Messages API 自动转换成 Chat Completions API。

推荐把翻译与关键词提取分开配置：

```json
{
  "llm": {
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-v4-pro",
    "parameters": {
      "reasoning_protocol": "auto",
      "enable_thinking": true,
      "reasoning_effort": "high"
    }
  },
  "llm_search": {
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-v4-flash",
    "parameters": {
      "reasoning_protocol": "auto",
      "enable_thinking": false
    }
  }
}
```
