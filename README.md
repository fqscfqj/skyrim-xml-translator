# Skyrim XML Translator Agent

## 简介

这是一个面向 Skyrim 模组文本的本地化工具，提供 PyQt6 图形界面、OpenAI 兼容 LLM 翻译、术语库检索和 XML/MCM 文件处理能力。

当前版本的 RAG 流程以关键词拆解、向量召回和关键词加权为主，召回强度主要由以下参数控制：

- `rag.short_term_max_results`
- `rag.long_term_max_results`
- `rag.short_term_max_chars`
- `rag.min_vector_score`

## 功能特点

- 图形用户界面：提供翻译任务、术语管理、配置管理和 RAG 调试可视化。
- 智能翻译：接入 OpenAI 兼容接口，支持主翻译模型、搜索模型和后备搜索模型。
- RAG 术语库：支持 CSV 导入、手动维护术语、向量索引重建和术语一致性检索。
- 文件处理：支持 Skyrim XML 文件和 MCM 文本文件。
- 多线程：翻译和向量化均支持并发执行。
- 可配置：支持模型参数、Embedding 参数、提示词风格、语言选项和缓存设置。

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

当前受支持的构建入口是 `build_exe.py`，`build_windows.ps1` 只是 Windows 下的辅助包装脚本。

`build_exe.py` 会在存在时包含以下资源：

- `locales/`
- `prompts/`
- `config.json` 或 `config.example.json`
- `assets/`

默认不会把运行期生成的术语索引、缓存或日志目录打进包内；这些内容应由用户在运行后生成或放置。

快速打包：

```powershell
./build_windows.ps1 -OneFile:$true -Windowed:$true
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

## 成本优化建议

- **关键词提取使用更便宜的模型**：`llm_search` 配置项用于 RAG 关键词提取，这是一个结构化的低复杂度任务，不需要高端模型。建议将 `llm_search.model` 配置为轻量模型（如 `gpt-4o-mini`、`deepseek-chat`），而 `llm.model` 保留高质量模型用于翻译。关键词提取约占总 LLM 调用量的 50%，使用更便宜的模型可以在不影响翻译质量的前提下显著降低成本。
- **统一关键词提取链路**：所有启用 RAG 的原文都会先经过关键词提取，再进入术语检索；若关键词提取结果为空或服务暂时不可用，仍会回退到本地规则兜底，避免完全失去术语召回。
