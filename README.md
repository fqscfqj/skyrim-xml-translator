# Skyrim XML Translator Agent

## 简介

一个面向 Skyrim 模组 XML 文本的翻译与术语管理工具，结合 GUI 与 RAG（检索增强生成）流程，旨在通过向量检索和术语库确保专有名词和术语的一致性与可控性，加快本地化工作流。

这是一个基于 PyQt6 和 LLM (大语言模型) 的 Skyrim XML 文件翻译工具。它利用 RAG (检索增强生成) 技术来确保术语的一致性，特别适合用于翻译 Skyrim 模组的 XML 文本文件。
这是一个基于 PyQt6 和 LLM (大语言模型) 的 Skyrim XML 文件翻译工具。它利用 RAG (检索增强生成) 技术来确保术语的一致性，特别适合用于翻译 Skyrim 模组的 XML 文本文件。
## 功能特点

* **图形用户界面 (GUI)**: 直观的 PyQt6 界面，方便操作。
* **智能翻译**: 集成 OpenAI 兼容的 LLM API 进行高质量翻译。
* **RAG 术语库支持**:
  * 使用向量嵌入 (Embeddings) 检索相关术语，确保专有名词翻译一致。
  * 支持导入 CSV 格式的术语表。
  * 支持手动添加、删除和搜索术语。
  * 支持重建向量索引。
* **XML 文件处理**: 专门针对 Skyrim 模组的 XML 格式进行解析和保存。
* **多线程处理**: 支持多线程翻译和向量化，提高处理速度。
* **高度可配置**:
  * 自定义 LLM 模型参数 (Temperature, Top-p 等)。
  * 自定义 Embedding 模型参数。
  * 支持 "Default" 和 "NSFW" 等不同的提示词风格。
  * 可调整并发线程数。

## 安装
## Recent Fixes
- Fixed: LLM optional parameters could be enabled but became unchangeable in the UI on some platforms; the checkbox now uses the boolean `toggled` signal to enable and disable the related parameter controls reliably (see `src/gui_main.py`).

1.  确保已安装 Python 3.8 或更高版本。
2.  克隆或下载本项目。
3.  安装依赖库：

```bash
pip install -r requirements.txt
```

依赖列表:
* PyQt6
* openai
* lxml
* numpy
* scikit-learn

## 使用说明

1. **启动程序**:
   运行 `main.py` 启动应用程序：

   ```bash
   python main.py
   ```

2. **配置设置**:
   * 进入 "设置" 标签页。
   * 配置 **LLM 设置** (Base URL, API Key, Model Name)。
   * 配置 **Embedding 设置** (用于 RAG 功能)。
   * 根据需要调整线程数和其他参数。
   * 点击 "保存配置"。

3. **术语管理 (可选但推荐)**:
   * 进入 "术语管理" 标签页。
   * 可以导入现有的术语表 (CSV 格式) 或手动添加术语。
   * 点击 "重建/更新向量索引" 以确保 RAG 引擎生效。

4. **开始翻译**:
   * 进入 "翻译任务" 标签页。
   * 点击 "浏览" 选择要翻译的 XML 文件。
   * 文件加载后，点击 "翻译全部" 或选中特定行点击 "翻译选中"。
   * 翻译完成后，点击 "保存文件" 或 "另存为" 保存结果。

## 文件结构

* `main.py`: 程序入口点。
* `src/`: 源代码目录。
  * `gui_main.py`: 主界面逻辑。
  * `llm_client.py`: LLM API 客户端。
  * `rag_engine.py`: RAG 引擎和向量检索逻辑。
  * `xml_processor.py`: XML 文件解析与处理。
  * `translator.py`: 翻译核心逻辑。
  * `config_manager.py`: 配置管理。
* `config.json`: 配置文件 (自动生成)。
* `glossary.json`: 术语库存储文件。
* `vector_index.npy` & `terms_index.json`: 向量索引文件。

## 注意事项

* 首次使用前请务必配置正确的 API Key 和 Base URL。
* RAG 功能依赖于 Embedding 模型，请确保 Embedding 服务可用。