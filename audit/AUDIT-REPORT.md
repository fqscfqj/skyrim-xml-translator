# Skyrim XML Translator 全量审查报告

- 审查对象：`D:\github\skyrim-xml-translator`，提交 `d305b00`（工作区干净）
- 审查日期：2026-11-04
- 代码规模：64 个 Python 文件 / 约 20,261 行；PyQt6 + lxml + numpy + openai
- 基线：`python -m pytest tests -q` → **254 passed, 51 subtests passed（1.5s）**
- 方法：15 个只读子代理并行审计（翻译管线、RAG 检索、向量索引、词库导入、LLM 传输、GUI 性能、GUI 状态机、提示词/质量、缓存/配置/日志、XML/ESP、MCM、运行时实测、红队模糊测试、端到端提示词模拟），主代理独立复核关键结论；全部实验在 `%TEMP%` 完成，仓库未改动。
- 标记：`[实测]` 有脚本复现/计时；`[复核]` 主代理亲自验证过代码；`[审查]` 代码路径确认。

> **最重要的结论：当前 HEAD 无法打开文件。** `MainWindow` 引用了 4 个从未定义的类常量，`browse_file / dropEvent / 分页填表` 三条主路径都会抛 `AttributeError`。详见 P0-01。254 个测试全绿，是因为没有任何测试实例化 `MainWindow`。

---

## 0. 执行摘要

### 0.1 五个必须先处理的问题

1. **打开/拖入文件必崩**（P0-01）：`SUPPORTED_DROP_SUFFIXES`、`XML_SIZE_WARN_BYTES`、`XML_SIZE_HARD_LIMIT_BYTES`、`TRANS_TABLE_PAGE_SIZE` 全仓库无定义，却在 `browse_file → load_xml_to_table`、`dragEnterEvent/dropEvent`、`_fill_trans_table_paged` 中被访问。`[复核]`
2. **跨文件数据污染**（P0-02）：翻译进行中加载另一个文件，旧 Worker 的排队结果会按“行号”写进新文件的 `<Dest>`，保存即损坏新文件。`[审查]`
3. **关窗即丢本轮译文并可能崩溃**（P0-03）：`closeEvent` 不 `wait()` 线程、不保存翻译缓存；`main.py` 随即 `sys.exit`，触发 `QThread: Destroyed while thread is still running`。`[实测子代理]`
4. **写文件先截断后校验，编码失败清空目标**（P0-05）：MCM `save_file` 用 `open(path,"w")` 先截断；XML `update_dest` 先清空节点再赋值。两者失败后 GUI 仍可能显示“成功”。`[实测]`
5. **RAG 在两个默认配置下会静默失效**（P0-11）：`get_embedding` 从不发送 `dimensions`，默认 `text-embedding-3-large` 返回 3072 维而配置写 1536 维 → 每条向量被 `_check_vector_dimension` 丢弃、索引为空；而兜底词法检索仍能跑，用户看不到错误。`[复核]`

### 0.2 TOP 20 优先级

| # | 级别 | 模块 | 问题 | 位置 | 修复要点 |
|---|---|---|---|---|---|
| 01 | P0 | GUI | 4 个类常量缺失，打开/拖入/分页必抛 `AttributeError` | `gui_main.py:3054,4962,4966,5044` | 补齐常量 + 覆盖 3 条路径的 GUI 测试 |
| 02 | P0 | GUI | 翻译中加载新文件，旧结果写入新文件 | `gui_main.py:4767,5010-5028,5169-5205` | 加载入口互斥 + 文档代次(epoch)过滤结果 |
| 03 | P0 | GUI/缓存 | 关窗不等线程、不保存缓存；QThread 析构崩溃 | `gui_main.py:3016-3051`；`main.py:266` | stop → wait(3s) → save → close；`aboutToQuit` 兜底 |
| 04 | P0 | XML | `update_dest` 先清空后赋值，非法字符导致 Dest 永久丢失 | `xml_content.py:65-67,122-128`；`xml_processor.py:195-207` | 先校验 XML 字符合法性；失败回滚并标失败 |
| 05 | P0 | MCM | `save_file` 先截断后写，编码失败把目标清成 0 字节 | `mcm_processor.py:166-183` | 先 `encode` 校验，再 tmp+replace；GUI 检查返回值 |
| 06 | P0 | 向量 | 检索线程使用 mmap 时另一线程 `_close_mmap()` → 进程段错误 | `vector_store.py:154-168,1230-1258` | 读写锁/读者计数；或加载为内存数组 |
| 07 | P0 | 向量 | `vectors=None` 且 `terms` 非空时仍增量写入 → 索引错位/错误译文 | `vector_store.py:842-942` | 入口强制全量重建；保存前断言行数一致 |
| 08 | P0 | 向量 | 重建先删旧索引；取消/失败留下半成品且状态 `ready` | `vector_store.py:989-999,1132-1147` | 写临时索引后原子替换；meta 增 `complete/expected` |
| 09 | P0 | 词库 | 主文件+sidecar 非原子、无锁、tmp 名仅 pid → 并发保存大面积失败/三方不一致 | `glossary_manager.py:232-266,386-462` | 单文件提交或 generation；RLock；tmp 加 uuid |
| 10 | P0 | 词库 | CSV `strict=False`：一个落单引号静默吞掉后续所有行 | `glossary_import.py:353-354` | `strict=True` + 捕获 `csv.Error` 带行号 |
| 11 | P0 | LLM/RAG | 不发送 `dimensions`，默认配置下向量全被丢弃 | `llm/client.py:315`；`vector_store.py:676-687` | 显式传 `dimensions`；响应后校验维度 |
| 12 | P1 | 翻译 | 无覆盖率检查；`finish_reason=length` 被丢弃；截断/漏译静默缓存 | `quality_checker.py:110-191`；`client.py:660-667` | 加 `coverage` 规则 + `ResponseTruncatedError` |
| 13 | P1 | 翻译 | 长文本每块不质检、重试原样重发；坏块只能靠随机重采样 | `translator.py:1124-1206` | 逐块 QC + 带证据的局部重试 |
| 14 | P1 | 翻译 | 单词 UI 标签（Health/Armor）被当专名，未译也缓存；OK/HP 反被判错误 | `quality_checker.py:229-242,338-365` | 区分 UI 词表/缩写；identity warning 不缓存 |
| 15 | P1 | 翻译 | 术语三重弱化 + 残留仅 warning → 术语一致性基本不生效 | `default.json:10`；`templates.json:3`；`quality_checker.py:411-440` | 术语默认强制 + 一致性升级为可重试错误 |
| 16 | P1 | 翻译 | 长文重试盲发 + `__FMT_*__` 无类型 + CJK 允许标记重排 | `translator.py:1124`；`text_analyzer.py:32,466` | 分块纠错；哨兵带类型（T/P/N）+ 图例 |
| 17 | P1 | RAG | 多词术语默认被分解；精确短语通路失效 | `keyword_extractor.py:1044-1121`；`schema.py:46` | `keep_original=True`；glossary 命中不分解 |
| 18 | P1 | RAG | 非拉丁文本关键词/词条被整条丢弃（俄语/日语 MOD 静默空转） | `keyword_extractor.py:20`；`glossary_manager.py:20` | Unicode 感知归一化 + 回归测试 |
| 19 | P1 | GUI | 状态着色触发 `itemChanged` 风暴，失败/警告行被改回 success | `gui_main.py:3527-3535,5214-5235` | 着色/编辑信号分离；非 Dest 列直接返回 |
| 20 | P1 | 流畅 | 表头 `ResizeToContents` 使 10k 行填充 30.9s、单块卡 1.36s | `gui_main.py:3310,5032-5074` | 填充期 `Interactive` + 结束后一次 `resizeColumnToContents`；实测 0.19s |

### 0.3 产品级判断

**流畅性**：瓶颈不是网络，而是 GUI 线程上的三次方放大（表头自适应 × 逐行 item 写 × QSS 重设）和 RAG 的重复全表扫描。修复优先级：表格/状态刷新（10k 行从 30.9s→0.2s）> `openai` 惰性导入（启动 −364ms，−73%）> RAG 批量 GEMM（2×）> 词表/索引后台化。

**RAG**：骨架（LLM 关键词 → 向量 + 词法 → 桶限制）合理，但“召回靠唯一通道 + 分数是三种尺度混比 + 索引陈旧检测不到 + rich metadata 全死”。三个最高杠杆：①整篇原文的确定性命中扫描（Aho-Corasick），逐字出现的术语召回接近 100% 且不依赖 LLM；②批量 matmul + 统一融合分/全局重排；③把 `glossary_hash`/`complete` 纳入索引健康判定，配合增量补齐。

**翻译质量**：单句提示词（约 950 token 系统提示）质量明显高于同类工具；天花板由“无完整性度量、术语不强制、重试无证据、长文结构与段落丢失”四件事压低。修完这四类，书籍/任务文本的提升会大于换模型带来的提升。

---

## 1. P0：阻断、崩溃与数据损坏

### P0-01 打开/拖入文件必崩：4 个类常量未定义 `[复核]`

`gui_main.py` 引用但全仓库无定义（`grep` 仅命中使用处）：

- `self.SUPPORTED_DROP_SUFFIXES` → `3054`（`_is_supported_drop_file`，被 `dragEnterEvent/dropEvent` 调用）
- `self.XML_SIZE_HARD_LIMIT_BYTES` / `self.XML_SIZE_WARN_BYTES` → `4962/4966`（`load_xml_to_table`）
- `self.TRANS_TABLE_PAGE_SIZE` → `5044`（`_fill_trans_table_paged`）

`MainWindow` 只继承 `QMainWindow`，无 mixin；`git log -S` 显示它们在提交 `802ba53`（“全仓缺陷修复与硬化”）里被引用却未定义。实测 `hasattr(MainWindow, ...)` 全部 `False`。

影响：点“浏览”选文件 → 槽函数抛 `AttributeError`（PyQt6 默认会中止进程，实测子代理得 exit 127）→ 无提示退出；拖入文件同样崩；即使绕过前两者，分页填表第一轮也必抛。

修复（值按语义与 README 对齐）：
```python
class MainWindow(QMainWindow):
    SUPPORTED_DROP_SUFFIXES = frozenset({".xml", ".txt"})
    XML_SIZE_WARN_BYTES = 50 * 1024 * 1024
    XML_SIZE_HARD_LIMIT_BYTES = 200 * 1024 * 1024
    TRANS_TABLE_PAGE_SIZE = 200
```
并补 3 个测试：`load_xml_to_table`、`_is_supported_drop_file`、`_fill_trans_table_paged`（当前 `tests/` 对这三个名字 0 命中）。

### P0-02 翻译中加载新文件 → 旧结果污染新文件 `[审查]`

`start_translation` 只检查 `_translation_task_active`；`browse_file`（4653）与 `dropEvent`（3137）不检查翻译任务，`load_xml_to_table`（4952）只检查 `_xml_load_worker`。加载完成后 `_on_xml_load_finished` 替换 `self.current_processor` 并重建表格，而旧 Worker 仍以**行号**定位结果：

`update_table_row:5174` 取当前表 `item(row,2)` 的 node（新文件的节点）→ `5195-5199` 调 `self.current_processor.update_dest(node, 旧译文, overwrite=True)`。A 文件的译文被写进 B 文件同号行，保存即损坏 B。

修复：加载入口统一互斥（翻译/术语任务任一活跃则拒绝），并给结果加文档代次：
```python
self._doc_epoch += 1
# Worker 创建时快照 epoch；update_table_row/row_failed 首行：
if epoch != self._doc_epoch: return
```

### P0-03 关窗丢缓存 + QThread 析构崩溃 `[实测]`

`closeEvent`（3016-3051）先 `close_clients()`，再 `stop()`，全程无 `wait()`；`main.py:266-272` 在 `app.exec()` 返回后立即 `sys.exit`。子代理用真实 QThread 复现：`del w` 后打印 `QThread: Destroyed while thread '' is still running` 并非 0 退出。同时 `Worker.run:1068` 是唯一缓存落盘点，中途关窗则本轮全部新译文丢失，下次整篇重译。

修复：
```python
def closeEvent(self, e):
    for w in (self._xml_load_worker, self.glossary_worker, self.worker):
        if w is not None and w.isRunning():
            w.stop()
            if not w.wait(3000):
                self.log("Worker did not stop in 3s; exiting"); break
    try: self.translator.save_translation_cache()
    except Exception: pass
    self.translator.llm_client.close_clients()
    e.accept()
```
另在 `main.py` 的 `app.aboutToQuit` 挂一次兜底保存。

### P0-04 XML：先清空节点再赋值，非法字符导致 Dest 丢失 `[实测]`

`xml_content.set_node_inner_content` 先 `_clear_node_children(node); node.text=None`，随后赋值时才抛 `ValueError: All strings must be XML compatible`。异常后节点已空；GUI 只记日志（`5199-5200`）并继续，之后保存把空 Dest 写盘。

同一路径还有两个数据风险：`set_node_text_content` 同样先清空；`_write_tree_atomic` 失败前已覆盖 `.bak`（见 X-02）。

修复：赋值前过滤 XML 1.0 非法码点（`\x00-\x08\x0B\x0C\x0E-\x1F` 等），或在临时 Element 上构建成功后再替换；`update_dest` 向上抛错，GUI 标 FAILED 而不是继续。

### P0-05 MCM：保存先截断后写，编码失败清空文件 `[实测]`

`mcm_processor.py:166-174`：
```python
with open(output_path, "w", encoding=self.encoding, newline="") as f:
    f.write(content)
```
`open(..., "w")` 先截断；`f.write` 抛 `UnicodeEncodeError` 后被 `except` 吞成 `return False`。实测 cp1252 文件 `$OptionA\tIt’s café – dash` → 43B 变 0B。GUI 自动导出只留一行日志（`5402-5405`），`save_as_xml_file` 还忽略返回值（`5159`）并弹“已保存”。

修复：先 `data = content.encode(self.encoding)`（失败直接返回 False，不碰目标），再写同目录 tmp + `fsync` + `os.replace` + 覆盖前 `.bak`；GUI 检查返回值并提示失败。

### P0-06 向量索引：检索中关闭 mmap → 进程段错误 `[实测子代理]`

`_close_mmap`（154-168）在重建/导入/删除/`reload_embedding_runtime`/`load` 时被调用，但检索线程可能正持有 `np.asarray(self.vectors[s:e])` 视图（1230-1258）。numpy memmap 无公开 `close()`，只能 `_mmap.close()`；关闭后下一次访问即访问违例，子代理实测线程版退出码 139，`except Exception` 捕获不到。

触发链真实存在：`rebuild_index:5503` 只检查 `_glossary_task_active`，不检查翻译任务；`save_config → reload_embedding_runtime → load()` 同样会关闭。

修复（任一）：
1. store 级读写锁：检索整段 `read_lock`，mutation 先拷贝数组、等待读者归零再 close；
2. `load()` 直接把向量读入内存（`np.array`），大索引用容量预算控制；
3. 禁止在有读者时 close，改为“延迟关闭到进程/索引代次切换”。

同步修复 P0-07：`add_vectors_batch` 在 `vectors is None and terms` 时必须拒绝增量并强制全量重建；`save_index_state` 前断言 `len(terms)==vectors.shape[0]`。

### P0-07/08 索引失步与“先删后建” `[实测子代理]`

- `load()` 因 size_mismatch/npy 损坏把 `vectors=None` 但保留 `terms`；`add_vectors_batch` 只拒绝 `invalid_vector_shape/dimension_mismatch`，于是把“旧 terms + 仅新向量”落盘 → 重载后 size_mismatch、向量全卸载（语义检索永久关闭）；重载前则把第 0 行向量配 `terms[0]`，返回**错误词条的译文**。
- `build_index(force_full=True)` 在第一次嵌入前 `clear_index(delete_files=True)`；API key 错/断网/维度不符时旧索引已消失。取消（`stop_flag` 全仓库从未被置 True）或失败后仍写 checkpoint + 完整 glossary hash，`_evaluate_index_status` 不检查 `glossary_hash/complete`，把 20% 的索引标成 `ready`。

修复：写 `index.new.npy/.new.json`，成功（或达到阈值）后原子替换；meta 增加 `complete`、`expected_term_count`、`built_terms`、`failed_terms`；状态机比对 `glossary_hash` 并暴露“部分构建”；`GlossaryWorker.stop()` 同步 `rag_engine.stop_flag`。

### P0-09 词库主文件 + sidecar 非原子、无锁 `[实测子代理]`

`add_terms_batch` 先 `glossary.update()`，再写 sidecar，再 `save()`，最后 `rebuild_lookup()`；任一步失败即内存/磁盘/lookup 不一致。子代理实测：把 glossary.json 设只读后导入，主文件未变、sidecar 已写入、内存已生效、`lookup_normalized('new')` 为 None。并发同进程两线程各 `add_term` 30 次：**60 次里 57 次 PermissionError**（`tmp_path = f"{path}.tmp.{os.getpid()}"` 相同，互相覆盖/删除）。

修复：进入前快照 `glossary/rich_meta`，异常整体回滚，`finally: rebuild_lookup()`；tmp 名加 `uuid4().hex`；进程内 `RLock` + 跨进程文件锁；`os.replace` 对 WinError 5/32 退避重试 3 次；长期建议把 rich meta 放进同一 JSON（或加 `generation`），一次 `os.replace` 提交。

### P0-10 CSV 静默丢行/错库 `[实测子代理]`

- `csv.reader` 未 `strict=True`：`A,"甲\nB,乙\nC,丙\n` → `total_rows=1`，B/C 永久丢失，`invalid_rows=0`，预览显示“有效 1 条”。
- 分隔符启发式按前 8192 字符的原始字符计数（`;` > `,` 就选分号），引号/译文里的分号也计入：`Sword,剑;刀;刃;锋` 名为 `.csv` 时 → `{'Sword,剑': '刀'}`；同字节改名 `.txt` 则正确。结果依赖扩展名，错译文直接进提示。

修复：`strict=True`，捕获 `csv.Error` 转 `GlossaryImportError` 并带行号；`csv.field_size_limit` 提高；分隔符对 `, \t ; |` 逐个试解析，按“列数一致率最高且≥2 列”选择，`.csv` 不因字符计数切分号。

### P0-11 Embedding 维度未发送，RAG 静默失效 `[复核]`

`get_embedding`（`client.py:315`）调用 `embeddings.create(input=text, model=model)`，**从不传 `dimensions`**；默认配置 `text-embedding-3-large` + `dimensions: 1536`，OpenAI 实际返回 3072 维 → `_check_vector_dimension` 逐条 WARNING 拒绝、`failed_terms=total`、什么都不落盘；旧索引又已被重建删除。指纹记录的是“配置维度”，无法自愈。

修复：请求加 `dimensions=int(cfg)`（仅对 `text-embedding-3-*` 等支持模型）；首个响应校验 `len(vec)==dimensions`，不一致 ERROR + 一次性提示；或探测实际维度并写回 fingerprint。

---

## 2. 流畅性（启动 / UI / 线程 / I/O）

### 2.0 本机实测基线

| 项目 | 规模 | 实测 | 说明 |
|---|---|---|---|
| `import src.gui_main` | — | **442ms**（`openai` 364ms，73%） | 打桩 openai 后 126ms |
| `GlossaryManager.load` | 5k/20k/100k | 17/75/**455ms** | 每术语归一化 2 次 + 全量指纹 |
| `VectorStore.load`（含 trigram） | 5k/20k/100k | 20/82/**471ms** | trigram 400-522ms、73.4MB@100k |
| 首次 RAG 搜索（懒建 token_df） | 100k | **279ms** | 含 174ms token_df |
| `RAGSearcher.search`（5 keyword） | 5k/20k/100k | 3.1/**19.4**/**92ms** | 93% 在逐 keyword 全表余弦 |
| `Translator.translate_text`（假 LLM） | 20k 索引 | **20.5ms/条**（426 字 24.3ms） | RAG 占 20ms |
| `QualityChecker.check` | 426/5124 字 | 2.49/**40.6ms** | 78% 在逐空白大正则 |
| `build_protected_format_shell` | 426/5124 字 | 1.15/**19.5ms** | 同上 |
| GUI 单行结果更新（offscreen） | 每行 | **129.8µs**（124.6µs 是 5×setStyleSheet） | 5k 行 649ms |
| `Worker.run` 进度求和 | 1k/5k/20k | 21/**620**/**13,798ms** | O(n²) |
| 分页填表 10k 行 | 现状/改 Interactive | **30.9s / 0.19s** | 单块 1.36s → 5.6ms |
| 词表搜索框每击键 | 20k/100k | 3.8/**21.4ms** | 每次全表 f-string |
| 主题切换/保存配置（10k 行） | — | **1.98s** | 重排 + 逐行 restyle |
| EmbeddingCache 满载 | 4809×1536 | **283MB** | 存 `list[float]`；float32 仅 30MB |

### 2.1 UI

- **F-01 表头 `ResizeToContents`（P0 性能）** `gui_main.py:3310`：分页填充时每写第 0 列都让表头对已有行全列重算，10k 行 30.9s、单块 1.36s，Windows 判“未响应”。改 `Interactive` + 填完 `resizeColumnToContents(0)` 一次（实测 0.19s）；根本方案是 `QAbstractTableModel`。
- **F-02 逐行重设 5 个筛选气泡 QSS（P1）** `3469,3492,3408`：`setStyleSheet` 占单行成本 96%（0.112/0.14ms）。计数用 100-200ms `QTimer` 合并；气泡样式只在筛选/主题变化时刷。实测 10k 行 3.8s → 0.3s。
- **F-03 `itemChanged` 风暴（P1）** `3527-3535`：一次 `_set_row_status` 触发 12 次 `itemChanged`；全选 10k 行时 `update_translate_buttons_enabled` 每次 `selectedItems()` 5.32ms，每行约 32ms。非 Dest 列直接返回 + 着色时 `blockSignals`（详见 P1-19）。
- **F-04 进度 O(n²)** `1031`：每个 future 对全部已完成 key 重求和；20k 行 13.8s 纯 CPU，和 GUI 抢 GIL。改 `completed_rows += len(all_rows)`。
- **F-05 重复行逐条 emit** `969-976`：一个 API 结果按 `all_rows` 逐行发信号，每个都跑完整 `update_table_row`；改为批量信号 + `setUpdatesEnabled(False)`。
- **F-06 分页填充无代次令牌** `5032-5074`：旧填充链的 `singleShot` 会在二次加载时续填新数据并提前恢复 `blockSignals`/汇总刷新；加 `_fill_token`。
- **F-07 空表点“翻译全部”空跑** `4776-4781`：`load_xml_to_table` 只 `start()` worker 就返回 True，表格仍 0 行 → 立即“没有可翻译内容”；改成 `_pending_translate`，填表完成后自动开始。
- **F-08 词表搜索 O(N)/键** `5454-5469`：100k 术语 21.4ms/键；缓存小写显示串 + 150ms 去抖或 `QSortFilterProxyModel`。
- **F-09 导入预览在 UI 线程同步解析** `5542`：20k 行 28.5s（解析本身近二次方），worker 再解析一遍；复用 `GlossaryWorker(dry_run=True)`，并把 `parsed` 直接传给正式导入。
- **F-10 新增术语在 UI 线程同步 embedding** `5423 → engine.py:199-212`：网络慢/失败时按钮卡到超时；失败仍提示“已添加”且词条无向量。改后台任务 + `mark_index_incomplete()`。
- **F-11 RAG 可视化在 GUI 线程发 LLM 请求** `1778`：缓存被文件加载清空后必然走无缓存分支，对话框构造期同步 HTTP，最长 120s；改 QThread + 先显示骨架。
- **F-12 主题切换 2s** `2499-2503,5845`：逐行 `_apply_row_status_style`；改 model/view 或先卸表再装回（实测 10k 行 213ms）。
- **F-13 启动同步构建 RAGEngine** `2178`：30k 术语 `MainWindow()` 601ms（RAG 493ms），100k 约 1.6-2s；窗口先 `show()`，`QTimer.singleShot(0, _lazy_init_rag)`。
- **F-14 日志热路径**：每条 51µs（路径解析 + 每行 `makedirs` + flush），10 万行 ≈5.1s；`_resolve_log_file_path`/目录创建应缓存/只做一次。工作线程调用 `MainWindow.log` 会跨线程 `QTimer.start()`（Qt 禁止），应改 signal。
- **F-15 日志轮转并发丢行** `logging_helper.py:128-147`：直接 `handler.emit` 绕过 `Handler.handle` 的锁，8 线程 1200 行丢 7 行；改 `handler.handle(record)` 或自建锁。
- **F-16 缓存/配置保存**：`LRUCache.save_to_disk` 每条先 `json.dumps` 试探再整体 dump（两遍）；并发 save 共享 `.tmp.<pid>`，8 线程 200 次实测 154 次 WinError 32 被吞。加锁/线程唯一 tmp/重试。
- **F-17 词表/索引内存**：100k trigram 73MB、`EmbeddingCache` 283MB（float32 可 30MB）、翻译缓存 50k 30MB；均建议按预算收敛。

### 2.2 启动

- **F-18 `openai` 全量导入**（P1 启动）：`client.py:5` + `retry.py:10` + `llm/__init__.py:1` 形成 `rag → retry → llm.__init__ → client → openai` 链，启动 364ms/73%。改为函数内导入、`classify_error` 内惰性导入、`__getattr__` 导出；预期 442→126ms。
- **F-19 词表加载全量归一化 + 指纹**：100k 455ms，`rebuild_lookup` 每术语归一化 2 次 + `json.dumps+sha256` 全量；复用 normalize 结果、增量指纹。
- **F-20 trigram 在 load 同步构建**：100k 400-522ms/73MB，只为偶尔的 containment；改首次查询懒构建或后台预热。
- **F-21 首条翻译额外 174ms**：`token_df` 懒建在首查同步执行；后台化。

### 2.3 超线性正则（可被恶意/异常文本触发，worker 无超时）

| 输入 | 耗时 | 位置 |
|---|---|---|
| `"%"+"0"*20,000` | **7.76s**；32k → **20.7s** | `text_analyzer.py:39-45` `_PERCENT_PLACEHOLDER_PATTERN` |
| 8,000 个未闭合 `<!--`（432KB） | **8.4s** | `text_analyzer.py:35` `_COMMENT_CDATA_PATTERN` |
| `"[a"*20,000` | 236ms | `_BRACKET_TOKEN_PATTERN` |
| `"<"*20,000` | 118ms | `<[^>]*>` |
| `"{"*40,000`（`_JSON_EXTRACT_RE`） | 316ms | `response_parser.py:17` |
| 166KB `{"translation":"xxx…` 重复 | 665ms | `response_parser.py:258` 对每个 `{` 做 `raw_decode` |

修复：量词加界（`[#0\-+]{0,8}`、`{1,4096}` 等）、注释/标签/括号加长度上限、JSON 抽取改 `find` 切片并限制候选 `{` 数、入口按 `long_text_chunk_threshold_chars` 闸门。RAG 模块的 `keyword_extractor` 对超长原文也没有截断（长文会整篇发关键词请求）。

---

## 3. RAG：召回、排序、索引、元数据

### 3.1 召回（漏检）

- **R-01 多词术语默认被分解（P0 功能）** `keyword_extractor.py:1044-1121`：`keyword_task_keep_original=False`（`schema.py:46`），`Erik the Slayer → ['Erik','Slayer']`；`_resolve_direct_match_term` 再无整句命中机会，只能靠“唯一候选”或余弦。修复：默认 True；glossary 命中短语禁止分解；仅 >3 token 且未命中时分解。
- **R-02 非拉丁文本被整条丢弃（P1）** `keyword_extractor.py:20` / `glossary_manager.py:20`：`[^0-9a-zA-Z\u4e00-\u9fff]+` 会把西里尔/假名全部替换为空；实测 `_keyword_appears_in_text("Мёртвый", …) == False`，俄/日 MOD 的 RAG 静默全空。修复：`re.sub(r"[^\w\u4e00-\u9fff]+"," ",s)` + NFKC + ru/fr/ja 回归测试。
- **R-03 精确术语命中被低信号过滤误杀（P1）** `keyword_extractor.py:991-1002`：`-ly`/常用词规则先于 `direct_glossary_hit` 判断，`Rally`（glossary 有）被丢；`Watch` 因在 `_COMMON_WORDS` 被丢。把 direct hit 判断提前一行。
- **R-04 正则兜底 O(全词表) + 噪声（P1）** `_extract_proper_nouns_regex:1213-1280`：40k 词条 760ms/条；`is_signal_token` 对 df=0 返回 True，导致 `Press MCM to open the menu. → ['Press','MCM','menu']`、`Act III`。改用 `lookup_token_candidates` 倒排 + 句首/缩写/罗马数字排除。
- **R-05 复数不命中（P1）** `_raw_term_appears_in_source:412-421` 无词形边界：术语 `Sword` 对 `Swords are sharp.` 判 False，连带 source bonus、sentence-like 过滤、token 消歧三处失效。复用 `prompt_builder._build_source_term_variants`。
- **R-06 形态学回退假阳性（P1）** `_build_query_lookup_variants:452-465`：`leaves → Leaf`、`wolves → Wolf/Wolfe` 直接 1.2 分过阈，无原文/语义校验。派生变体必须“候选表面形出现在原文”或“唯一映射”才接受。
- **R-07 containment 无词边界（P2）** `search.py:267-268`、`vector_store.py:1224`：`war → Warning: Low Health` +0.06，`iron → Irony`。改为 token 序列包含 / 单 token 加边界或 trigram+边界。
- **R-08 单 token 消歧“唯一候选或放弃”（P1）** `search.py:473-531`：词表有 `Whiterun Guard`/`Whiterun Stable` 时直接 None；短 token（≤3）直接拒绝。多候选按余弦/DF/长度取 top-1 降分，或全部交给融合排序。
- **R-09 低信号误杀（P1）** `_is_low_signal_query:210-250`：`family`、`reply` 因 `-ly` 被判低信号；`my sword` 因所有格首词被判低信号；且低信号时连纯词法 containment 也一起跳过。删 `-ly` 一刀切、所有格剥离后查中心名词、低信号仍跑词法。
- **R-10 1-2 字 CJK 无法 containment（P2）** `vector_store.py:1209-1218`：<3 字符只查 token 精确命中，200k 中文查“护甲”返回 0；建 2-gram/单字倒排。
- **R-11 归一化三份实现不一致（P2）** `glossary_manager.py:106-117`（NFKC+去标点）vs `vector_store.py:529-535`（无 NFKC）vs `keyword_extractor.py:446-452`；`Sjórs → sj rs` 后 `rs` 又被 <3 长度过滤，与 `Sjors` 不等价。统一 `normalize_key()`。
- **R-12 大小写/标点重复词条各占向量（P2）** `engine.py:259-260` 用原始串判断“已索引”，`Sword/sword`、`Sword (Iron)`/`Sword Iron` 重复入库；建索引前按归一化 key 去重（保留 priority 最高者）。

### 3.2 排序/融合

- **R-13 三种分数尺度混比（P1）** `search.py:401/745/865`：直配 1.2、语义 cosine 0.75 阈值、weighted 路径 `min_vector_score=0.45`；同一配置在两条路径含义不同，UI 还会显示 >1 的相似度。召回用低门槛原始 cosine，排序用融合分，阈值只作用于归一化后的最终分。
- **R-14 `min(1.0)` 饱和 + 并列反序（P1）** `search.py:400,806-809`：base+boost 常超 1.0，多个 1.0 并列时 `argsort[::-1]` 让最后入库的术语优先；重建索引后结果漂移。不 clamp；显式 tie-break `(-score, len(term), term)`。
- **R-15 每 query 独立桶上限、无全局重排（P1）** `_apply_bucket_limits:102-130`：单 query 超 7 条按本 query 排序截断，跨 query 不重排；被丢术语只进 debug。先收集 `(term, fused)` 全局排序，再按 short/long 配额截断。
- **R-16 桶选顺序违例（P1）** `search.py:117-128`：short 满后仍接纳更低的 long，最终术语集不单调。桶只作配额，先全局 top-(short+long) 再分流。
- **R-17 上下文截断按查询顺序而非分数（P1）** `prompt_builder.py:402-409`：`glossary_context_max_chars` 超限时从尾部 pop，先到先得；第 1 个关键词命中的弱相关词保住、第 10 个命中被截。`search()` 返回有序 `[(term, score)]`，按分截断。
- **R-18 anchor token 反选 df=0（P2）** `search.py:306-311`：升序选“最稀有”，df=0 的词根本不在任何术语里 → overlap 恒 0；只在 df>0 中按 df 降序选，分母用实际命中数。
- **R-19 debug 记账常开 + 锁内 deepcopy（P1 性能）** `translator.py:300,331,330-337`：`return_debug=True` 恒定，`record_candidate_decision` 不受开关控制，`_set_last_rag_debug_info` 在全局 Lock 内 deepcopy 数千 dict；`get_last_rag_debug_info` 全仓无调用者。debug 由调用方开关，删除死缓存。
- **R-20 stop/pause 不进入检索（P1）** `engine.py:88-89,342-345`：`search.py:623` 循环无检查；128 个 query 的全量余弦 + embedding 仍会跑完。`RAGSearcher` 持有 stop_event，query/embedding 批次边界检查。

### 3.3 索引完整性（除 P0-06/07/08 外）

- **R-21 `glossary_hash` 只写不读（P1）** `vector_store.py:239` 写入，`_evaluate_index_status:332-468` 不比较；外部改词表/`add_term` 向量化失败后索引陈腐永远检测不到，界面仍显示 ready。状态机比对 key 集合哈希（只哈希 key 集合，译文更新不触发重建）。
- **R-22 增量构建只增不删（P1）** `build_index:1019-1020`：`terms_to_process` 只含新词，删词后孤儿向量永久保留，占 top-24 名额、拉低召回。差集删除 + 对账。
- **R-23 checkpoint 全量重写（P2 性能）** `vector_store.py:1132-1147`：每次 checkpoint `vstack(旧,新)+np.save+json.dump`；默认每 1000 条，200k 时 200 次、累计写 ≈123GB、峰值 2.4GB。预分配容量/分片追加。
- **R-24 `add_vectors_batch` 无 checkpoint + 峰值 4.8GB（P2）** `868-942`：全部批次先驻留，再两次 vstack；中途失败丢全部 API 花费。复用 checkpoint，预分配 buffer。
- **R-25 删除逐条全量重写（P1）** `engine.py:237-248` 每条 delete 调 `delete_term`+`delete_vector`（O(n) 删、全量词法重建、全量 npy）；20k 词库 200 条 33.4s。改用已有 `delete_terms_batch` 一次完成（1000 条 0.09s）。
- **R-26 terms 过滤导致行号错位（P1）** `vector_store.py:479` 过滤空串/非字符串但不调整 npy；`build_index` 的 `keep=min(v,t)` 只截长度、不修映射。加载时任何过滤即标 corrupt，删除 min() 截断分支。
- **R-27 metadata/normalization 校验不足（P2）** `_evaluate_index_status:402` 的 `not stored_fingerprint` 恒 False（normalize 总是返回 3 键），旧 meta 无 `embedding` 键时会误判 ready；load 直接相信 `unit_l2=true`，混合归一化会走点积分支。抽样 64 行校验范数；以 `embedding` 键存在性判 metadata_missing。
- **R-28 Windows 写可靠性（P2）** `os.replace` 未捕获 WinError 5/32（杀软/OneDrive），异常冒泡中断构建且内存 terms 已扩展；索引无 `.bak`。replace 重试 + `.bak` 回退。
- **R-29 并发写（P2）** `_write_metadata_atomic` tmp 名仅 pid；`_persist_detected_normalization_state` 在检索线程读改写 metadata；`_append_terms_to_lexical_index` 原地 mutation 与 `search_containment` 迭代冲突（`RuntimeError: Set changed size` 被 try 吞）。store 级 RLock/不可变快照。

### 3.4 元数据与术语一致性

- **R-30 rich_meta 全链路死代码（P1）** `glossary_manager.py:85-88` 注释自认 store-only；`search.py`/`prompt_builder.py` 完全不读 priority/forbidden/variants/domain/examples；多译法导入只取 `value[0]`（`glossary_import.py:559-560`）。建议：priority 参与排序；forbidden 进提示词并做输出拦截；variants 进 `_glossary_lookup` 别名；domain 按上下文过滤。
- **R-31 术语一致性无强制（P1）** 仅检查“英文原词仍残留”（`quality_checker.py:432-439`，warning），从不检查“命中的术语译文是否出现”，也不用 forbidden/variants。加一致性检查（advisory→可重试 error）+ forbidden 拦截。
- **R-32 词表注入（P1 安全）** `prompt_builder.py:349-362` 直接拼 `- term -> value`，只有源文走 `_escape_source_delimiters`；术语含换行/`<<<END_GLOSSARY_DATA>>>` 可提前闭合数据块。导入时净化控制字符并拒绝分隔符；渲染时转义。
- **R-33 词表指纹不含 sidecar（P2）** `_content_fingerprint` 只哈希 glossary，改 rich meta 不失效；缓存键同理。

### 3.5 成本/缓存（RAG 入口）

- **R-34 每条唯一文本 1 次关键词 LLM（P1 成本）** `translator.py:487,808` 无条件调用；`OK/Health/MCM/Yes/Whiterun` 各一次；glossary 为空/索引未建时纯浪费。短文本（≤40 字符/≤3 token）走本地路径（原文自身当 query + 本地词表精确/包含匹配），空索引前置短路。实测 prompt 378 字符（system 286+user 92）；按 1000-3000 条估算可省 50-70% 抽取调用、约 55% prompt token。
- **R-35 LLM 合法返回 `[]` 被正则兜底覆盖（P1）** `extract:126-147`：`"Arrive at Whiterun by nightfall." + []` → `['Arrive','Whiterun','nightfall']`；应区分“解析成功但为空”与“调用/解析失败”。
- **R-36 缓存语义（P1）** `extract:157-159` 失败/降级结果也缓存且 TTL=0；指纹只含 prompt/model/glossary，不含 `keyword_task_keep_original/keyword_max_queries/decompose`（改配置无效）；key 用原始文本（3 种空白变体 3 次调用）；直接返回内部 list（调用方 append 后污染）。raw 输出入缓存、finalize 每次重算；失败不缓存；key 归一化；返回副本。
- **R-37 关键词解析覆盖不足（P2）** `_parse_keyword_response:714-782`：纯行 `Erik\nWhiterun` → `[]`；`Keywords (see [1]):` → `[1]`；英文拒绝语不识别（备用模型永不触发）。校验 list[str]、补行解析、补英文拒绝标记。
- **R-38 请求参数（P2）** 关键词走 `llm_search` 永远不加 `response_format`（`client.py:429-433` 只认 `llm+translate`）；`max_tokens` 被 pop；temperature 默认 null。给 `keyword_extract*` 开 JSON mode、temperature=0、小 `max_tokens≈256`。
- **R-39 缓存持久化缺失（P2）** `EmbeddingCache` 未传 persist_path（`engine.py:76`），重启重付全部 embedding；关键词缓存同样只在内存。npz/sqlite 持久化 + fingerprint 命名空间。
- **R-40 失败熔断（P1 稳定性）** 搜索模型限流/宕机时每条最坏 2×300s（`retry_total_timeout` 每次 _call 独立）；`extract` 拿不到 stop/pause。连续 N 次失败后 60s 本地兜底熔断，调用前检查 stop。

### 3.6 RAG 改进优先级

1. **整篇确定性命中扫描**：构建期把 glossary 归一化后建 Aho-Corasick（或首 token 倒排），每条源文 O(len) 扫一遍，命中即直配（1.0/1.2）并标 `match=exact`。预期逐字出现（含多词/所有格/标点）的术语召回趋近 100%，不依赖 LLM 关键词；30k 词约数 MB 内存。
2. **保留短语 + 批量 GEMM**：`keep_original=True`；`Q(K×d) @ V.T` 按 10-20k 行分块一次算完（实测 100k 5 keyword 65.2→33.4ms，1.95×）。
3. **统一融合分 + 全局重排**：`fused = cosine + w·exact + w·contains + w·overlap·idf`，不 clamp；batch 内 min-max 或分位数；`search()` 返回 `[(term, score, match_type)]`，上下文截断按分。
4. **索引健康**：`glossary_hash` + `complete/expected` 入状态机；重建写临时索引再原子替换；失败词表回传并增量补齐。
5. **rich metadata 真正接入**：priority 排序、forbidden 提示+拦截、variants 别名、domain 过滤。
6. **成本**：短文本本地化、关键词结果/embedding 持久化、失败熔断、debug 关闭。
7. **词法**：统一 normalize、2-gram CJK 倒排、IDF 降权通用词、df>0 anchor、词边界 containment。

---

## 4. 翻译质量

### 4.1 现状评估（实测提示词）

- 核心 system prompt 约 950 token（10 条），对话 +4 条、书籍 +8 条；实测 a “Health” 请求 system 949 + user 196 = 1145 字符，是源文的 1000 倍；书籍 1719 字符请求 1704 字符提示（4.0×），长文 5754 字符请求 6061 字符。
- 优点：语义骨架/作用域/否定比较/反翻译腔/格式壳/分类重试模板齐全；原文用 `<<<SOURCE_TEXT>>>` 隔离并中和闭合串；JSON schema 与解析器一致。
- 天花板（产品级）：①无完整性度量；②术语一致性无强制；③长文分块无质检且段落哨兵会丢；④UI/对话缺长度、称呼、上下文约束。

### 4.2 完整性（最影响书籍/任务文本）

- **Q-01 无覆盖率检查（P1，建议按 P0 处理）** `quality_checker.py:110-191`：实测 `check(120 词英文, "古老的诺德遗迹。") == []`；`check(3280 字英文, 3120 字中文+328 字未译英文段) == []`。译文缺段直接写盘并缓存。加 `coverage` 规则（源 ≥200 可见字符且 `len(tgt)/len(src) < 0.25` → error，加入 `_BLOCKING_UNTRANSLATED_RULES`）。
- **Q-02 `finish_reason` 被丢弃（P1）** `client.py:660-667`：200 + `finish_reason=length/content_filter`、`content=None` 都当成功；Responses 的 `incomplete_details` 同样不查。抛 `ResponseTruncatedError`，触发拆块重译而不是原样重试。
- **Q-03 containment 提前 return 跳过 latin_ratio（P0 质量）** `quality_checker.py:244-262`：整句原文 + 任意一个汉字/字母 → `warning/containment` 后 `return`，不再执行拉丁比例检查；`"The quick brown fox … river bank. 狐狸"` 放行且可缓存。降级条件改为“整段命中且 CJK 占比 ≥ 阈值”，或降级后继续 latin_ratio。
- **Q-04 纯标记/纯标点译文通过（P1）** `check("<font>Hello</font>", "<font></font>") == []`、`"Hello there" → "。。。" == []`。`visible` 为空或仅标点 → `empty_visible` error；零宽字符（Cf）告警。

### 4.3 术语一致性

- **Q-05 三重弱化（P1）** `default.json:10`“仅在语义匹配时采用” + `prompt_builder.py:376`“仅在…时采用” + `templates.json:3`“仅是候选约束…忽略”。合并为一条强约束：术语是指称一致时的默认译名，仅原文证据明确冲突时才忽略。
- **Q-06 残留仅 warning（P1）** `_check_untranslated_fragments`：英文原词仍在译文只记 warning，不重试、不阻止缓存；缓存命中时 `matched_terms=None` 直接跳过。升级为 error（排除全大写缩写/UI token/引号包裹），并纳入 `_BLOCKING_UNTRANSLATED_RULES`。
- **Q-07 同文译名不校验（P1）** 不检查期望译文是否出现，也不用 forbidden/variants；长文本按整篇计算，跨块不一致无法发现。
- **Q-08 术语分布噪声（P1）** 实测 “Health” 6 字符标签带 4 条无关词（196 token）；长文 4 个块每块都列全部 5 条，chunk4 原文一个词都不含仍标“在原文中出现”。逐块过滤 `_term_appears_in_source` + priority 排序/截断；reference 段在有 in-source 命中时可省略。
- **Q-09 术语注入（P1）** 见 R-32。
- **Q-10 术语值本身是英文时检查跳过（P2）** `quality_checker.py:424-426`：`Health -> Health` 时禁用残留检测；导入时应拒绝“译文=原文”（keep-original 标记除外）。

### 4.4 重试

- **Q-11 不可重试错误仍重试（P1）** `translator.py:658` 只放行 `CONTENT_BLOCK`；`AUTH_ERROR/INVALID_REQUEST` 也会被外层重试，实测 401 打 5 次 HTTP/5 份 traceback；8 线程即 40 次。catch 里按 `classify_error` 直通。
- **Q-12 隐藏 +2 预算（P1）** `max_retry_limit = base + format_extra` 无条件相加；实测 `max_retries=0 → 3 次`、`2 → 5 次`；长文本路径又只用 base（每块 3 次）。只在格式错误时加 extra，两条路径统一。
- **Q-13 长文块重试盲发（P0 质量）** `1124-1127,1129-1140,1191-1206`：`messages` 只建一次，重试原样再调，无“缺什么”的反馈；块级只查空/格式 token，不跑 untranslated/术语；整篇最后按全文字符比例算 latin_ratio，328 字未译英文被 3120 字中文稀释。逐块 QC + 证据化重试（见 4.9 E）。
- **Q-14 角色交替被破坏（P2）** 重试追加 `user` 消息到已以 user 结尾的 messages，实测 `['system','user','user']`；严格模板（Qwen/Llama 本地网关）会 400。把上轮译文作为 `assistant` 或并入同一 user 消息。
- **Q-15 重试模板优先级错误（P2）** `1579-1589` 格式优先，但最终判定 untranslated 优先；两类并存时重试在修不会导致失败的那类。按阻塞优先级选模板或合并。
- **Q-16 重试证据不可执行（P1）** identity/latin_ratio 无具体 span；`previous[:1976]+"…[truncated]"` 而源文可达 4000 字；format 证据来自已 restore 的真实标记（`expected '[PlayerName]'`），模型输入里却是 `__FMT_1_0001__`。把残留拉丁 span 放进 `fragments`，`error_details` 上限 500→1500，格式证据按哨兵序号说明。
- **Q-17 自己抛的 RuntimeError 被 except 捕回（P2）** `620-634` 的拒绝 raise 在 try 内，落到 `658` 又写一份 ERROR、把 attempt 标 failed；移到 try 外或前置 `except RuntimeError: raise`。

### 4.5 解析器

- **Q-18 取第一个 translation（P1）** `_try_parse_first_json_object:63-66`、`_TRANSLATION_KV_RE:355`：草稿对象覆盖最终答案；实测 `草稿:{…}\n最终答案:{…} → 草稿译文`。优先取最后一个合法对象，或跳过 `这里填写译文`/`...` 模板值。
- **Q-19 提示词回显劫持（P1）** system/retry 模板内嵌 `{"translation":"这里填写译文"}`，模型复述格式说明时被当译文；QC 全中文无 issue 并缓存。跳过模板占位值 + prompt-leak 检测。
- **Q-20 JSON 标量被当译文（P1）** `"null"/"true"/"false"/"[]"` 原样返回；QC 因 alpha≤5 放行并缓存。fallback 前 `json.loads`，标量/空数组返回空；QC 增加“不是合法译文”规则。
- **Q-21 纯文本兜底接受解说词（P1）** `"这句话是问候语。译文：你好，旅行者。"` 原样返回；`_META_TRANSLATION_PREFIX_RE` 只锚定开头。检测任意位置的“译文/翻译/原文/意思是”，截取标记后或判失败。
- **Q-22 尾逗号正则破坏内容（P2）** `re.sub(r",\s*}", ...)` 全局替换：`"使用 ,} 结束"` → `"使用 } 结束"`。只在 JSON 末尾做修复。
- **Q-23 bare KV 转义未还原（P2）** `translation: "\"你好\""` 返回带字面反斜杠。
- **Q-24 JSON 形状的 UI 文本无法回译（P2）** `{"a":1}` 抛 broken-fragment，`{"a": 1}` 返回空；源文本身是 JSON 时应原样接受。
- **Q-25 属性含 `>` 的标签未保护（P2）** `<a title="1 > 0">` 开标签未进哨兵，丢失也能过 QC；改用引号感知的标签正则。
- **Q-26 批量 ID 全有或全无（P1）** `parse_batch` 要求 `set(parsed)==expected`，8 项缺 1 项 → 整批回退 8 次单条；实测一次调用变 9 次。只回退缺失 id。
- **Q-27 批量无 id 时按位置映射（P1）** `raw_id = item.get("id", pos)`：模型不带 id 且顺序打乱时，`{0:耐力,1:护甲,2:生命值}` 通过校验 → 译文错配且 QC 通过。无 id 项较多时判该批不可信。
- **Q-28 批量失败 fail-open（P2）** `940-971` 失败时把原文当“译文”返回；`return_debug_info=False` 的 API 调用方无法区分。返回 `""` 或 `strict=True` 抛错。

### 4.6 质量检查误判（两个方向都错）

- **Q-29 单词 UI 标签被当专名（P0 质量）** `quality_checker.py:229-235,338-365`：`Armor/Health/Lockpicking/Damage/Sneak` 全部判专名 → warning 接受，且单英文词被缓存（`_is_suspicious_identity_translation` 只在 ≥2 英文词时命中）。实测 `Health→Health` 第二次 0 次 LLM 调用。zh 目标下“源含拉丁、译文完全无 CJK”应 error；identity warning 一律不缓存；扩充 UI/属性词表。
- **Q-30 缩写/合法不译被判错误（P1）** `SKSE/MCM/FNIS/RaceMenu/OK/HP/FPS` 原样返回 → identity error → 重试 3-5 次后整行失败；`is_common_ui_token("OK")` 为 True 也仍然报错。全大写/短 token/常见 UI 词 → warning/pass。
- **Q-31 大写实词被剥出拉丁比例统计（P1）** `"Restore 20 points of Health." → "恢复 20 点 Health。"` 与 `"她是 Thalmor 特工。"` 均 `[]`；`_strip_safe_latin_spans` 把 Title-case 词当专名。增加 `_COMMON_STAT_WORDS`（health/magicka/stamina/damage/armor/weight/value/speed/fire/frost/shock/poison/disease/…）。
- **Q-32 双语重复被接受（P1）** `"Open the door" → "打开门 Open the door"` 仅 warning，should_retry=False 并缓存。只有 glossary 命中词/标识符被包含时才降级，整句保持 error。
- **Q-33 短标签+英文专名误报（P2）** `"SkyUI" → "SkyUI 界面"` 报 alpha=5/cjk=2；源文中逐字出现过的拉丁片段不计入 alpha，或 len<20 放宽阈值。
- **Q-34 合法源文本触发拒绝（P1）** `is_model_refusal("我无法处理这个请求。", "I cannot process this request.") == True` —— 正常台词的忠实译文被当模型拒绝，抛 `ModelRefusalError` 并在重试后失败；同类“我无法提供该内容”“这个请求超出了我的能力范围”。拒绝规则要求任务元词/“作为 AI”上下文，或加游戏对话白名单。
- **Q-35 中文源译中文被 identity 误杀（P2）** `"你必须先找到龙石"→同文` 报 identity；目标 zh* 且源 CJK 占比 ≥0.8 时直接 passthrough。

### 4.7 长文本与空白

- **Q-36 段落换行静默丢失（P0 质量）** `translator.py:1171-1174,1208,1213-1220`：`\n\n` 被哨兵化，模型不回抄则恢复为空；终检用 `RELAXED_ALL` 忽略空白，`status=success`；同样内容在 <4000 字符单条路径会报 format error，两条路径语义不一致。不要哨兵化 `\n/\n\n`（或用 `<PARA>` 并说明必须保留），终检保留空白 token。
- **Q-37 CJK 分块硬切句（P2）** `text_analyzer.py:293-303`：前两级边界要求标点后 `\s+`，中文句末无空格 → 3400 字按 300 切 14 块全部句中。CJK 分支去掉 `\s+` 要求；块头加“第 i/n 段，仅译本段”。
- **Q-38 首尾空白丢失（P2）** `_finalize_translation_text` 不恢复源首尾空白；`_should_protect_whitespace` 在 `start==0/end==len` 时直接 False。`" Lockpicking " → "开锁"`，严格策略也查不出。按源串恢复前后空白。
- **Q-39 尾部换行被冻结（P2）** `"Hello\n" → "你好"` 报 protected_token_sequence error；行尾换行应可容忍（先判边界再判换行）。
- **Q-40 长文 RAG/术语不按块（P1）** 整篇 100k 源文做关键词抽取、每块重复全量术语 + 整篇 `_term_appears_in_source` 扫描；关键词用前 8k 或逐块合并，块内只传本块命中的术语。
- **Q-41 块失败丢全部已译块（P2）** 失败只写 debug `partial_translation` 后 raise，已译块不缓存、不返回；按块缓存/部分返回。

### 4.8 非中文目标 / 风格

- **Q-42 提示词硬编码“中文”（P1）** `default.json:6`“中文表达：按中文信息顺序重组…”；`style_profiles.json:5`“《上古卷轴》式奇幻中文”。target 可选 ja/ko/fr/de/es/ru，指令与 `{target_language}` 冲突。改为 `{target_language}表达`；法语/德语文风后续按语言拆文件。
- **Q-43 非 CJK 目标无质量门禁（P1）** `_should_check_latin_ratio` 只对 zh/ja/ko 生效，en/fr/de/es/ru 只做 identity/containment；选法语/德语等于无质检。按目标语言脚本（西里尔/拉丁/CJK）判定。
- **Q-44 武器/护甲/法术 DESC 被当物品名（P1）** `style_profiles.json:96-100` `WEAP→item_name` 等；实测 `{'record_type':'WEAP','field_type':'DESC','text_kind':'document'} → item_name`，长描述拿到“保持短而稳定”规则。record_field 补 `*:DESC → document` 或把 text_kind 判定提前。
- **Q-45 `journal_letter` 不可达（P2）** 无映射指向；BOOK+EDID 含 Journal/Letter/Note 时映射或加 text_kind。
- **Q-46 对话缺角色上下文（P1 质量）** 当前只有“保留身份/阶层/亲疏”规则，没有 speaker/listener/上一句/语气数据；敬称、自称、代词一致只能靠猜。GUI 可从 ESP INFO/相邻行取，无数据时至少传上一行原文。
- **Q-47 MCM 重复标题 + 消息误判对话（P2）** `prompt_builder.py:268,293` 标题与首条重复；82 字符 `<font>Warning:</font>…` 被注入“普通对话空白上下文”。去重标题；仅 text_kind=dialogue/有引号时注入。
- **Q-48 无 UI 长度/中文排版硬约束（P1）** system 全文无“字符/长度/全角/简体”；MCM 选项可译成超长句；无 `，。！？`、CJK-Latin 间距、ASCII 引号规则。补“短标签≤8 字/选项≤10 字/说明≤原文×1.4”+ 中文排版四条 + 反翻译腔一条。
- **Q-49 源文/术语含 `{text}` 直接终止翻译（P1）** `prompt_builder._check_leftover_vars` 命中即 `ValueError`，`build()` 在重试循环外，异常冒到 worker；`build("Press {text} to continue.")` 实测抛错。只检查模板本身，或先替换为唯一占位再还原。
- **Q-50 style 兜底可能取到 nsfw（P2）** `prompt_builder.py:512-520` `next(iter(...))`；default.json 损坏时可能把成人指令套到所有译文。只认 `default`，缺失用内置 fallback。

### 4.9 具体改动（可直接落地）

**A. 覆盖率门禁 + finish_reason**（Q-01/02/04）
```python
# quality_checker._check_untranslated 末尾，latin_ratio 之前
if len(source_visible) >= 200:
    ratio = len(translation_visible) / max(1, len(source_visible))
    if ratio < 0.25:
        return QualityIssue(QualityIssueType.UNTRANSLATED, "error",
            f"coverage {ratio:.2f} < 0.25 (src={len(source_visible)}, tgt={len(translation_visible)})",
            rule_id="coverage")
```
`_BLOCKING_UNTRANSLATED_RULES` 加 `coverage`；`client._call` 在 `finish_reason in {"length","content_filter"}` 抛 `ResponseTruncatedError`。

**B. 重试差异修复**（Q-16）
- `untranslated` 模板加 `{error_fragments}`；`generic` 加“以下问题必须逐条消除：{error_details}”，上限 500→1500。
- `format_error` 用哨兵序号描述“缺第几个受保护标记”，`last_translation` 上限 2000→4000。
- `_check_untranslated` 失败时把 `_LATIN_SPAN_RE.findall(text_only)[:8]` 放进 `fragments`。

**C. 术语默认采用**（Q-05/06）
`templates.json:3` 改为“术语条目是指称一致时的默认译名，应全篇采用以保证一致；仅当原文证据明确冲突（同词异指、专名被当普通词、性/数变形）时才忽略，且忽略后仍须保持同一选择。”删除 default.json 与 prompt_builder 的重复 hedge；残留检测升级为 error。

**D. 哨兵类型化**（Q-36 相关）
`__FMT_T*__`（XML/运行时标记）、`__FMT_P*__`（占位符）、`__FMT_N*__`（换行/制表），prompt 加图例；同时不要把 `\n` 哨兵化。

**E. 长文逐块质检 + 反馈重试**（Q-13）
每块 restore 后 `_quality_checker.check(chunk, chunk_translation, 块内术语, target_lang)`；有 error 时把 `missing_tokens`/残留片段作为新 user 轮追加，只重译本块；终检保留空白 token。块头加“第 i/n 段，仅译本段，不补开场/收束”，并附上一块原文末尾 200-300 字作为指代锚点。

**F. 语言无关核心**（Q-42/43）
`"{target_language}表达：按{target_language}信息顺序重组…"`；`_should_check_latin_ratio` 按目标语言脚本判定（ru 用西里尔计数）。

**G. 称呼/量词**（Q-46/48）
dialogue 增加：“称呼按说话人与受话人的阶层、亲疏、场合决定你/您与头衔；原文用敬语/头衔时中文须体现同等礼貌等级；原文无标记 you 时用‘你’，不得一律省略称谓。量词/代词按所指对象类别与性别选择，源文未明示时不擅自补出性别。”

**H. 精简提示词**（Q-48 相关）
系统提示约 1147 字符（dialogue 1323），固定成本约 1100-1300 token；删重复的术语 hedge、style 头部 `profile_id`、重复文体表述；补 1 个 JSON few-shot + “字符串内不要用 ASCII 双引号、换行写 `\n`”。预期省 15-25% 输入 token、降低解析失败率。

### 4.10 缓存相关（翻译质量）

- **Q-51 缓存指纹遗漏** `translator.py:1497-1508`：缺 `glossary_entry_max_chars`、`batch_prompt_max_chars`、`latin_ratio_threshold`、`long_text_chunk_*`；改配置后仍命中旧译文。并入 payload 或整体 hash。
- **Q-52 entry_id 过度隔离** `translator.py:1429-1431`：无空白短文本把 `entry_id` 放进 key，同一 “Health” 在不同条目既不 dedupe 也不命中缓存 → 重复 LLM+RAG。把 entry_id 从缓存键移除（保留 text_kind/entry_type/domain），或加共享短标签命名空间。
- **Q-53 坏译文被缓存** `_should_cache_translation:1674-1685` 只挡 error 与多词 identity；warning 级（专名 identity、containment、latin 比例）仍缓存。identity/containment/coverage 规则一律不缓存。
- **Q-54 缓存命中质检跑两遍（P2）** `444-450` + `463-470`、batch/长文同理；让 `_should_reject_cached_translation` 返回 `(rejected, issues)` 复用。
- **Q-55 matched_terms 口径不一致（P2）** 单条/batch 传原始集合，长文传过滤集合，缓存命中传 None；统一在 `_run_rag_phase` 返回前过滤。
- **Q-56 缓存只在整轮结束落盘** 崩溃/停止丢全部新译文；每 200 条或 30s 脏条目节流保存（或 jsonl 日记）。

---

## 5. 稳定性与数据安全（非 P0 部分）

### 5.1 XML / ESP

- **X-01 全量重写破坏字节保真（P1）** `xml_processor.py:95-137` / `esp_xml_processor.py:72-114`：`tree.write(..., encoding="utf-8", xml_declaration=True)` 即使未改任何 Dest 也会改写声明（standalone 丢失、引号变单引号）、丢 BOM、CRLF→LF、丢末尾换行、`&#160;`→U+00A0、`&quot;`→`"`、`<Dest></Dest>`→`<Dest/>`、单引号属性→双引号、cp1252/UTF-16→UTF-8。游戏多数能接受，但会破坏与其它工具的 diff、编码敏感文件。修复：保留原声明/BOM/EOL，采用“只替换被修改 Dest 的文本区间”的补丁式写回。
- **X-02 `.bak` 只有一代且失败也留备份（P1）** `xml_processor.py:118-122`：第二次保存后 `.bak` 是第一次保存后的内容，原始不可恢复；只读目标失败也产生 `.bak`。仅当 `.bak` 不存在时创建或时间戳多份；备份移到 replace 成功之后。
- **X-03 另存为不切换写回目标（P1）** `save_file(output_path)` 不更新 `self.file_path`；另存后再 Ctrl+S 写回原文件，副本保持旧内容。成功后更新路径并同步 GUI。
- **X-04 译文片段扫描 O(n²)（P1，可冻结 GUI）** `xml_content.py:212-271`：`"<a>"*16000`（48KB）单次 `update_dest` 70.7s（长度×2 耗时×4）；`"<b>x</b><i>y"*4000` 12.8s。失败时 cursor 跳到已扫最远位置，或对超长内容直接按文本写回。
- **X-05 CDATA Dest 的标签译文变字面文本（P2）** `xml_content.py:65-72`：CDATA Dest + `<font>` 译文 → 标签成字面字符，普通 Dest 则生成元素；行为不一致且无提示。
- **X-06 新建 Dest 位置/形式不统一（P2）** 追加在末尾、无缩进、`<Dest></Dest>`；空译文也建节点。按兄弟缩进插入；空值不建。
- **X-07 `forbid_dtd` 分支死代码（P2）** 当前 lxml 不支持 `forbid_dtd=`，每次都走 `except TypeError`；实际防护靠 512KB 预检 + `docinfo.doctype`。删除死分支，注释写真实策略。
- **X-08 预检误拒合法 PI（P2）** `<?pi <!DOCTYPE foo?>` 被拒；预检先剥离 `<?...?>`。
- **X-09 512KB 预检窗可被绕过（P2 安全）** 600KB 注释后的 DOCTYPE 预检漏过，由 lxml 先解析、再被 `docinfo` 事后拒绝；因 `resolve_entities=False` 实测未展开，但“解析前拒绝”的设计被绕过。分块扫到文件尾。
- **X-10 lxml 缺失提示被吞（P2）** `load_file` 只返回 False，GUI 显示“load_file returned False”；把异常文本交给 `load_failed` 回调。
- **X-11 reload 后旧节点静默写废树（P2 API）** 同实例二次 load 后对旧 node `update_dest` 无异常，新树保存不含译文；校验 `node.getroottree()` 与当前 tree。
- **X-12 50MB 上限可能拒合法大文件（P3）** `safe_xml._MAX_XML_BYTES` 对 10 万条以上的大型 mod 可能偏小，建议可配置。

### 5.2 MCM

- 除 P0-05 外：
- **M-01 编码回退只有 utf-8→gbk（P1）** cp1252 文件被按 GBK 解码：实测 `'It抯 caf dash'`，再按 GBK 写回或触发清空。先试 cp1252，记录读入编码并原样写回；含 U+FFFD 拒绝保存。
- **M-02 UTF-16 BOM 未剥离污染首个 key（P1）** `\ufeff$OptionA`；`decode("utf-16")` 并记录 BOM，写回时先写 BOM。
- **M-03 未知语言后缀导致游戏不读（P1）** `MyMod_italian_CHINESE.txt`；`_KNOWN_LANG_SUFFIXES` 只有 8 项，缺 ITALIAN/POLISH/PORTUGUESE/CZECH/TURKISH/UKRAINIAN 等；剥离尾部 `_<2-12 位字母>` 并补全。
- **M-04 非法后缀回退为覆盖源文件（P2）** 配置 `"Chinese (Simplified)"` 不匹配时 `return self.file_path`，GUI 判 same_target 后静默覆盖 `MyMod_english.txt`；应返回 `""` 让上层报错。
- **M-05 `splitlines()` 按 U+2028 等分行（P2）** `$A\thello\u2028world` 保存成两行 + 无 tab 行；用 `re.split(r"\r\n|\r|\n", text)`。
- **M-06 纯 CR/混合换行被改写（P2）** 纯 CR 文件变 LF；用 `splitlines(keepends=True)` 保留逐行结尾。
- **M-07 行首缩进丢失（P2）** `  $OptionD\t...` 写回为 `$OptionD...`；MCMEntry 保存前缀。
- **M-08 清空译文不落盘（P2）** dest 为空回退 src，表格显示空/UNTRANSLATED 与导出矛盾；用 `cleared` 标志区分“空=用默认”。
- **M-09 `update_dest` 默认参数恒 no-op（P2）** 载入时 dest=value，未传 overwrite 时任何非空行不更新；默认值应改为 overwrite=True 或改名。
- **M-10 `file_formats` 内容识别是死代码（P2）** `detect_translation_file_type/classify_translation_file` 全仓 0 调用；GUI 重复实现且只认大写 `ESP`；`.txt` 无条件判 MCM。收敛到一处。

### 5.3 词库（除 P0-09/10 外）

- **G-01 BOM 触发整库隔离（P1）** `glossary_manager.py:358,296` 用 `utf-8` 读，导入侧却是 `utf-8-sig`；记事本另存 UTF-8 with BOM 即 `JSONDecodeError` → 文件被改名 `.corrupt.*`、空库运行、`_glossary_corrupt` 无人读取。改 `utf-8-sig`，失败先多编码重读再隔离，并把损坏状态弹窗。
- **G-02 op=delete 被归一化冲突跳过（P1）** `engine.py:233-241` + 两套 normalize：已有 `Foo Bar→旧译`，导入 `Foo-Bar→新译` + `Foo Bar→delete` → 删除被跳过、旧译生效、新译在归一化检索中不可达。统一 normalize；冲突时删除旧词面写入新词面。
- **G-03 op=delete 逐条全量重写（P1）** 20k 词库 200 条删除 33.4s；复用 `delete_terms_batch/delete_vectors_batch`。
- **G-04 两列表头被当术语（P2）** `Term,Translation`/`原文,译文` 入库为 `{'Term':'Translation'}`；首行两列同时命中词面/译文 key 即视为表头。
- **G-05 rich metadata 死代码/字段名不一致（P2）** CSV 存 `meta_source`，JSON 存 `source`；`get_rich_meta/set_rich_meta_batch/prune_rich_meta/last_backup_path` 无调用；priority/confidence 无校验。
- **G-06 sidecar 覆盖 envelope 元数据（P2）** `load()` 先解析 envelope 的 rich_meta，再被 `_load_sidecar()` 无条件清空；无 sidecar 时 JSONv2 的 domain/priority 丢失。
- **G-07 无 BOM 的 UTF-16/NUL 静默入库（P2）** `'Sword,剑'.encode('utf-16-le')` → 词面带 NUL；读入检测 NUL/UTF-16 BOM 并重解。
- **G-08 限额只覆盖词面/译文（P2）** rich 字段可存 5000 字符 note；`max_rows` 在全文解析后才检查；>131072 字节字段抛裸 `_csv.Error`，worker 只 catch `GlossaryImportError` 会无 `finished` 死掉。流式解析 + `field_size_limit` + 异常包装。
- **G-09 未知 op 当 upsert（P2）** `op=skip` 也覆盖；op 白名单。
- **G-10 备份不可用（P2）** 每次写生成 `.bak.<ts>`，只留 5 份，GUI 无恢复入口；提供“从备份恢复”并写日志。
- **G-11 内存峰值（P2）** 100k 行导入：全文 + rows + 向量批次 + 两次 vstack，峰值 1.8GB+；`max_rows` 无法限制内存。流式/预分配。
- **G-12 GUI 删除按 `" -> "` 切词（P2）** 词面含 `" -> "` 时删错；用 UserRole 存原词，按返回值计数。
- **G-13 导入预览/worker 解析两遍**（见 F-09）。

### 5.4 配置 / 缓存 / 日志 / i18n / 打包

- **C-01 可写路径锚在 CWD（P1）** `main.py:184`、`manager.py:47-49`、`logging_helper.py:82-83`、`translator.py:57-66`；快捷方式起始目录不同 → 空配置/另一套 cache/glossary/logs。显式传入配置目录（frozen 用 exe 目录）。
- **C-02 配置保存失败静默（P1）** `manager.save_config` 吞异常无返回值；GUI 无条件弹“已保存”，重启回退。返回 bool + 错误详情；清理 tmp。
- **C-03 `cache_ttl_hours` 浮点被吞成 0（P2）** `manager.get` 按 default 类型校验：默认 `0`（int），配置 `12.5`（float）→ 返回 0，TTL 静默关闭。默认改 `0.0`，按字段类型 coerce。
- **C-04 中文 Windows 自动语言检测失败（P1）** `i18n._detect_system_lang` 用 `locale.getlocale()[0]`，中文 Windows 返回 `'Chinese (Simplified)_China'` → `startswith("zh")` 失败 → 首次启动全英文。先 `getdefaultlocale()`/`GetUserDefaultUILanguage`，并认 “chinese”。
- **C-05 “清除选中译文”清空全库缓存（P1）** `_clear_translation_caches:6029-6033` 调 `translator.clear_translation_cache()` → `invalidate_all()`；清 3 行 → 5 万条缓存全灭、下次全量重翻并在 GUI 线程写 9MB。`TranslationCache` 维护 `source→keys` 侧索引并提供 `invalidate_source`。
- **C-06 缓存失效 API 死代码（P2）** `invalidate_by_style` 无调用，且 `removed==0` 时 `clear()`；切 style 后旧条目继续占额度。真实调用并去掉全清兜底。
- **C-07 可疑缓存命中不剔除（P2）** 拒绝命中只记 DEBUG，不 invalidate；下次继续命中。拒绝即删。
- **C-08 TTL 过期条目不清理/仍写盘（P2）** 只在 get/has 时删；save 不过滤；TTL=0.2s、3 条过期后 size 仍 3。put 时清扫、save 时过滤。
- **C-09 缓存只有条数上限（P2）** 5 万长句可达 73MB 内存/90MB JSON，50 万上限 0.7-2GB；加字节预算。
- **C-10 schema 与运行时键漂移（P2）** `embedding.max_retries/backoff_base/retry_total_timeout`、`rag.batch_prompt_max_chars/glossary_entry_max_chars`、`general.short_text_batch_circuit_timeout_seconds` 运行时读取但 schema 没有 → 写入报 Unknown key 且 `get_typed` 丢弃。
- **C-11 GUI 范围超 schema（P2）** `setRange(1,99)` vs 上限 64、0-200 vs 100、0-200000 vs 100000；GUI 可保存必然校验失败的值。
- **C-12 废弃键直接删除不迁移（P2）** 24 个 key 直接 pop，仅 1 个迁移；老用户调优参数静默回默认。
- **C-13 i18n 缺 10 个 key（P2）** `msg_file_too_large`、`msg_large_file_confirm`、`msg_drop_directory_rejected` 等；界面直接显示 key。补齐并把 `i18n.t(` 字面量扫描纳入 CI。
- **C-14 无单实例互斥（P2）** 两实例互相删 tmp/覆盖配置/缓存；加 QLocalServer/QLockFile，tmp 带 pid+thread。
- **C-15 打包路径（P2）** exe 旁 `locales/prompts` 副本永远不生效（frozen 固定读 `_MEIPASS`）；改为 exe 同级 → `_MEIPASS`。首次运行不生成 config.json，`config.example.json` 无人使用；首启复制 example。
- **C-16 `crash.log` 可能写进 onefile 临时目录（P2）** frozen 时 `__file__` 在 `_MEIPASS`，被删；frozen 跳过该候选，直接 `%LOCALAPPDATA%`。
- **C-17 启动/异常观测缺口（P2）** 顶层 import 早于 faulthandler/excepthook，缺 DLL/prompts 损坏时 windowed 双击无窗口无日志；全仓无 `qInstallMessageHandler`。包 try/except 先写临时 crash 文件 + Qt message handler。
- **C-18 Windows 配置文件权限（P3）** `_protect_config_file` 仅 posix；Windows 下含 API key 的 config.json 默认 ACL 可读。文档提示或 DPAPI 加密。

### 5.5 LLM 传输层

- **L-01 `APITimeoutError` 被归类为 connection（P1）** `retry.py:211-214` 先判 `APIConnectionError`（父类）→ TIMEOUT 策略死代码，超时按连接错误重试 5 次/退避 30s。超时判断前置；`httpx.TimeoutException/TimeoutError` 也映射。
- **L-02 关闭思考时参数回退被禁用（P1）** `client.py:586-589` 的 `thinking_enabled is not False` 条件导致“关闭思考 + 供应商拒绝参数”时只调 1 次就失败；去掉该条件（回退已 strip）。
- **L-03 400/403/404 全折叠为 invalid_request（P1）** `retry.py:199-208`；错误日志只打 `str(exc)`，丢 code/request_id；无权限 key、模型名错、上下文超长三种处置在 UI 上同一句话。新增 AUTH/MODEL_NOT_FOUND/CONTEXT_LENGTH，非重试分支用 `format_provider_error`。
- **L-04 质量层对永久错误仍重试（P1）** 与 Q-11 同源：单条最坏 5 轮 × 4 次请求、1500s；批量失败还有两级逐条回退放大。按 `classify_error` 直通。
- **L-05 退避 sleep 不响应停止（P1）** `retry.py:418-421` `time.sleep(delay)`，RATE_LIMIT 最长 60s；加 cancel_event，`cancel_event.wait(delay)`。
- **L-06 reload 关闭在途 client（P1）** `client.py:93-106` 换属性后 lock 外 close 旧实例，`_call` 重试仍持旧引用；翻译中保存设置 → 8 线程被打断 + 重试调已关闭对象。按 section 每次取最新 client（getter），reload 用引用计数/延迟 close。
- **L-07 embedding 返回不校验长度/顺序（P1）** `client.py:356-358` 直接 `[item.embedding ...]`，`search.py:983` zip；少返回静默截断、乱序错配。校验并排序。
- **L-08 Retry-After 只对 429（P2）** 503+Retry-After:7 被忽略；embedding 路径 `retry_total_timeout` 默认 0 → 可能睡 1 小时。429/5xx 统一读取并 clamp；embedding 加默认预算。
- **L-09 不检查 finish_reason（P1）** 见 Q-02。
- **L-10 成本表过旧（P2）** `cost_tracker.py:22-29` 仅 6 个老模型，deepseek-v4/qwen/claude/gemini 成本恒 0；`cached_input` 未定义，缓存命中不折价；`prompt_cache_hit_rate` 把 embedding token 算进分母；`_records` 无上限。补报价/缓存价、单列 embedding、环形缓冲。
- **L-11 base_url 不补 /v1（P2）** 漏写 `/v1` → 404，只显示 “Error code: 404”；空 path 时补 `/v1`，404 提示检查。
- **L-12 推理参数拒绝标记不全（P2）** Responses 把参数改名为 `reasoning`，现有标记不匹配，回退不触发；补 “reasoning/effort/unsupported parameter/unknown field”。
- **L-13 按错误类型设计的重试策略被 config 覆盖成统一值（P2）** RATE_LIMIT=5/2s 等全部退化为 3/0.5s，429 仅 4 次约 3.5s；config 作为上限，RATE_LIMIT 取 max。
- **L-14 抖动单边正偏（P2）** `random.random()*jitter*delay` 只增不减，8 线程相位仍对齐；改 full jitter `uniform(0.5,1.5)`。
- **L-15 预算与超时误配置（P2）** `retry_total_timeout < request_timeout` 无告警，第三次尝试只剩 ~45s；加载时告警，remaining < 0.5×proposed 直接抛预算异常。
- **L-16 客户端构造异常被吞（P2）** `_build_client` `except: return None`，调用侧永远提示“check API Key”；区分“未配置 key”与“构造失败”。
- **L-17 超时重试可能重复计费（P2）** 非幂等 POST 无 Idempotency-Key；Responses 未传 `store=False`。固定 `store=False`，超时重试计数并在汇总标注。

---

## 6. 测试与工程

- **T-01 无 GUI 测试（P0 工程）**：254 个测试没有一个实例化 `MainWindow`，P0-01 这类“整个应用不可用”的缺陷完全漏过。最低补 3 条：打开文件、拖入文件、分页填表；再补状态机（失败行不被改回 success）、关闭流程。
- **T-02 CI 无测试（P2）**：`.github/workflows/release-to-nexus.yml` 只在 release 构建，没有 push/PR 的 pytest job；加 Windows + Python 3.11 的测试工作流。
- **T-03 测试卫生（P3）**：从 repo 根跑 pytest 会追加 `logs/app.log` 并写 `.pytest_cache`；fixture 把 `general.log_file`/cwd 指到 `tmp_path`。
- **T-04 关键盲区**（按模块）：
  - parser：`null/true/false/[]`、提示词回显、模板占位值、`,}` 破坏、bare KV 转义、JSON 形状 UI 文本、深嵌套、NUL；
  - quality：refusal 假阳性、缩写 identity、双语 containment、proper-noun 英文标签、尾部换行、零宽/纯标点、覆盖率；
  - translator：长文本重试消息、完整性、非中文目标、cache 指纹、角色交替；
  - RAG/vector：空/重复 keyword、CJK 1-2 字、NaN/零范数、重复 add_vector、top_k=0、并发 close/检索、部分构建；
  - glossary：128KB 字段、小写两列表头、未闭合引号、分号启发式、并发保存、BOM；
  - XML/MCM：控制字符、编码保真、.bak、另存为路径、NUL、UTF-16、纯 CR、未知语言后缀；
  - LRU：TTL/过期清扫、并发 save、字节上限。
- **T-05 死代码（P2/P3）**：`extract_titlecase_phrases`（无调用）、`get_last_rag_debug_info`（无调用）、`invalidate_by_style`（无调用）、`file_formats.detect_translation_file_type/classify_translation_file`（无调用）、`mcm_processor._detect_encoding`（无调用）、`GlossaryWorker._wait_if_paused`（无调用）、`GlossaryWorker._log_batch`（从不 append）、`search_cosine`（无调用且 top_k=0 有 bug）、`prune_rich_meta/last_backup_path/get_rich_meta/set_rich_meta_batch`（无调用）、`_glossary_corrupt`（只写不读）。
- **T-06 文档偏差（P3）**：README 称“254 测试通过”“RAG 完善”，但未提 GUI 无测试、rich metadata 未接入、索引无完整性标记；README 的 `scikit-learn`、“Python 3.8”与 `requirements.txt`（3.10+）不一致；`build_exe.py` 的 locales/prompts 覆盖说明与 frozen 行为不符。

---

## 7. 修复路线图

### 迭代 1（0.5-1 天）：P0 阻断与数据损坏

1. 补 4 个类常量 + 3 条 GUI 冒烟测试（P0-01）。
2. 加载入口互斥 + 文档代次过滤（P0-02）。
3. `closeEvent` stop→wait→save→close；`aboutToQuit` 兜底（P0-03）。
4. XML 赋值前校验/回滚；MCM 先 encode 再 tmp+replace；GUI 检查返回值（P0-04/05）。
5. 向量 store 读写锁或内存加载；`vectors is None and terms` 强制全量重建；保存前行数断言（P0-06/07）。
6. 重建写临时索引 + meta `complete/expected`；停止信号真正下发（P0-08）。
7. 词库 RLock + uuid tmp + 回滚；CSV `strict=True` + 分隔符试解析（P0-09/10）。
8. embedding 请求带 `dimensions` + 首响应校验（P0-11）。

**验收**：GUI 能打开/拖入/分页 10k 行文件；中途加载第二个文件不污染；关窗无 QThread 警告且缓存落盘；编码失败不产生 0 字节文件；默认文本嵌入配置能建成索引；断网/取消后索引状态为“部分/失败”而不是 ready。

### 迭代 2（2-3 天）：翻译质量与流畅性硬伤

1. 覆盖率门禁 + `finish_reason`；containment 不再短路 latin_ratio；纯标记/纯标点拒绝（Q-01/02/03/04）。
2. 长文逐块质检 + 证据化重试 + 保留段落空白 + 块头 i/n（Q-13/36/37）。
3. identity 两端修正：UI 词表/缩写/中文源 passthrough；identity warning 不入缓存；`entry_id` 不再让标识符直通失效（Q-29/30/35）。
4. 术语默认采用 + 一致性检查 + forbidden/variants（Q-05/06/07/08）。
5. 重试语义：永久错误直通、预算只在格式错误时加、角色交替、模板优先级（Q-11/12/14/15/17）。
6. 解析器：最后对象优先 + 模板占位跳过 + 标量/解说词拒绝 + 批量部分回退/无 id 不可信（Q-18/19/20/21/26/27）。
7. UI：表头 Interactive + 填完一次 resize、汇总 100ms 合并、着色信号屏蔽、进度增量、批量信号、填充令牌、空表翻译全部（F-01~F-07）。
8. 配置/缓存：保存失败提示、i18n 中文 Windows、清除所选不清全库、TTL 浮点、配置键补齐（C-02/03/04/05/10）。
9. XML 保真（声明/BOM/EOL/实体）+ .bak 多代 + 另存为路径（X-01/02/03）。

**验收**：书籍/长文不再丢段；`Health` 不再回填英文；`SKSE` 不再失败；术语表改动能生效；10k 行翻译时窗口可用、进度不卡；保存失败可见。

### 迭代 3（1-2 周）：RAG 完善与工程化

1. 确定性命中扫描（Aho-Corasick）+ 保留短语 + 批量 GEMM + 统一融合分/全局重排（R-01/13/14/15/16/17）。
2. 索引健康：glossary_hash/complete、增量补齐/差集删除、checkpoint 追加写、预分配（R-21/22/23/24/25）。
3. 成本：短文本本地化、关键词/embedding 持久化、失败熔断、debug 关闭（R-19/20/34/36/39/40）。
4. 元数据接入 search/prompt/quality（R-30/31）。
5. 启动：openai 惰性导入、RAG 延后初始化、trigram 懒建、词表指纹增量（F-18/19/20/21）。
6. 词表/向量 UI 线程后台化（F-09/10/11）、EmbeddingCache float32 + 持久化。
7. 正则加界（§2.3）、日志异步/加锁、单实例、打包路径、CI。
8. 表格迁移 `QAbstractTableModel`（一次解决填充/状态/主题/内存）。

**预期收益**：10k 行文件打开 30.9s→0.2s；启动 0.6-1.4s→0.3-0.5s；RAG 检索 20k 19.4→8ms、100k 92→50ms；关键词调用 −50-70%；索引重建写量 O(n²)→O(n)；GUI 每行回填 0.38→0.03ms。

---

## 附录 A：实测数据（关键）

| 指标 | 数值 |
|---|---|
| 测试基线 | 254 passed / 51 subtests / 1.52s |
| `import src.gui_main` | 442ms（openai 364ms；打桩后 126ms） |
| `MainWindow()` 空词表 / 30k 术语 | 123ms / 601ms（RAG 493ms） |
| GlossaryManager.load 100k | 455ms |
| VectorStore.load 100k（含 trigram） | 471ms（trigram 400-522ms，73.4MB） |
| 首次 RAG 搜索 100k | 279ms（token_df 174ms） |
| `search_cosine_full` 5k/20k/100k | 0.43/4.13/15.7ms |
| `RAGSearcher.search` 5 keyword 20k/100k | 19.4/92.0ms |
| `translate_text`（假 LLM，20k 索引） | 20.5ms/条 |
| 5000 条端到端 | 105.7s（21.1ms/条），峰值 RSS 424MB |
| QualityChecker 426/5124 字 | 2.49/40.6ms（78% 逐空白正则） |
| 分页填表 10k 行 | 30.9s（现状）/0.19s（Interactive） |
| 单行 GUI 更新 | 129.8µs（124.6µs = 5×setStyleSheet） |
| `completed_total` 求和 20k | 13,798ms |
| `refresh_term_list` 100k/键 | 21.4ms |
| EmbeddingCache 满载 4809×1536 | 283MB（float32 约 30MB） |
| TranslationCache 50k | 30.2MB；save 138ms/7.2MB |
| ReDoS `%`+20k/32k 零 | 7.8s/20.7s |
| ReDoS 未闭合注释 432KB | 8.4s |

## 附录 B：覆盖与验证

- GUI：`src/gui_main.py` 全文分段审读（性能 + 状态机），关键路径用 offscreen Qt 微基准和真实方法复现（itemChanged、QThread 析构、ResizeToContents、逐行 QSS）。
- 翻译：`translator.py`、`prompt_builder.py`、`quality_checker.py`、`response_parser.py`、`text_analyzer.py`、`style_profiles.py`、`prompt_manager.py`、`prompts/*` 全读；用假 LLM/RAG 构造 8 类样本抓取真实 prompt 并测量成本。
- RAG：`search.py`、`engine.py`、`vector_store.py`、`keyword_extractor.py`、`glossary_manager.py`、`glossary_import.py`、`embedding_cache.py`、`lru_cache.py` 全读；5k/20k/100k 合成索引实测。
- 文件格式：`xml_processor.py`、`esp_xml_processor.py`、`xml_content.py`、`safe_xml.py`、`mcm_processor.py`、`file_formats.py` 全读 + 真实 fixture 往返。
- 传输/基础：`llm/client.py`、`retry.py`、`reasoning.py`、`cost_tracker.py`、`config/*`、`logging_helper.py`、`i18n.py`、`main.py`、`build_exe.py` 全读；用真实 `openai` 异常对象与临时目录实验。
- 已核查无问题（摘要）：源文注入防护、受保护 token 顺序/大小写、空白策略对 CJK 的兼容、解析器对围栏/尾逗号/单引号/裸 KV/数组根/截断 JSON 的恢复、批量结果不错位、style profile 继承/防环、XXE/billion laughs/UTF-16 DOCTYPE 拒绝、CDATA/注释/BOM/CRLF 往返、sentinel restore 逐字节、LRU 基本淘汰/锁外序列化、向量归一化/维度/零范数/NaN 查询、`argpartition` 性能、空状态保护。
- 未覆盖：Windows 实机杀软/OneDrive 占用、真实 API 供应商兼容矩阵、多显示器 DPI、无障碍/键盘导航、翻译嵌入格式（ESP `.strings`/`.dlstrings`）直接读写（当前只支持 XML/MCM）。

> 本报告前 7 节为只读审查产物（审查时仓库未修改）；第 8 节记录随后落地的修复。审查脚本与 fixture 位于 `%TEMP%`（`skyrim_audit/`、`skyrim_fuzz/`、`skyrim_e2e/`、`skyrim_xml/`、`skyrim_mcm/`）。

---

## 8. 修复进展（2026-11-04，第二轮）

本轮按文件所有权拆成 6 个并行修复流（GUI / XML-MCM / 向量索引 / 词库-关键词 / 翻译质量 / 传输-缓存-配置），主代理负责集成、冲突修复与验收。**改动未提交 git**，当前 `git status` 为 41 个修改文件 + 6 个新测试文件（约 +5.2k 行）。

### 8.1 已修复（P0）

| 编号 | 修复内容 | 验证 |
|---|---|---|
| P0-01 | 补齐 4 个 `MainWindow` 类常量，打开/拖入/分页恢复 | 无头 `MainWindow` + 真实 XML 打开：2 行正确入表 |
| P0-02 | 加载入口互斥 + 文档代次；旧 Worker 结果（含 worker 置空后的迟到结果）按代次丢弃 | `tests/test_gui_smoke.py`；无头翻译写入 2 行 |
| P0-03 | `closeEvent`: stop → wait(3s) → 保存翻译缓存 → 关 client → accept | GUI 冒烟 + 单元测试 |
| P0-04 | XML 节点赋值前校验 XML 非法码点，失败保留原节点并向上抛错；GUI 标 FAILED | `tests/test_file_safety.py` + 手工探针 |
| P0-05 | MCM 先 encode 再同目录 tmp+fsync+replace，失败不碰目标、不产生 .bak | 手工探针：ASCII 编码失败后目标字节不变；50 例测试 |
| P0-06 | 向量 mmap 读者计数/写者屏障，关闭前等待读者排空 | `tests/test_vector_index_safety.py`（并发 search+close） |
| P0-07 | `vectors=None and terms` 拒绝增量；保存前断言行数一致 | 手工探针：拒绝写入；不一致保存抛错且不落盘 |
| P0-08 | 重建先备份旧索引、meta `complete/expected_term_count`、失败/取消回滚，状态机 `partial_build/glossary_mismatch` | 28 例索引安全测试 |
| P0-09 | 词库写锁、uuid tmp、WinError 重试、异常回滚 + `rebuild_lookup` | 并发/只读回滚测试 |
| P0-10 | CSV `strict=True`、`csv.Error` 转带行号错误、分隔符试解析 | 手工探针：未闭合引号报错；分号 CSV 两列正确 |
| P0-11 | embedding 显式发送 `dimensions`、按 index 排序、校验条数与维度、拒绝时回退并记忆 | 手工探针：请求 `dimensions=1536`，返回 1536 维；新增 80 例传输测试 |

### 8.2 已修复（P1，摘要）

- **翻译质量**：覆盖率 `coverage`、空可见译文 `empty_visible`、containment 不再短路 latin_ratio、UI 单词/缩写/中文同文/拒绝语误判、术语 `glossary_fragment` 一致性、长文逐块质检 + 证据化重试 + 段落保持、CJK 分块句界、解析器（最后候选/模板回显/标量/解说词/尾逗号/批量部分回退/无 id 拒绝）、重试差异证据/角色交替、提示词去硬编码与 UI 长度/称呼规则、`format_extra_retries` 仅用于格式错误。
- **RAG**：Unicode 归一化（俄/日/全角）、精确 glossary 命中前置于低信号规则、glossary 命中短语不分解、`[]` 不再丢给正则兜底、关键词缓存语义/指纹/失败不缓存、关键词解析仅接受 `str`/行解析/英文拒绝语、词库导入 BOM/表头/未知 op、索引 NaN/重复词/top_k、删除批量化、词表 key 哈希状态。
- **GUI/工程**：表头 `Interactive` + 单次列宽 + 关闭更新，10k 行填充 30.9s→0.19s；逐行汇总 100ms 合并、气泡样式移出热路径；进度 O(n²) 改增量；空表“翻译全部”自动等待加载；词库暂停/取消下发 stop/pause；主题/配置保存提示；状态着色屏蔽 `itemChanged`，FAILED/WARNING 不再被改回 success。
- **文件/配置/传输**：MCM cp1252/GBK/CJK 判定、UTF-16 BOM、语言后缀、EOL/清空语义；XML `.bak` 不覆盖、另存为切换路径、编码/BOM 保持；`save_config` 返回 bool 且 GUI 失败提示；TTL 浮点、i18n 中文 Windows 检测、日志 Handler 锁、LRU 并发保存、超时分类、403/404/上下文超长分类、`finish_reason` 截断异常、Retry-After/5xx、思考参数回退、embedding 顺序/条数校验。

### 8.3 集成阶段新发现并修复

1. **Windows 绝对路径全部被拒（阻断）**：`_validate_translation_path` 把盘符 `:` 当作非法字符，`browse_file` 对所有 `C:\...` 路径返回“路径包含非法字符”。已改为仅检查盘符/`\\?\` 前缀之后的字符，并新增回归测试。这个问题是端到端无头冒烟发现的，审计阶段未覆盖。
2. **解析器把合法译文当模板占位符**：`_TEMPLATE_PLACEHOLDER_VALUES` 含 `译文/翻译/translation/略`，模型返回“翻译”会被判空；已收窄为仅 `这里填写译文/填写译文/...`，避免误杀。
3. **缺失 i18n key**：补齐 11 个（含 `msg_invalid_path_chars`、`msg_config_save_failed`、`msg_vector_index_stale_prompt` 等），en/zh 各 321 键、键集一致。
4. **索引陈旧提示只认 fingerprint_mismatch**：现在 `partial_build/glossary_mismatch/dimension_mismatch/...` 也会提示重建。
5. **`format_extra_retries` 无条件 +2**：现在仅格式错误使用额外预算，identity 失败从 3 次调用降为 1 次。
6. **风格提示词仍硬编码“中文”**：改为目标语言条件表述（CJK 用全角标点、其他语言用标准标点）。

### 8.4 验收

- `python -m compileall -q src main.py` 通过；`git diff --check` 通过。
- `python -m pytest tests -q` → **451 passed, 86 subtests passed**（审查前 254）。
- 无头端到端：`MainWindow` 构造 → 打开 2 行 XML → 状态正确 → 翻译（假 LLM）→ 2 行 success、进度 100 → 保存后 XML 含 2 处译文。
- 手工探针：MCM 编码失败目标不变；XML 非法字符保留旧 Dest；CSV 未闭合引号报错、分号 CSV 正确；向量 stale 拒绝/不一致不落盘/`top_k<=0` 返回空；embedding 请求携带 `dimensions` 并校验长度。

### 8.5 仍未修（按报告编号，建议下一轮）

**RAG 召回/排序**：R-04（正则兜底 O(全词表)/噪声）、R-13..R-18（融合分/全局重排/桶顺序/上下文按分截断/IDF/anchor）、R-23/R-24（checkpoint 全量重写与预分配）、R-26/R-27（terms 行号错位、范数抽样）、R-30/R-31（rich metadata 接入排序/提示/一致性）、R-32 之外的词表注入加固、R-34 短文本本地化、R-39 embedding 持久化。
**性能**：RAG 批量 GEMM（多 keyword 一次矩阵乘）、启动 `openai` 惰性导入（−364ms）、词表/索引后台化、`QAbstractTableModel` 迁移、ReDoS 量词加界、日志异步。
**翻译**：Q-02 与 `ResponseTruncatedError` 的拆块重译联动、Q-43 非 CJK 目标脚本门禁、Q-45 journal_letter 映射、Q-46 speaker/listener 上下文、Q-51/52/54/55/56 缓存键与持久化细节。
**文件/工程**：X-01 完整字节保真（standalone/CRLF/属性引号/实体写法/空元素）、X-09 512KB 预检窗、M-07 行首缩进、`file_formats.py` 死代码收敛、CI pytest job、GUI 测试覆盖扩展到真实 Worker/配置保存失败路径。

---

## 9. 修复进展（2026-11-04，第三轮）

第三轮按"性能 / RAG 完善 / 翻译质量"三条主线并行推进，7 个子代理 + 主代理集成，完成 §8.5 中的绝大多数条目。所有改动仍未提交 git（见 §9.7）。

### 9.1 本轮完成项（按报告编号）

| 编号 | 修复 |
|---|---|
| R-04 | 正则兜底改为"原文 token → 词条倒排"候选查询（懒建、按词表指纹失效），df=0 不再算信号词，罗马数字/缩写/句首普通词/常见动词排除；40k 词表冷建 10.5ms、热查询 0.03ms/条 |
| R-07/08/09 | containment 改整 token/词序列匹配（`war` 不中 `Warning`）；单 token 多候选按原始 cosine 取 top-1×0.92（差 >0.15 才放弃）；去掉 `-ly` 一刀切，所有格剥离后判中心名词，低信号仍走词法 |
| R-13..R-16 | 统一融合分（raw cosine + direct/containment/overlap 加成，不 clamp）；阈值只作用于归一化最终分；`(-fused, len(term), term)` 确定性 tie-break；先全局排序再按 short/long 配额截断（单调、非先到先得） |
| R-17 | 检索结果按融合分有序返回；prompt 上下文按顺序截断（消费侧） |
| R-18 | anchor token 只从 df>0 且 df≤max_df 选，分母用实际 anchor 数 |
| R-19/20 | `return_debug=False` 时零 debug 分配/零 deepcopy；`RAGSearcher` 接收 stop/pause 回调，在关键词循环/embedding 批次/候选打分边界生效，engine 已接线 |
| R-21/22 | 状态机比较 key 集合哈希；增量构建差集删除孤儿向量，空词表清空整库 |
| R-23/24 | 1.5× 预分配缓冲 + 每批直接写入（去掉两次 vstack）；checkpoint 间隔自适应 `max(config, len/2)`，结束/停止/异常前 flush；12000 词实测 7 次全量写（固定 1000 行需 12 次、≈78k 行） |
| R-25 | 批量删除沿用 `delete_terms_batch` |
| R-26/27 | terms 任何非法项 → `terms_corrupt` 拒绝加载（不再截断错位）；`metadata_missing` 按 `embedding` 键判断；加载抽样 ≤64 行校验 L2 范数并与 `unit_l2` 对账修复 |
| R-28/29 | uuid4 tmp + WinError 5/32 重试 + `.bak` 回退；metadata RLock；词法索引局部构建替换 + 锁内快照 |
| R-30/31 | 多译法导入主译 + `variants`；`forbidden` 进提示词并做输出拦截（blocking error）；`priority` 参与排序；`variants` 作为别名召回；术语一致性做 warning；sidecar 与 envelope 逐字段合并 |
| R-32 | 词表注入转义（`_sanitize_glossary_field` → `_escape_source_delimiters`）覆盖新增的 variants/forbidden 字段 |
| R-34 | 空词表/索引未建跳过 RAG；≤40 字且 ≤3 token 用 `[text]` 直配，不调关键词 LLM；`rag.short_text_local_keywords` 可关 |
| R-36 | 缓存原始响应、失败/降级不缓存、配置指纹补全、空白归一化 key、返回副本 |
| R-39 | `EmbeddingCache` npz 持久化（float32 + fingerprint 命名空间 + 脏标记 + 200 条/30s 节流），换模型自动切命名空间 |
| R-40 | 关键词连续 3 次失败熔断 60s，日志一次 |
| 3.6-1 | **新增确定性命中扫描器**（`src/rag/deterministic_scan.py`）：词首倒排 + 第二 token 子索引 + CJK 字符 trie，整篇扫描，40k 术语重建 79ms、400 字扫描 0.06ms、内存 ≈5.6MiB；接入 `RAGSearcher.search`，逐字命中按配额优先返回（即使关键词 LLM 返回空） |
| Q-02 | `ResponseTruncatedError` 不再重试；单条/批量/长文块均可拆块重译（最多 2 层），保留成功部分 |
| Q-37/38/39 | CJK 分块允许标点后无空格切分（15/15 句末）；首尾空白按源串恢复；行尾换行可容忍 |
| Q-43 | 目标脚本门禁：ru/uk 西里尔、en/fr/de/es/it/pt/pl/tr 拉丁，`wrong_script` 纳入 blocking |
| Q-45/46/48 | BOOK+EDID 推断 `journal_letter`；dialogue 注入上一句原文（`context_hint["previous_source"]`，不进入缓存键）；system 补 UI 长度/中文排版/反翻译腔/JSON few-shot，删重复 hedge |
| Q-51..Q-56 | 指纹补 6 个配置键；无空白短文本缓存键去 `entry_id`；warning 不缓存；缓存命中质检复用 issues；matched_terms 口径统一；200 条/30s 线程安全落盘 |
| X-01 | 布局快照（sha256/声明/BOM/EOL/末尾换行/编码）；无修改保存**不重写文件**（字节天然一致），有修改手工拼 BOM+原声明+正文+原 EOL；实测 318B 文件无修改保存逐字节一致 |
| X-05..X-11 | CDATA 文本语义 + 警告；新建 Dest 按兄弟缩进、空译文不建；删 `forbid_dtd` 死分支；预检状态机分块扫到文件尾、先跳过 PI/注释/CDATA；失败原因回调 + `last_load_error`；旧树节点抛 `StaleNodeError` |
| M-06/07/09/10 | MCM 逐行 EOL 保持；行首缩进保存；`set_dest(overwrite=True)`；`file_formats` 内容识别（大小写不敏感、`.txt` 需 MCM 形状）并接入 GUI |
| F-08..F-15 | 词表搜索 150ms 去抖 + 小写缓存（100k 每键 0.002ms）；导入预览后台线程并复用解析结果；新增术语后台 embedding；RAG 可视化线程化；主题切换批量重绘（10k 行 1.98s→122ms）；日志目录缓存/单换行/跨线程 signal（1.4µs/条，8 线程 1200 行零丢失） |
| F-18 | `openai` 惰性导入：`import src.gui_main` 442→**134ms**，`sys.modules` 无 openai |
| F-13 | RAG 延迟初始化：窗口先出（构造 70ms），show 后后台建索引（30k 术语 129ms） |
| C-13 | 中英各 333 键、集合一致；src 内 i18n 字面量全覆盖 |
| T-02 | 新增 `.github/workflows/tests.yml`：push/PR 触发，Windows + Python 3.11/3.13，compileall + locale 键校验 + pytest |

### 9.2 实测对比

| 指标 | 修复前 | 修复后 |
|---|---|---|
| `import src.gui_main` | 442ms | **134ms** |
| 10k 行主题切换 | 1.98s | **122ms** |
| 词表搜索 100k 每键 | 21.4ms | **0.002ms**（150ms 去抖） |
| 正则 `%`+20k `0` | 7.76s | **1ms** |
| 8000 个未闭合注释 | 8.4s | **39ms** |
| `"[a"*20000` / `"<"*20000` | 236 / 118ms | **47 / 44ms** |
| 166KB 重复 JSON | 27.99s | **≤8ms** |
| 5124 字保护壳 / 质检 | 19.5 / 40.6ms | **1.2 / 5.7ms** |
| CJK 3.7k 字分块 | 句中硬切 | **15/15 句末** |
| XML 无修改保存 | 全量重写 | **字节完全一致** |
| RAG 批量打分 20k×5 | 1.2ms | **0.6ms**（误差 <2e-7） |
| 确定性扫描 40k 词表 | — | 重建 **82ms**、1.4KB 扫描 **8ms** |

### 9.3 设计决定与已知边界

- **确定性命中顺序**：`search()` 返回顺序为"逐字命中（源码序）+ 其余（融合分降序）"。逐字命中是最高置信类别，放在最前是有意为之，融合/配额层本身仍是严格降序。
- **归一化退化保护**：语义候选 min-max 仅在 `band≥3 且 spread≥0.05 且 raw_max≥threshold` 时启用，否则用原始 cosine；单条弱候选不会被抬过阈。
- **X-01 仍不保真**：`<Dest></Dest>`→`<Dest/>`、`&#160;`/`&quot;` 实体写法、属性引号统一、顶层 PI 间空白、XML 内部混合换行归并、未知编码回退 UTF-8（有 warning）。未修改保存时这些均不受影响。
- **R-23** 未做真正 append-only 分片文件，采用预分配 + 自适应间隔（写量从 O(n²/1000) 降到约 O(n log n)）。
- **Q-02** 边界：各子块都成功但合并后 QC 仍失败时记录 failed，不重发原请求。
- **R-40** 只捕获 `extract_keywords` 显式异常/超时（keyword_extractor 内部会把 LLM 错误降级为正则，不冒泡）。
- **EmbeddingCache** 无 atexit 强制 flush：<200 条且 <30s 退出时该段缓存不落盘。
- **未做**：`QAbstractTableModel` 迁移（批量刷新已把 10k 行主题切换降到 122ms）、Q-40/Q-41（按块 RAG 术语与失败块缓存）、R-30 的 `domain` 过滤（variants/forbidden/priority 已接入）、G-04 单行表头 legacy 兼容、X-12 可配置大小上限。

### 9.4 验证

- `python -m pytest tests -q` → **660 passed, 118 subtests passed**（第二轮 451/86，审查前 254）。
- `python -m compileall -q src main.py` 通过；`git diff --check` 通过。
- `python audit/bench_round3.py` 复现 §9.2 全部数字。
- 无头 GUI：`MainWindow` 构造 → RAG 未就绪/就绪两态、10k 行主题切换 <0.5s、100k 词表每键 <3ms、跨线程日志、`previous_source` 注入均通过。
- 手工探针：XML 无修改保存字节一致；MCM 编码失败目标不变；CSV/向量/embedding 行为与第二轮一致。

### 9.5 仍未做（建议下一轮）

1. `QAbstractTableModel` 迁移（一次性解决填充/状态/主题/内存，当前靠批量刷新达标）。
2. Q-40/Q-41：长文按块做 RAG（关键词取前 8k 或逐块合并）、失败块部分缓存。
3. R-30 `domain` 过滤与 `examples` 注入；多译法在 UI 中的选择。
4. R-23 真正 append-only 分片索引；R-27 全量范数校验（当前抽样 64 行）。
5. X-01 剩余字节形态（空元素/实体/属性引号）与混合换行。
6. X-12 可配置 50MB 上限；MCM 大文件流式解析。
7. 真实网络下的端到端回归（LLM/embedding 429/超时/断网），以及 CI 在 GitHub 上的首次运行。
8. R-38 关键词请求的 JSON mode/temperature 参数（client 已支持 `response_format`，`extract` 尚未启用）；R-33 sidecar 内容未纳入词表指纹。
9. 单实例、打包路径、异步日志队列。

