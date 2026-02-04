# 停用词配置说明 (Stopwords Configuration Guide)

## 概述

`stopwords.json` 是RAG引擎的外部停用词配置文件，用于过滤在关键词提取过程中容易被误判为专有名词的通用词汇。

## 为什么需要停用词？

在RAG (Retrieval-Augmented Generation) 流程中，系统会从原文中提取关键词以查询术语表。但是某些词汇容易导致误判：

### 常见问题

1. **句首大写误判**
   - 原文：`Time to confront him` 
   - 问题：`Time` 因句首大写被误判为专有名词
   - 后果：强制匹配术语 "Time : 时间"，导致翻译生硬

2. **通用词汇污染**
   - 原文：`Guard the gate`
   - 问题：`Guard` 被误判为专有名词
   - 后果：与 "Whiterun Guard" 混淆，浪费检索资源

3. **漏抓关键实体**
   - 原文：包含 "Riften"（地名）、"Love Potion"（物品）
   - 问题：这些重要实体未被提取
   - 后果：缺失关键术语参考

## 配置文件结构

```json
{
  "sentence_starters": {
    "_comment": "句首常见词",
    "terms": ["Time", "Now", "Then", ...]
  },
  
  "generic_actions": {
    "_comment": "通用动作词",
    "terms": ["Attack", "Defend", ...]
  },
  
  "generic_categories": {
    "_comment": "通用类别词",
    "terms": ["Time", "Space", "Thing", ...]
  },
  
  "risky_ambiguous": {
    "_comment": "容易混淆的词",
    "terms": ["Guard", "Hunter", ...]
  },
  
  "custom_exceptions": {
    "_comment": "自定义例外",
    "terms": []
  }
}
```

## 各类别说明

### 1. sentence_starters（句首常见词）
因句首大写而容易被误判的通用词。

**示例：**
- `Time` - "Time to go" 中的 Time 不是游戏术语
- `Now` - "Now I know" 中的 Now 不是专有名词
- `Then` - "Then he said" 中的 Then 不是游戏实体

**注意：** 如果词汇确实是游戏术语且在术语表中有精确匹配，仍会被提取。

### 2. generic_actions（通用动作词）
通用的英语动作动词，通常不是游戏专有名词。

**示例：**
- `Attack`, `Defend`, `Fight` - 通用动作
- `Open`, `Close`, `Lock` - 通用操作

### 3. generic_categories（通用类别词）
单独出现时为通用类别的词，除非与具体名称组合。

**规则：**
- 单独的 `Spell` 会被过滤 ❌
- 组合的 `Fire Spell` 会被保留 ✅
- 单独的 `Guard` 会被过滤 ❌
- 组合的 `Whiterun Guard` 会被保留 ✅

### 4. risky_ambiguous（容易混淆的词）
需要上下文判断的词汇，单独出现时不提取。

**示例：**
- `Guard` - 可能是"守卫"（动词）或"Whiterun Guard"（NPC）
- `Hunter` - 可能是"猎人"（职业）或"Predator Hunter"（特定NPC）

### 5. custom_exceptions（自定义例外）
用户根据实际翻译情况自定义添加的停用词。

## 使用方法

### 基本使用

1. 编辑 `stopwords.json` 文件
2. 在相应类别的 `terms` 数组中添加或删除词汇
3. **无需重启程序** - 下次翻译任务时自动加载

### 添加自定义停用词

在 `custom_exceptions` 部分添加：

```json
"custom_exceptions": {
  "_comment": "自定义例外 - 用户添加",
  "terms": [
    "Look",
    "Listen",
    "Another"
  ]
}
```

### 临时禁用某个类别

将整个类别注释掉或清空 `terms` 数组：

```json
"generic_actions": {
  "_comment": "通用动作词 - 已禁用",
  "terms": []
}
```

## 工作原理

### 过滤逻辑

1. **LLM 提取关键词** → 初步关键词列表
2. **应用停用词过滤**：
   - 单个词：检查是否在停用词集合中
   - 多词短语：检查实质性词汇是否全部为停用词
3. **停用词优先原则**：停用词过滤优先级**高于**术语表匹配
   - 如果单个词在停用词中，即使在术语表也会被过滤
   - 如果需要该词的游戏术语含义，应以组合形式出现（如 "Stop Time"）

### 示例流程

**原文：**
```
Time to confront Aerin about the Love Potion used in Riften.
```

**提取过程：**

| 步骤 | 关键词 | 结果 | 原因 |
|-----|--------|------|------|
| LLM提取 | Time | ❌ 过滤 | 在 sentence_starters 中（停用词优先） |
| LLM提取 | Aerin | ✅ 保留 | 人名，不在停用词中 |
| LLM提取 | Love Potion | ✅ 保留 | 物品名，组合词 |
| LLM提取 | Riften | ✅ 保留 | 地名，不在停用词中 |

**最终关键词：** `["Aerin", "Love Potion", "Riften"]`

**注意：** 即使 "Time" 在术语表中有匹配（Time: 时间），也会被停用词过滤。如果需要 "Time" 的游戏术语含义（如魔法 "Stop Time"），应该以组合形式提取。

## 最佳实践

### 1. 观察日志

检查翻译日志中的 RAG Debug 输出：
```
[Keywords]
- Time    ← 如果看到这个，可能需要添加到停用词
- Aerin   ← 正常的关键词
- Riften  ← 正常的关键词
```

### 2. 渐进式添加

不要一次性添加太多停用词：
1. 先运行几次翻译任务
2. 观察 Debug Log 中的误判
3. 逐步添加明确的误判词汇

### 3. 避免过度过滤

**警告：** 不要添加可能是游戏术语的词汇！

❌ **不要添加：**
- `Mara` - 这是游戏中的神祇
- `Sovngarde` - 这是游戏中的地名
- `Thane` - 这是游戏中的头衔

✅ **应该添加：**
- `Time` (在 "Time to go" 语境中)
- `Look` (在 "Look at this" 语境中)
- `Wait` (在 "Wait a moment" 语境中)

### 4. 定期审查

每隔一段时间检查停用词配置：
- 删除不再需要的词汇
- 添加新发现的误判词汇
- 调整类别归属

## 技术细节

### 归一化处理

所有词汇在比较前都会被归一化：
- 转为小写
- 移除标点符号
- 处理空格

因此 `"Time"`, `"time"`, `"TIME"` 都会被同样处理。

### 性能优化

- 停用词存储为 `frozenset`，查询时间复杂度为 O(1)
- 配置文件仅在启动时加载一次
- 不影响翻译主流程性能

## 故障排除

### 问题：添加停用词后仍被提取

**可能原因 1：** LLM 将其作为多词短语的一部分提取

**示例：** `"Stop Time"` - 即使 "Time" 在停用词中，作为魔法名称的组合词会被保留

**解决：** 
1. 这是正确行为 - 游戏术语应该保留
2. 如果不想要，需要在 `stopwords.json` 中添加完整短语

**可能原因 2：** 配置文件未正确加载

**解决：**
1. 检查 `config.json` 中的 `paths.stopwords_file` 路径
2. 验证 JSON 格式是否正确
3. 查看日志中的停用词加载信息

### 问题：重要的游戏术语被过滤了

**原因：** 错误地添加到了停用词中

**解决：**
1. 从 `stopwords.json` 中移除该词
2. 无需重启，下次翻译会自动生效

### 问题：停用词配置未生效

**检查：**
1. 配置文件路径是否正确（`config.json` 中的 `paths.stopwords_file`）
2. JSON 语法是否正确（使用 JSON 验证工具）
3. 查看启动日志中的加载信息

## 相关文件

- `stopwords.json` - 停用词配置文件
- `config.json` - 主配置文件（包含 stopwords_file 路径）
- `src/rag_engine.py` - RAG 引擎实现（包含停用词加载和过滤逻辑）

## 更新记录

- **2026-02-04**: 初始版本，支持5个停用词类别
