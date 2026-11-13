# Prompt v2 staging 说明

## 设计原则
两个变体都把弱模型最需要执行的动作写成短步骤：先找施事、动作、受事，再绑定修饰和代词，处理省略，保住否定/数量/条件/语气范围，最后用自然中文重组并静默核验。默认提示词用具体中英例句代替抽象语言学术语；A 追求最短前缀和高信息密度，B 增加一个自检清单与一条例句。两个版本都把格式锚点放在末尾，减少 JSON 输出被后续文字冲淡的机会。
NSFW 采用条件触发：源文没有成人内容时保持普通文本。确有成人内容时，中文直接度、粗俗度和露骨度与源文对齐，禁止含糊替换、淡出和新增警告；同时逐项保留动作角色、身体所属、同意/拒绝/胁迫区别，不补新动作或解剖事实。四条例句覆盖明确欲望、粗俗性交、请求强度和拒绝，解决实测的露骨度过弱问题。

## 文件变更与字符数
字符数按 JSON 中系统 prompt 数组字符串拼接后的 Python len() 统计；现有默认约 1177 字符，现有 NSFW 约 2100 字符（基准来自现场 A/B 记录）。本 staging 实测：
- variant_a default：581 字符；NSFW：597 字符。
- variant_b default：673 字符；NSFW：597 字符。
- style profile 每个 profile 的 rules + content_rules 文本均为 2–3 条短规则，约 20–65 字符。
两套目录都包含 nested loader layout：prompts/translator/system_prompts/*.json、translator/*.json 与 rag/keywords.json，文件内容保留 live keys。

## 示例与实测失败对应
- blade … farmer who has never drawn it：明确关系从句附着和代词 it，目标是修复“从未出鞘挥斩过的农夫”。
- Never look back：覆盖短语/祈使句，避免逐词直译。
- I told him not to open it：覆盖代词与否定作用域。
- Only the captain can enter：覆盖 only 的范围。
- Health：覆盖 UI 标签的短译法。
- Wait here, will you?：覆盖自然对白和语气。
- B 的 The door wasn’t locked…：增加否定与因果联动检查。
- NSFW 四条例句校准“插进来”“操你”“再深一点”和拒绝语句，针对实测弱化和 consent/refusal 混淆。

## 安装命令
在仓库根目录执行以下命令，把任一变体复制到运行目录（会覆盖对应提示词文件）：

    robocopy audit\prompt_v2\variant_a\prompts prompts /E
    robocopy audit\prompt_v2\variant_b\prompts prompts /E

Linux/macOS 或 Git Bash：
    cp -R audit/prompt_v2/variant_a/prompts/. prompts/
    cp -R audit/prompt_v2/variant_b/prompts/. prompts/

建议先备份现有 prompts/，然后在 GUI 中重新加载提示词。两套都保持 <<<SOURCE_TEXT>>> … <<<END_SOURCE_TEXT>>> 数据框定，源文不能覆盖任务指令。

## 验证与限制
已对每个 JSON 文件执行解析检查，并用 PromptManager 指向两个 staging prompts 目录检查 dotted keys。当前未运行 live_eval：本轮没有可用的实际网关凭据/端点，因此没有新增请求或 transcript。仍需在真实 300B/flash 网关上各跑不超过 12 请求，重点比较长关系从句、UI 短标签、否定范围、普通文本在 NSFW 模式下不被性化，以及成人句子的直接度。示例会引导注册用词；不同网关的安全策略、采样和术语表质量仍可能造成残余波动。
