# 基线无病与发病(Incident)标签说明

开题报告里写到“纳入基线无目标慢性病人群”，这类设计通常对应“发病风险预测”(incident risk prediction)：

- Wave1 作为 baseline
- 过滤掉 baseline 已患病的人
- 结局标签用后续随访(wave2-5)的“新发”来定义

## 为什么 NHANES 当前流水线做不到严格 incident

目前仓库使用的是 NHANES 单次横断面数据(问卷 + 实验室)，没有个人随访的发病时间点。
因此训练目标更接近“当前患病/既往诊断的概率”(prevalence modeling)，不是严格的“未来新发风险”。

这不影响答辩展示与方法学训练，但论文需要在“局限性”里明确说明。

## CHARLS 里怎么做 incident(推荐)

如果你拿到 Wave1-5 + Life History，可以按下面口径构造:

1. baseline 定义
- 以 Wave1 为 baseline
- baseline 年份可用 Wave1 调查年份或 Life History 的访谈年份

2. baseline 无病过滤
- 对每个任务(如 diabetes)，用 Wave1 的诊断变量/自报变量筛掉 baseline 已患病者

3. incident 标签定义(二选一或组合)
- 随访新发: wave2-5 任一 wave 首次出现“已诊断”=1
- 发病年份: Life History 的发病年份在 baseline 年份之后

4. 模型评估
- 用 NHANES 训练好的模型(不再训练)去 CHARLS 上做外部验证时，如果 CHARLS 标签是 incident，那么解释就更严谨

## 实操建议

为了让脚本能自动化跑起来，建议 CHARLS 处理后数据(见 `docs/charls.md`)补齐两类列:
- baseline 状态: `baseline_{task}` (0/1)
- incident 结局: `incident_{task}` (0/1)

等你 CHARLS 数据到手并确认原始变量名后，我可以把“从原始 wave 文件 -> 生成 processed/charls_model_dataset.csv + incident 标签”的脚本补齐。