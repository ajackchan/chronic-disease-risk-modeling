# CHARLS 外部验证数据放置约定

本项目的开题报告要求: 在 NHANES 完成内部验证后, 使用 CHARLS 做外部验证。

由于 CHARLS 需要申请授权, 仓库不会提供原始数据。你拿到数据后, 按下面约定放到本地即可。

## 1. 你需要准备什么

你已申请的内容: Wave1-5 + Life History。

建议你优先下载包含以下信息的数据文件 (具体文件名以 CHARLS 提供为准):
- 人口学: 年龄, 性别等
- 体格/身体测量: BMI 或身高体重
- 实验室/生化 (若有): 血糖, 血脂 (TG, HDL, LDL), 胰岛素
- 慢病结局标签: 心血管病, 糖尿病, 肝病, 肿瘤, 多病共存

## 2. 目录结构

把原始压缩包解压到:

- `data/raw/charls/`

推荐做法是保留原始文件名, 并按 wave 分子目录(可选):
- `data/raw/charls/wave1/...
- `data/raw/charls/wave2/...
- ...`

如果你不想分 wave 目录也行, 只要都在 `data/raw/charls/` 下。

## 3. 处理后数据(外部验证输入)

外部验证脚本默认读取:
- `data/processed/charls_model_dataset.csv`

这个 CSV 需要包含和 NHANES 建模一致的字段(列名一致):

特征列(来自 `configs/modeling.yaml`):
- `ridageyr` (年龄)
- `lbxin` (胰岛素)
- `lbxtr` (甘油三酯 TG)
- `lbdldl` (LDL)
- `aip`
- `tyg`
- `tyg_bmi`

标签列(来自 `configs/modeling.yaml` 的 tasks):
- `cardiovascular`
- `diabetes`
- `liver`
- `cancer`
- `multimorbidity_label`

说明:
- 如果 CHARLS 原始变量名不同, 需要做一次映射/重命名, 并按本项目口径计算 `aip/tyg/tyg_bmi`。
- 若暂时缺某些特征(例如没有胰岛素), 可以先做一个缺失版本用于论文局限性说明, 但最好后续补齐。

## 4. 外部验证怎么跑

当 `data/processed/charls_model_dataset.csv` 准备好后, 运行:

```powershell
python scripts/validate_on_charls.py
```

输出位置:
- 每个任务一份指标表: `reports/tables/charls_external_{task}_metrics.csv`
- 汇总对比表: `reports/tables/charls_external_comparison.csv`

## 5. 数据拿不到怎么办

如果 CHARLS 审批还没下来, 你仍然可以先把:
- NHANES 内部验证(已完成)
- 校准曲线/决策曲线(已完成)
- SHAP 解释图(脚本已提供)

这些先写进论文, 把 CHARLS 外部验证部分在方法里写清楚流程, 等数据到手再补实验结果。