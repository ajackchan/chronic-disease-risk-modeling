# 慢性病风险建模网页演示子项目设计

## 1. 目标

本设计文档对应毕业设计的第二个子项目：网页演示系统。该系统不是面向真实医疗业务的产品，而是面向毕业答辩场景的研究成果展示与交互预测页面。

网页首版目标：

1. 展示五类慢性病任务的建模结果
2. 展示 ROC、混淆矩阵、模型对比图等图表
3. 支持预置样本演示
4. 支持手动输入体检指标并调用后端预测
5. 直接复用当前研究主线生成的 CSV、JSON、PNG、joblib 产物

## 2. 定位

网页定位为“单页答辩展示台”，不是普通健康管理产品。

首要受众：

1. 答辩老师
2. 指导教师

因此页面设计优先级如下：

1. 研究结果表达清晰
2. 现场演示稳定
3. 交互预测直观但不过度产品化

## 3. 范围

### 3.1 首版包含

1. 单页研究概览展示
2. 五类任务结果总览卡片
3. 图表切换展示
4. 预置样本演示区
5. 手动输入预测区
6. 前端 React + Vite
7. 后端 FastAPI
8. 调用当前真实模型与结果产物

### 3.2 首版不包含

1. 用户登录
2. 数据库持久化
3. 多用户历史记录
4. 真实医疗合规流程
5. CHARLS 外部验证展示

## 4. 总体方案

采用前后端分离：

1. 前端：React + Vite
2. 后端：FastAPI

前端负责：

1. 结果展示
2. 图表切换
3. 预置样本交互
4. 手动表单输入
5. 调用后端接口并渲染结果

后端负责：

1. 读取研究产物
2. 整理任务摘要
3. 对外提供统一接口
4. 进行单样本特征构造与模型推理

## 5. 页面结构

网页采用单页滚动结构，共六个区块：

1. Hero 区
   - 展示课题名称、研究目标、数据来源、当前完成状态

2. 研究流程区
   - 展示 NHANES 下载、interim、processed、训练、评估、导出流程

3. 结果总览区
   - 五张任务卡片：cardiovascular、diabetes、liver、cancer、multimorbidity_label
   - 显示最佳模型、AUC、Accuracy、Recall、F1

4. 图表分析区
   - 按任务切换显示模型对比图、ROC、混淆矩阵

5. 预置样本演示区
   - 内置若干典型样本卡片
   - 点击后展示五类预测结果

6. 手动预测区
   - 输入体检指标
   - 调用后端接口
   - 展示五类风险结果和简要说明

## 6. 前后端边界

### 6.1 前端职责

1. 获取项目概览与任务摘要
2. 获取指定任务的图表和结果
3. 展示预置样本
4. 提交手动输入到预测接口
5. 展示多任务预测结果

### 6.2 后端职责

1. 读取 `reports/tables` 与 `artifacts` 中的结果文件
2. 返回任务摘要
3. 返回任务级图表信息
4. 加载模型并执行五类任务预测
5. 提供预置样本接口

## 7. 接口设计

建议接口如下：

1. `GET /api/overview`
   - 返回项目概览、五类任务最佳模型摘要

2. `GET /api/tasks`
   - 返回所有任务的模型表现汇总

3. `GET /api/tasks/{task}/artifacts`
   - 返回指定任务的 ROC、混淆矩阵、模型对比图、最佳模型摘要

4. `GET /api/samples`
   - 返回预置样本列表

5. `POST /api/predict`
   - 接收单个样本输入
   - 返回五类任务预测概率与结果说明

## 8. 输入字段

手动输入区首版支持以下字段：

1. `ridageyr`
2. `bmxbmi`
3. `lbxglu`
4. `lbxtr`
5. `lbdhdd`
6. `lbdldl`
7. `lbxin`

后端内部负责构造：

1. `aip`
2. `tyg`
3. `tyg_bmi`
4. `glm7_score`

## 9. 技术结构

### 9.1 前端结构建议

1. `pages/HomePage`
2. `components/HeroSection`
3. `components/ResearchFlowSection`
4. `components/TaskOverviewSection`
5. `components/ChartsSection`
6. `components/SampleDemoSection`
7. `components/ManualPredictionSection`
8. `services/api.ts`

### 9.2 后端结构建议

1. `app/main.py`
2. `app/api/routes_overview.py`
3. `app/api/routes_tasks.py`
4. `app/api/routes_predict.py`
5. `app/services/overview_service.py`
6. `app/services/artifact_service.py`
7. `app/services/prediction_service.py`
8. `app/services/sample_service.py`
9. `app/schemas/*.py`

## 10. 错误处理

1. 图表文件缺失
   - 返回任务级错误，不影响整页

2. 模型文件缺失
   - 后端返回明确错误信息

3. 输入字段缺失
   - 前端校验并阻止提交

4. 输入值非法
   - 前端和后端双重校验

5. 单个任务推理失败
   - 接口允许部分成功，前端按任务显示失败原因

## 11. 测试方案

### 11.1 前端

1. 结果卡片渲染测试
2. 图表切换测试
3. 预置样本点击更新测试
4. 手动输入校验测试
5. API 调用成功与失败状态测试

### 11.2 后端

1. `/api/overview` 响应结构测试
2. `/api/tasks/{task}/artifacts` 响应测试
3. `/api/predict` 合法输入测试
4. `/api/predict` 非法输入测试
5. 模型缺失与图表缺失测试

## 12. 设计结论

网页子项目首版应构建为：

1. 一个面向答辩场景的单页展示系统
2. 基于 React + Vite 的前端
3. 基于 FastAPI 的后端
4. 复用研究主线当前已经产出的真实结果文件
5. 同时支持预置样本演示和手动输入预测

该方案最符合当前毕业设计推进节奏，也能最大化复用现有研究成果。
