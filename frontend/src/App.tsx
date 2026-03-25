import { startTransition, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

import {
  getOverview,
  getSamples,
  getTaskArtifacts,
  healthCheck,
  predictAll,
  type DemoSample,
  type OverviewResponse,
  type PredictRequest,
  type TaskArtifactsResponse,
} from './lib/api'
import { riskLabelZh, taskLabel } from './lib/labels'

type ViewKey = 'overview' | 'charts' | 'predict' | 'about'

type ToastState = {
  title: string
  message: string
}

type ModalState = {
  title: string
  src: string
}

type PredictionRun = {
  id: string
  at: number
  inputs: PredictRequest
  predictions: Record<string, { probability: number; risk_label: string }>
}

const DEFAULT_INPUTS: PredictRequest = {
  ridageyr: 56,
  bmxbmi: 24.8,
  lbxglu: 105.0,
  lbxtr: 160.0,
  lbdhdd: 45.0,
  lbdldl: 118.0,
  lbxin: 9.5,
}

function fmt3(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '-'
  return v.toFixed(3)
}

function fmtPct(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '-'
  return `${Math.round(v * 100)}%`
}

function riskTone(riskLabel: string): 'low' | 'medium' | 'high' | 'unknown' {
  if (riskLabel === 'low' || riskLabel === 'medium' || riskLabel === 'high') return riskLabel
  return 'unknown'
}

function iconPath(view: ViewKey): string {
  switch (view) {
    case 'overview':
      return 'M4 5h16v14H4z M7 8h10v2H7z M7 12h6v2H7z'
    case 'charts':
      return 'M4 19V5h2v12h14v2H4z M9 15V9h2v6H9z M13 15V7h2v8h-2z M17 15v-4h2v4h-2z'
    case 'predict':
      return 'M5 6h14v4H5z M5 12h7v6H5z M14 12h5v6h-5z'
    case 'about':
      return 'M12 2a10 10 0 1 0 0 20a10 10 0 0 0 0-20z M11 10h2v7h-2z M11 6h2v2h-2z'
  }
}

function validateInputs(inputs: PredictRequest): string[] {
  const issues: string[] = []
  const pairs: Array<[keyof PredictRequest, number, number]> = [
    ['ridageyr', 0, 120],
    ['bmxbmi', 10, 70],
    ['lbxglu', 40, 500],
    ['lbxtr', 20, 2000],
    ['lbdhdd', 5, 200],
    ['lbdldl', 10, 400],
    ['lbxin', 1, 300],
  ]

  for (const [k, min, max] of pairs) {
    const v = inputs[k]
    if (!Number.isFinite(v)) issues.push(`${k} 不是有效数字`)
    if (v < min || v > max) issues.push(`${k} 超出合理范围 (${min}-${max})`)
  }

  if (inputs.lbdhdd <= 0) issues.push('HDL 必须大于 0')
  return issues
}

function ArtifactFigure(props: {
  title: string
  url: string
  onOpen: (modal: ModalState) => void
}) {
  const [broken, setBroken] = useState(false)

  return (
    <div className="figure">
      <div className="figureTop">
        <div className="figureTitle">{props.title}</div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <a className="btn" href={props.url} target="_blank" rel="noreferrer">
            打开
          </a>
          <button
            type="button"
            className="btn"
            onClick={() => props.onOpen({ title: props.title, src: props.url })}
            disabled={broken}
            title={broken ? '图片文件可能不存在' : '点击放大'}
          >
            放大
          </button>
        </div>
      </div>
      <div className="figureBody">
        {broken ? (
          <div className="figurePlaceholder">
            未能加载图片。请确认已运行训练与导出流程，生成对应的 reports/tables 下的 PNG 文件。
            <div className="small mono" style={{ marginTop: 8 }}>
              {props.url}
            </div>
          </div>
        ) : (
          <img
            src={props.url}
            alt={props.title}
            loading="lazy"
            onClick={() => props.onOpen({ title: props.title, src: props.url })}
            onError={() => setBroken(true)}
          />
        )}
      </div>
    </div>
  )
}
export default function App() {
  const [view, setView] = useState<ViewKey>('overview')
  const [backendOk, setBackendOk] = useState<boolean | null>(null)

  const [overview, setOverview] = useState<OverviewResponse | null>(null)
  const [overviewLoading, setOverviewLoading] = useState<boolean>(true)

  const [samples, setSamples] = useState<DemoSample[]>([])
  const [samplesLoading, setSamplesLoading] = useState<boolean>(true)

  const [selectedTask, setSelectedTask] = useState<string>('')
  const [artifacts, setArtifacts] = useState<TaskArtifactsResponse | null>(null)
  const [artifactsLoading, setArtifactsLoading] = useState<boolean>(false)

  const [inputs, setInputs] = useState<PredictRequest>(DEFAULT_INPUTS)
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null)

  const [predicting, setPredicting] = useState<boolean>(false)
  const [predictions, setPredictions] = useState<Record<string, { probability: number; risk_label: string }> | null>(
    null,
  )

  const [toast, setToast] = useState<ToastState | null>(null)
  const [modal, setModal] = useState<ModalState | null>(null)

  const [chartTab, setChartTab] = useState<'roc' | 'confusion' | 'comparison' | 'csv'>('roc')

  const [history, setHistory] = useState<PredictionRun[]>([])

  const toastTimer = useRef<number | null>(null)

  function pushToast(t: ToastState) {
    setToast(t)
    if (toastTimer.current) window.clearTimeout(toastTimer.current)
    toastTimer.current = window.setTimeout(() => setToast(null), 6500)
  }

  async function refreshHealth() {
    try {
      await healthCheck()
      setBackendOk(true)
    } catch {
      setBackendOk(false)
    }
  }

  async function refreshOverview() {
    setOverviewLoading(true)
    try {
      const data = await getOverview()
      setOverview(data)
      if (!selectedTask && data.tasks.length > 0) setSelectedTask(data.tasks[0].task)
    } catch (err: unknown) {
      pushToast({ title: '概览加载失败', message: err instanceof Error ? err.message : String(err) })
    } finally {
      setOverviewLoading(false)
    }
  }

  async function refreshSamples() {
    setSamplesLoading(true)
    try {
      const data = await getSamples()
      setSamples(data.samples)
      if (!selectedSampleId && data.samples.length > 0) {
        setSelectedSampleId(data.samples[0].id)
        startTransition(() => setInputs(data.samples[0].inputs))
      }
    } catch (err: unknown) {
      pushToast({ title: '样例加载失败', message: err instanceof Error ? err.message : String(err) })
    } finally {
      setSamplesLoading(false)
    }
  }

  async function refreshArtifacts(task: string) {
    if (!task) return
    setArtifactsLoading(true)
    try {
      const data = await getTaskArtifacts(task)
      setArtifacts(data)
    } catch (err: unknown) {
      pushToast({ title: '图表加载失败', message: err instanceof Error ? err.message : String(err) })
    } finally {
      setArtifactsLoading(false)
    }
  }

  async function refreshAll() {
    await refreshHealth()
    await Promise.all([refreshOverview(), refreshSamples()])
    if (selectedTask) await refreshArtifacts(selectedTask)
  }

  useEffect(() => {
    void refreshAll()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    void refreshArtifacts(selectedTask)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTask])

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') setModal(null)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  const derived = useMemo(() => {
    const tyg = Math.log((inputs.lbxtr * inputs.lbxglu) / 2.0)
    return {
      aip: Math.log10(inputs.lbxtr / inputs.lbdhdd),
      tyg,
      tyg_bmi: tyg * inputs.bmxbmi,
    }
  }, [inputs.bmxbmi, inputs.lbdhdd, inputs.lbxglu, inputs.lbxtr])

  const issues = validateInputs(inputs)
  const canPredict = issues.length === 0 && backendOk === true && predicting === false

  async function runPredict() {
    if (!canPredict) return

    setPredicting(true)
    try {
      const result = await predictAll(inputs)
      setPredictions(result.predictions)

      const run: PredictionRun = {
        id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        at: Date.now(),
        inputs: { ...inputs },
        predictions: result.predictions,
      }
      setHistory((prev) => [run, ...prev].slice(0, 3))

      pushToast({ title: '预测完成', message: '结果已更新，可在右侧查看概率与风险等级。' })
    } catch (err: unknown) {
      pushToast({ title: '预测失败', message: err instanceof Error ? err.message : String(err) })
    } finally {
      setPredicting(false)
    }
  }

  function applySample(sample: DemoSample) {
    setSelectedSampleId(sample.id)
    startTransition(() => setInputs(sample.inputs))
    setPredictions(null)
    pushToast({ title: '已填充样例', message: `当前样例: ${sample.label}` })
  }

  function pageTitle(v: ViewKey): string {
    switch (v) {
      case 'overview':
        return '概览'
      case 'charts':
        return '图表'
      case 'predict':
        return '预测'
      case 'about':
        return '说明'
    }
  }

  const bestAuc = overview?.tasks.length
    ? Math.max(...overview.tasks.map((t) => t.auc ?? 0))
    : null

  const taskCount = overview?.tasks.length ?? null
  const taskOptions = overview?.tasks ?? []
  const selectedTaskOverview = overview?.tasks.find((t) => t.task === selectedTask) ?? null
  return (
    <div className="appShell">
      <aside className="sidebar">
        <div className="brand">
          <div className="logo" aria-hidden="true" />
          <div>
            <div className="brandTitle">慢性病风险建模</div>
            <div className="brandSub">答辩展示台</div>
          </div>
        </div>

        <div className="statusRow">
          <div className="pill" title="后端 API 状态">
            <span className={`dot ${backendOk === null ? '' : backendOk ? 'ok' : 'bad'}`} aria-hidden="true" />
            {backendOk === null ? '后端检测中' : backendOk ? '后端已连接' : '后端未连接'}
          </div>
          <div className="pill" title="当前任务">
            任务: <span style={{ color: 'var(--text)' }}>{selectedTask ? taskLabel(selectedTask) : '-'}</span>
          </div>
        </div>

        <nav className="nav" aria-label="侧边栏导航">
          {([
            { key: 'overview', hint: '任务与指标' },
            { key: 'charts', hint: 'ROC/混淆矩阵' },
            { key: 'predict', hint: '输入与推理' },
            { key: 'about', hint: '答辩讲解要点' },
          ] as const).map((item) => {
            const k = item.key as ViewKey
            const active = view === k
            return (
              <button
                key={k}
                type="button"
                className={`navItem ${active ? 'navItemActive' : ''}`}
                onClick={() => setView(k)}
              >
                <svg className="navIcon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d={iconPath(k)} />
                </svg>
                <span className="navMeta">
                  <span className="navLabel">{pageTitle(k)}</span>
                  <span className="navHint">{item.hint}</span>
                </span>
              </button>
            )
          })}
        </nav>

        <div className="sidebarFooter">
          <div className="small">一键启动: start.cmd prod</div>
          <div className="small">关闭: close.cmd</div>
          <div className="small" style={{ marginTop: 10 }}>
            提示: 图表来自 reports/tables 下的 PNG 与 CSV，请先运行训练与导出流程。
          </div>
        </div>
      </aside>

      <main className="main">
        <div className="mainInner">
        <header className="mainHeader">
          <div className="headerLeft">
            <div className="pageTitle">{pageTitle(view)}</div>
            <div className="pageSub">
              {view === 'overview'
                ? '快速浏览每个任务的最佳模型与核心指标。'
                : view === 'charts'
                  ? '切换任务后联动更新图表。点击图片可放大。'
                  : view === 'predict'
                    ? '选择样例或手工输入，点击预测查看风险。'
                    : '用于答辩讲解：数据、特征、模型与局限性。'}
            </div>
          </div>

          <div className="headerRight">
            <select
              className="select"
              value={selectedTask}
              onChange={(e) => setSelectedTask(e.target.value)}
              disabled={!taskOptions.length}
              aria-label="选择任务"
            >
              {taskOptions.map((t) => (
                <option key={t.task} value={t.task}>
                  {taskLabel(t.task)}
                </option>
              ))}
            </select>

            <button type="button" className="btn" onClick={() => void refreshAll()} title="重新拉取后端数据">
              刷新
            </button>

            <button
              type="button"
              className="btn btnPrimary"
              onClick={() => {
                setView('predict')
                void runPredict()
              }}
              disabled={!canPredict}
              title={backendOk !== true ? '后端未连接，无法预测' : issues.length ? '输入需要修正' : '一键预测'}
            >
              一键预测
            </button>
          </div>
        </header>
        {view === 'overview' ? (
          <div className="overviewLayout">
            <section className="card cardTall overviewTable" aria-busy={overviewLoading ? 'true' : 'false'}>
              <div className="cardHeader">
                <div>
                  <div className="cardTitle">任务概览</div>
                  <div className="cardHint">点击某一行可设置为当前任务，并联动图表与预测。</div>
                </div>
                <div className="pill">
                  AUC: <span style={{ color: 'var(--text)' }}>{fmt3(bestAuc)}</span>
                </div>
              </div>
              <div className="cardBody">
                {overviewLoading ? (
                  <div className="skeleton" style={{ height: 420 }} />
                ) : overview && overview.tasks.length ? (
                  <div className="tableWrap" role="region" aria-label="任务指标表">
                    <table>
                      <thead>
                        <tr>
                          <th>任务</th>
                          <th>最佳模型</th>
                          <th>AUC</th>
                          <th>Accuracy</th>
                          <th>Precision</th>
                          <th>Recall</th>
                          <th>F1</th>
                        </tr>
                      </thead>
                      <tbody>
                        {overview.tasks.map((t) => {
                          const active = t.task === selectedTask
                          return (
                            <tr
                              key={t.task}
                              className={active ? 'rowActive' : ''}
                              onClick={() => setSelectedTask(t.task)}
                              style={{ cursor: 'pointer' }}
                              title="点击选择任务"
                            >
                              <td>{taskLabel(t.task)}</td>
                              <td className="mono">{t.best_model_name}</td>
                              <td className="mono">{fmt3(t.auc)}</td>
                              <td className="mono">{fmt3(t.accuracy)}</td>
                              <td className="mono">{fmt3(t.precision)}</td>
                              <td className="mono">{fmt3(t.recall)}</td>
                              <td className="mono">{fmt3(t.f1)}</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="figurePlaceholder">没有可用任务数据。请确认后端可访问，并且 reports/tables 下的 CSV 已生成。</div>
                )}
              </div>
            </section>

            <section className="card cardTall overviewSide">
              <div className="cardHeader">
                <div>
                  <div className="cardTitle">关键指标</div>
                  <div className="cardHint">右侧提供摘要和跳转，让首屏更像应用。</div>
                </div>
              </div>
              <div className="cardBody">
                <div className="kpiRow">
                  <div className="kpi">
                    <div className="kpiLabel">建模任务数</div>
                    <div className="kpiValue">{taskCount ?? '-'}</div>
                  </div>
                  <div className="kpi">
                    <div className="kpiLabel">最佳 AUC</div>
                    <div className="kpiValue">{bestAuc === null ? '-' : bestAuc.toFixed(3)}</div>
                  </div>
                  <div className="kpi">
                    <div className="kpiLabel">特征维度</div>
                    <div className="kpiValue">5</div>
                  </div>
                </div>

                <div className="figurePlaceholder" style={{ marginTop: 12 }}>
                  <div style={{ fontWeight: 820 }}>当前任务摘要</div>
                  <div className="small" style={{ marginTop: 8 }}>
                    任务: <span className="mono">{selectedTask ? taskLabel(selectedTask) : '-'}</span>
                  </div>
                  <div className="small" style={{ marginTop: 6 }}>
                    最佳模型: <span className="mono">{selectedTaskOverview?.best_model_name ?? '-'}</span>
                  </div>
                  <div className="small" style={{ marginTop: 6 }}>
                    AUC/Acc/Recall: <span className="mono">{fmt3(selectedTaskOverview?.auc)}</span> /{' '}
                    <span className="mono">{fmt3(selectedTaskOverview?.accuracy)}</span> /{' '}
                    <span className="mono">{fmt3(selectedTaskOverview?.recall)}</span>
                  </div>

                  <div style={{ marginTop: 12, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                    <button type="button" className="btn" onClick={() => setView('charts')}>
                      看图表
                    </button>
                    <button type="button" className="btn" onClick={() => setView('predict')}>
                      去预测
                    </button>
                  </div>
                </div>

                <div className="figurePlaceholder" style={{ marginTop: 12 }}>
                  建议答辩讲解顺序: 数据来源(NHANES) / 特征(AIP, TyG) / 模型对比(AUC) / 在线推理演示。
                </div>
              </div>
            </section>

            <section className="card overviewPreview" aria-busy={artifactsLoading ? 'true' : 'false'}>
              <div className="cardHeader">
                <div>
                  <div className="cardTitle">图表预览</div>
                  <div className="cardHint">首屏直接看到图，让布局更饱满。点击图片可放大。</div>
                </div>
                <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                  <button type="button" className="btn" onClick={() => setView('charts')}>
                    进入图表页
                  </button>
                  {artifacts ? (
                    <a className="btn" href={artifacts.summary_csv_url} target="_blank" rel="noreferrer">
                      打开 CSV
                    </a>
                  ) : null}
                </div>
              </div>
              <div className="cardBody">
                {!selectedTask ? <div className="figurePlaceholder">请先选择任务。</div> : null}

                {selectedTask && artifactsLoading && !artifacts ? (
                  <div className="skeleton" style={{ height: 300 }} />
                ) : null}

                {selectedTask && artifacts ? (
  <div className="previewGrid">
    <div className="previewSlot previewRoc">
      <ArtifactFigure title="ROC 曲线(基线模型)" url={artifacts.roc_plot_url} onOpen={setModal} />
    </div>
    <div className="previewSlot previewConf">
      <ArtifactFigure title="混淆矩阵(基线模型)" url={artifacts.confusion_matrix_url} onOpen={setModal} />
    </div>
    <div className="previewSlot previewComp">
      <ArtifactFigure title="候选模型对比" url={artifacts.comparison_plot_url} onOpen={setModal} />
    </div>
  </div>
) : null}

                {selectedTask && !artifactsLoading && !artifacts ? (
                  <div className="figurePlaceholder">
                    暂无图表数据。请先运行训练与导出流程，生成 reports/tables 下的 PNG 与 CSV。
                  </div>
                ) : null}
              </div>
            </section>
          </div>
        ) : null}
        {view === 'charts' ? (
          <div className="grid">
            <section className="card" aria-busy={artifactsLoading ? 'true' : 'false'}>
              <div className="cardHeader">
                <div>
                  <div className="cardTitle">评估图表</div>
                  <div className="cardHint">选择任务后自动更新，支持放大查看细节。</div>
                </div>
                <div className="segmented" aria-label="图表类型">
                  <button
                    type="button"
                    className={`segBtn ${chartTab === 'roc' ? 'segBtnActive' : ''}`}
                    onClick={() => setChartTab('roc')}
                  >
                    ROC
                  </button>
                  <button
                    type="button"
                    className={`segBtn ${chartTab === 'confusion' ? 'segBtnActive' : ''}`}
                    onClick={() => setChartTab('confusion')}
                  >
                    混淆矩阵
                  </button>
                  <button
                    type="button"
                    className={`segBtn ${chartTab === 'comparison' ? 'segBtnActive' : ''}`}
                    onClick={() => setChartTab('comparison')}
                  >
                    对比
                  </button>
                  <button
                    type="button"
                    className={`segBtn ${chartTab === 'csv' ? 'segBtnActive' : ''}`}
                    onClick={() => setChartTab('csv')}
                  >
                    CSV
                  </button>
                </div>
              </div>
              <div className="cardBody">
                {artifactsLoading && !artifacts ? <div className="skeleton" style={{ height: 360 }} /> : null}

                {!selectedTask ? <div className="figurePlaceholder">请先选择任务。</div> : null}

                {selectedTask && artifacts ? (
                  <>
                    {chartTab === 'roc' ? (
                      <ArtifactFigure title="ROC 曲线(基线模型)" url={artifacts.roc_plot_url} onOpen={setModal} />
                    ) : null}
                    {chartTab === 'confusion' ? (
                      <ArtifactFigure
                        title="混淆矩阵(基线模型)"
                        url={artifacts.confusion_matrix_url}
                        onOpen={setModal}
                      />
                    ) : null}
                    {chartTab === 'comparison' ? (
                      <ArtifactFigure title="候选模型对比" url={artifacts.comparison_plot_url} onOpen={setModal} />
                    ) : null}
                    {chartTab === 'csv' ? (
                      <div className="figure">
                        <div className="figureTop">
                          <div className="figureTitle">候选模型汇总(CSV)</div>
                          <a className="btn" href={artifacts.summary_csv_url} target="_blank" rel="noreferrer">
                            下载/打开
                          </a>
                        </div>
                        <div className="figureBody">
                          <div className="figurePlaceholder">
                            点击右上角按钮打开 CSV。
                            <div className="small mono" style={{ marginTop: 8 }}>
                              {artifacts.summary_csv_url}
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : null}
                  </>
                ) : null}
              </div>
            </section>

            <section className="card">
              <div className="cardHeader">
                <div>
                  <div className="cardTitle">讲解提示</div>
                  <div className="cardHint">答辩现场常见追问，可以在此页快速回应。</div>
                </div>
              </div>
              <div className="cardBody">
                <div className="figurePlaceholder">
                  ROC 说明: 曲线越靠左上越好，AUC 越大表示整体判别能力越强。
                </div>
                <div className="figurePlaceholder" style={{ marginTop: 10 }}>
                  混淆矩阵说明: 可回答“误报/漏报”以及模型偏差。
                </div>
                <div className="figurePlaceholder" style={{ marginTop: 10 }}>
                  对比图说明: 用于展示不同候选模型在同一指标下的差异，支撑最终选型。
                </div>
              </div>
            </section>
          </div>
        ) : null}

        {view === 'predict' ? (
          <div className="grid">
            <section className="card">
              <div className="cardHeader">
                <div>
                  <div className="cardTitle">输入</div>
                  <div className="cardHint">支持样例一键填充与手工输入，输入会传给后端推理服务。</div>
                </div>
                <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                  <button
                    type="button"
                    className="btn"
                    onClick={() => {
                      setInputs(DEFAULT_INPUTS)
                      setSelectedSampleId(null)
                      setPredictions(null)
                      pushToast({ title: '已重置', message: '输入已恢复到默认值。' })
                    }}
                    disabled={predicting}
                  >
                    重置
                  </button>
                  <button
                    type="button"
                    className="btn btnPrimary"
                    onClick={() => void runPredict()}
                    disabled={!canPredict}
                    title={backendOk !== true ? '后端未连接' : issues.length ? '请修正输入' : '开始预测'}
                  >
                    {predicting ? '预测中...' : '预测'}
                  </button>
                </div>
              </div>
              <div className="cardBody">
                {backendOk === false ? (
                  <div className="figurePlaceholder" style={{ borderColor: 'rgba(255, 93, 115, 0.22)' }}>
                    后端未连接。请确认 start.cmd prod 已启动，或手动运行:
                    <div className="small mono" style={{ marginTop: 8 }}>
                      uvicorn backend.app.main:create_app --factory --reload
                    </div>
                  </div>
                ) : null}

                <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 10 }}>
                  {samplesLoading ? (
                    <div className="skeleton" style={{ height: 36, width: 220 }} />
                  ) : (
                    samples.map((s) => (
                      <button
                        key={s.id}
                        type="button"
                        className={`btn ${selectedSampleId === s.id ? 'btnPrimary' : ''}`}
                        onClick={() => applySample(s)}
                      >
                        {s.label}
                      </button>
                    ))
                  )}
                </div>

                <form
                  className="form"
                  onSubmit={(e) => {
                    e.preventDefault()
                    void runPredict()
                  }}
                >
                  <div className="field">
                    <div className="label">
                      年龄 <span>岁</span>
                    </div>
                    <input
                      className="input"
                      type="number"
                      value={inputs.ridageyr}
                      min={0}
                      max={120}
                      step={1}
                      onChange={(e) => setInputs((p) => ({ ...p, ridageyr: Number(e.target.value) }))}
                    />
                  </div>

                  <div className="field">
                    <div className="label">
                      BMI <span>kg/m2</span>
                    </div>
                    <input
                      className="input"
                      type="number"
                      value={inputs.bmxbmi}
                      min={10}
                      max={70}
                      step={0.1}
                      onChange={(e) => setInputs((p) => ({ ...p, bmxbmi: Number(e.target.value) }))}
                    />
                  </div>

                  <div className="field">
                    <div className="label">
                      空腹血糖 <span>mg/dL</span>
                    </div>
                    <input
                      className="input"
                      type="number"
                      value={inputs.lbxglu}
                      min={40}
                      max={500}
                      step={0.1}
                      onChange={(e) => setInputs((p) => ({ ...p, lbxglu: Number(e.target.value) }))}
                    />
                  </div>

                  <div className="field">
                    <div className="label">
                      甘油三酯 <span>mg/dL</span>
                    </div>
                    <input
                      className="input"
                      type="number"
                      value={inputs.lbxtr}
                      min={20}
                      max={2000}
                      step={0.1}
                      onChange={(e) => setInputs((p) => ({ ...p, lbxtr: Number(e.target.value) }))}
                    />
                  </div>

                  <div className="field">
                    <div className="label">
                      HDL-C <span>mg/dL</span>
                    </div>
                    <input
                      className="input"
                      type="number"
                      value={inputs.lbdhdd}
                      min={5}
                      max={200}
                      step={0.1}
                      onChange={(e) => setInputs((p) => ({ ...p, lbdhdd: Number(e.target.value) }))}
                    />
                  </div>

                  <div className="field">
                    <div className="label">
                      LDL-C <span>mg/dL</span>
                    </div>
                    <input
                      className="input"
                      type="number"
                      value={inputs.lbdldl}
                      min={10}
                      max={400}
                      step={0.1}
                      onChange={(e) => setInputs((p) => ({ ...p, lbdldl: Number(e.target.value) }))}
                    />
                  </div>

                  <div className="field" style={{ gridColumn: '1 / -1' }}>
                    <div className="label">
                      胰岛素 <span>uU/mL</span>
                    </div>
                    <input
                      className="input"
                      type="number"
                      value={inputs.lbxin}
                      min={1}
                      max={300}
                      step={0.1}
                      onChange={(e) => setInputs((p) => ({ ...p, lbxin: Number(e.target.value) }))}
                    />
                  </div>
                </form>

                <div className="helper" aria-label="派生特征预览">
                  <div className="helperItem">
                    <div className="helperK">AIP = log10(TG/HDL)</div>
                    <div className="helperV">{fmt3(derived.aip)}</div>
                  </div>
                  <div className="helperItem">
                    <div className="helperK">TyG = ln(TG*GLU/2)</div>
                    <div className="helperV">{fmt3(derived.tyg)}</div>
                  </div>
                  <div className="helperItem">
                    <div className="helperK">TyG-BMI</div>
                    <div className="helperV">{fmt3(derived.tyg_bmi)}</div>
                  </div>
                </div>

                {issues.length ? (
                  <div className="figurePlaceholder" style={{ marginTop: 10, borderColor: 'rgba(255, 180, 84, 0.22)' }}>
                    输入提示:
                    <div className="small" style={{ marginTop: 6 }}>
                      {issues.slice(0, 3).join('；')}
                      {issues.length > 3 ? '...' : ''}
                    </div>
                  </div>
                ) : null}

                {history.length ? (
                  <div style={{ marginTop: 12 }}>
                    <div className="small">最近预测(可点击回看):</div>
                    <div style={{ marginTop: 8, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      {history.map((h) => (
                        <button
                          key={h.id}
                          type="button"
                          className="btn"
                          onClick={() => {
                            setInputs(h.inputs)
                            setPredictions(h.predictions)
                            pushToast({ title: '已回看', message: new Date(h.at).toLocaleString() })
                          }}
                        >
                          {new Date(h.at).toLocaleTimeString()}
                        </button>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>
            </section>

            <section className="card">
              <div className="cardHeader">
                <div>
                  <div className="cardTitle">结果</div>
                  <div className="cardHint">概率为 0-1，风险等级由后端阈值划分。</div>
                </div>
              </div>
              <div className="cardBody">
                {predictions ? (
                  <div className="resultList" aria-label="预测结果">
                    {Object.entries(predictions)
                      .sort(([a], [b]) => a.localeCompare(b))
                      .map(([task, pred]) => {
                        const tone = riskTone(pred.risk_label)
                        const p = pred.probability
                        return (
                          <div className="resultRow" key={task}>
                            <div>
                              <div className="resultTask">{taskLabel(task)}</div>
                              <div className="resultMeta">
                                概率: <span className="mono">{fmt3(p)}</span> ({fmtPct(p)})
                              </div>
                            </div>
                            <div className="bar" aria-label="概率条">
                              <div className="barFill" style={{ width: `${Math.max(0, Math.min(1, p)) * 100}%` }} />
                            </div>
                            <div className={`badge ${tone === 'unknown' ? '' : tone}`}>{riskLabelZh(pred.risk_label)}</div>
                          </div>
                        )
                      })}
                  </div>
                ) : (
                  <div className="figurePlaceholder">还没有结果。选择样例或填写输入后点击“预测”。</div>
                )}
              </div>
            </section>
          </div>
        ) : null}
        {view === 'about' ? (
          <div className="grid">
            <section className="card">
              <div className="cardHeader">
                <div>
                  <div className="cardTitle">答辩讲解要点</div>
                  <div className="cardHint">这页用来背稿，保证现场节奏稳定。</div>
                </div>
              </div>
              <div className="cardBody">
                <div className="figurePlaceholder">
                  1) 数据: NHANES 公共健康调查数据，包含年龄、BMI、血脂、血糖等指标。
                </div>
                <div className="figurePlaceholder" style={{ marginTop: 10 }}>
                  2) 特征: AIP/TyG/TyG-BMI 等组合指标，增强对代谢异常的刻画。
                </div>
                <div className="figurePlaceholder" style={{ marginTop: 10 }}>
                  3) 模型: 多任务分别训练并比较候选模型，最终按 AUC 选择最佳模型。
                </div>
                <div className="figurePlaceholder" style={{ marginTop: 10 }}>
                  4) 结果: 展示 ROC、混淆矩阵与对比图，并用在线预测演示输入变化带来的风险变化。
                </div>
                <div className="figurePlaceholder" style={{ marginTop: 10 }}>
                  5) 局限: 样本偏倚、测量误差、阈值选择与可解释性等问题，后续可引入更多特征和模型解释。
                </div>
              </div>
            </section>

            <section className="card">
              <div className="cardHeader">
                <div>
                  <div className="cardTitle">运行方式</div>
                  <div className="cardHint">答辩现场建议使用单服务模式。</div>
                </div>
              </div>
              <div className="cardBody">
                <div className="figurePlaceholder">
                  推荐: <code>start.cmd prod</code>
                  <div className="small" style={{ marginTop: 8 }}>
                    该模式会构建前端并启动后端，浏览器访问 <span className="mono">http://127.0.0.1:8000/</span>
                  </div>
                </div>
                <div className="figurePlaceholder" style={{ marginTop: 10 }}>
                  关闭: <code>close.cmd</code>
                  <div className="small" style={{ marginTop: 8 }}>
                    脚本会按 .run/pids.env 记录的 PID 结束进程树。
                  </div>
                </div>
              </div>
            </section>
          </div>
        ) : null}

        <div style={{ marginTop: 14 }} className="small">
          © {new Date().getFullYear()} Graduation Project Demo
        </div>
        </div>
      </main>

      {toast ? (
        <div className="toast" role="status" aria-live="polite">
          <div className="toastTitle">{toast.title}</div>
          <div className="toastMsg">{toast.message}</div>
          <div style={{ marginTop: 10, display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
            <button type="button" className="btn" onClick={() => setToast(null)}>
              关闭
            </button>
          </div>
        </div>
      ) : null}

      {modal ? (
        <div
          className="modalOverlay"
          role="dialog"
          aria-modal="true"
          aria-label="图表放大预览"
          onClick={() => setModal(null)}
        >
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modalTop">
              <div className="modalTitle">{modal.title}</div>
              <button type="button" className="btn" onClick={() => setModal(null)}>
                关闭
              </button>
            </div>
            <div className="modalBody">
              <img src={modal.src} alt={modal.title} />
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
