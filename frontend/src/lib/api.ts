export type TaskOverview = {
  task: string
  best_model_name: string
  auc: number
  accuracy: number
  precision: number
  recall: number
  f1: number
}

export type OverviewResponse = {
  title: string
  tasks: TaskOverview[]
}

export type TaskArtifactsResponse = {
  task: string
  roc_plot_url: string
  confusion_matrix_url: string
  comparison_plot_url: string
  summary_csv_url: string
}

export type PredictRequest = {
  ridageyr: number
  bmxbmi: number
  lbxglu: number
  lbxtr: number
  lbdhdd: number
  lbdldl: number
  lbxin: number
}

export type DemoSample = {
  id: string
  label: string
  inputs: PredictRequest
}

export type SamplesResponse = {
  samples: DemoSample[]
}

export type TaskPrediction = {
  probability: number
  risk_label: string
}

export type PredictResponse = {
  predictions: Record<string, TaskPrediction>
}

type JsonValue = null | boolean | number | string | JsonValue[] | { [k: string]: JsonValue }

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  })

  const text = await resp.text()
  const data = (text ? (JSON.parse(text) as JsonValue) : null) as unknown

  if (!resp.ok) {
    const msg =
      typeof data === 'object' && data && 'detail' in (data as any)
        ? String((data as any).detail)
        : text
    throw new Error(msg || `HTTP ${resp.status}`)
  }

  return data as T
}

export async function healthCheck(): Promise<{ status: string }> {
  return fetchJson('/api/health')
}

export async function getOverview(): Promise<OverviewResponse> {
  return fetchJson('/api/overview')
}

export async function getSamples(): Promise<SamplesResponse> {
  return fetchJson('/api/samples')
}

export async function getTaskArtifacts(
  taskName: string,
): Promise<TaskArtifactsResponse> {
  return fetchJson(`/api/tasks/${encodeURIComponent(taskName)}/artifacts`)
}

export async function predictAll(payload: PredictRequest): Promise<PredictResponse> {
  return fetchJson('/api/predict', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}