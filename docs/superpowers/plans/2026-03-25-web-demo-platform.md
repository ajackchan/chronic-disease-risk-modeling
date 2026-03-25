# Web Demo Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a React + Vite frontend and FastAPI backend that present the current NHANES research results, expose five-task prediction APIs, and support both preset-sample and manual-input demo flows for thesis defense.

**Architecture:** Keep the current research pipeline in `src/` as the source of truth for features, labels, trained models, and result files. Add a thin `backend/` service that reads existing CSV/PNG/joblib artifacts and performs single-sample inference, plus a `frontend/` SPA that renders the dashboard and talks only to the backend APIs.

**Tech Stack:** React 18, Vite, TypeScript, FastAPI, Pydantic, Uvicorn, pytest, Vitest, Testing Library, joblib, pandas

---

## Planned File Structure

- Modify: `requirements.txt`
- Create: `backend/app/main.py`
- Create: `backend/app/config.py`
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/routes_health.py`
- Create: `backend/app/api/routes_overview.py`
- Create: `backend/app/api/routes_tasks.py`
- Create: `backend/app/api/routes_predict.py`
- Create: `backend/app/services/overview_service.py`
- Create: `backend/app/services/artifact_service.py`
- Create: `backend/app/services/sample_service.py`
- Create: `backend/app/services/prediction_service.py`
- Create: `backend/app/schemas/overview.py`
- Create: `backend/app/schemas/task.py`
- Create: `backend/app/schemas/predict.py`
- Create: `backend/tests/test_health_api.py`
- Create: `backend/tests/test_overview_api.py`
- Create: `backend/tests/test_task_artifacts_api.py`
- Create: `backend/tests/test_predict_api.py`
- Create: `backend/tests/test_sample_api.py`
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/styles.css`
- Create: `frontend/src/lib/api.ts`
- Create: `frontend/src/types.ts`
- Create: `frontend/src/components/HeroSection.tsx`
- Create: `frontend/src/components/ResearchFlowSection.tsx`
- Create: `frontend/src/components/TaskOverviewSection.tsx`
- Create: `frontend/src/components/ChartsSection.tsx`
- Create: `frontend/src/components/SampleDemoSection.tsx`
- Create: `frontend/src/components/ManualPredictionSection.tsx`
- Create: `frontend/src/components/PredictionResults.tsx`
- Create: `frontend/src/components/TaskCard.tsx`
- Create: `frontend/src/components/ChartTabs.tsx`
- Create: `frontend/src/components/MetricBadge.tsx`
- Create: `frontend/src/components/LoadingBlock.tsx`
- Create: `frontend/src/components/ErrorBlock.tsx`
- Create: `frontend/src/__tests__/App.test.tsx`
- Create: `frontend/src/__tests__/TaskOverviewSection.test.tsx`
- Create: `frontend/src/__tests__/ManualPredictionSection.test.tsx`
- Create: `frontend/src/__tests__/SampleDemoSection.test.tsx`
- Create: `frontend/src/__tests__/ChartsSection.test.tsx`
- Modify: `README.md`

### Task 1: Bootstrap FastAPI Backend And Shared Python Configuration

**Files:**
- Modify: `requirements.txt`
- Create: `backend/app/main.py`
- Create: `backend/app/config.py`
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/routes_health.py`
- Create: `backend/tests/test_health_api.py`

- [ ] **Step 1: Write the failing backend health API test**

```python
from fastapi.testclient import TestClient

from backend.app.main import create_app


def test_health_endpoint_returns_ok():
    client = TestClient(create_app())

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/tests/test_health_api.py -q`
Expected: FAIL with `ModuleNotFoundError` for `backend.app.main`

- [ ] **Step 3: Add backend runtime dependencies**

```text
fastapi>=0.115
uvicorn>=0.30
pydantic>=2.7
```

- [ ] **Step 4: Implement the minimal FastAPI app**

```python
from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="Chronic Disease Risk Demo API")
    app.include_router(health_router, prefix="/api")
    return app
```

```python
from fastapi import APIRouter

health_router = APIRouter()


@health_router.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 5: Run the backend health test**

Run: `python -m pytest backend/tests/test_health_api.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt backend/app backend/tests/test_health_api.py
git commit -m "功能：初始化网页演示后端骨架"
```

### Task 2: Implement Overview And Artifact APIs Against Existing Result Files

**Files:**
- Create: `backend/app/services/overview_service.py`
- Create: `backend/app/services/artifact_service.py`
- Create: `backend/app/schemas/overview.py`
- Create: `backend/app/schemas/task.py`
- Create: `backend/app/api/routes_overview.py`
- Create: `backend/app/api/routes_tasks.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_overview_api.py`
- Create: `backend/tests/test_task_artifacts_api.py`

- [ ] **Step 1: Write the failing overview API test**

```python
def test_overview_endpoint_returns_five_task_cards(client):
    response = client.get("/api/overview")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["tasks"]) == 5
    assert payload["tasks"][0]["task"]
    assert "best_model_name" in payload["tasks"][0]
```

- [ ] **Step 2: Write the failing task artifacts API test**

```python
def test_task_artifacts_endpoint_returns_chart_urls(client):
    response = client.get("/api/tasks/diabetes/artifacts")

    assert response.status_code == 200
    payload = response.json()
    assert payload["task"] == "diabetes"
    assert payload["roc_plot_url"].endswith("baseline_diabetes_roc.png")
    assert payload["comparison_plot_url"].endswith("candidate_diabetes_comparison.png")
```

- [ ] **Step 3: Run the API tests to verify they fail**

Run: `python -m pytest backend/tests/test_overview_api.py backend/tests/test_task_artifacts_api.py -q`
Expected: FAIL with missing services/routes

- [ ] **Step 4: Implement overview and artifact services**

```python
def load_task_overview(task_name: str, reports_dir: Path) -> dict:
    summary = pd.read_csv(reports_dir / f"candidate_{task_name}_summary.csv")
    best_row = summary.sort_values("auc", ascending=False).iloc[0]
    return {
        "task": task_name,
        "best_model_name": best_row["model_name"],
        "auc": float(best_row["auc"]),
        "accuracy": float(best_row["accuracy"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "f1": float(best_row["f1"]),
    }
```

```python
def build_task_artifact_payload(task_name: str, reports_dir: Path) -> dict:
    return {
        "task": task_name,
        "roc_plot_url": f"/static/reports/tables/baseline_{task_name}_roc.png",
        "confusion_matrix_url": f"/static/reports/tables/baseline_{task_name}_confusion.png",
        "comparison_plot_url": f"/static/reports/tables/candidate_{task_name}_comparison.png",
    }
```

- [ ] **Step 5: Mount static file serving and register routes**

```python
app.mount("/static", StaticFiles(directory=repo_root), name="static")
```

- [ ] **Step 6: Run the overview and task artifact API tests**

Run: `python -m pytest backend/tests/test_overview_api.py backend/tests/test_task_artifacts_api.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backend/app backend/tests/test_overview_api.py backend/tests/test_task_artifacts_api.py
git commit -m "功能：接入结果概览与图表接口"
```

### Task 3: Implement Preset Samples And Manual Prediction API

**Files:**
- Create: `backend/app/services/sample_service.py`
- Create: `backend/app/services/prediction_service.py`
- Create: `backend/app/schemas/predict.py`
- Create: `backend/app/api/routes_predict.py`
- Create: `backend/tests/test_sample_api.py`
- Create: `backend/tests/test_predict_api.py`

- [ ] **Step 1: Write the failing preset sample API test**

```python
def test_samples_endpoint_returns_demo_cases(client):
    response = client.get("/api/samples")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["samples"]) >= 3
    assert "ridageyr" in payload["samples"][0]["inputs"]
```

- [ ] **Step 2: Write the failing manual predict API test**

```python
def test_predict_endpoint_returns_five_task_probabilities(client):
    response = client.post(
        "/api/predict",
        json={
            "ridageyr": 56,
            "bmxbmi": 24.5,
            "lbxglu": 105.0,
            "lbxtr": 160.0,
            "lbdhdd": 45.0,
            "lbdldl": 118.0,
            "lbxin": 9.5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload["predictions"]) == {
        "cardiovascular",
        "diabetes",
        "liver",
        "cancer",
        "multimorbidity_label",
    }
```

- [ ] **Step 3: Run the prediction API tests to verify they fail**

Run: `python -m pytest backend/tests/test_sample_api.py backend/tests/test_predict_api.py -q`
Expected: FAIL with missing sample/predict services

- [ ] **Step 4: Implement preset sample service**

```python
def get_demo_samples() -> list[dict]:
    return [
        {"id": "sample-low", "label": "低风险样本", "inputs": {...}},
        {"id": "sample-mid", "label": "中风险样本", "inputs": {...}},
        {"id": "sample-high", "label": "高风险样本", "inputs": {...}},
    ]
```

- [ ] **Step 5: Implement prediction service by reusing existing feature logic and joblib models**

```python
def predict_all_tasks(payload: PredictRequest, model_dir: Path) -> dict[str, dict]:
    feature_row = build_feature_row(payload)
    predictions = {}
    for task_name in TASK_NAMES:
        model = joblib.load(model_dir / f"candidate_best_{task_name}.joblib")
        probability = float(model.predict_proba(feature_row)[0, 1])
        predictions[task_name] = {"probability": probability}
    return predictions
```

- [ ] **Step 6: Run the sample and predict API tests**

Run: `python -m pytest backend/tests/test_sample_api.py backend/tests/test_predict_api.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backend/app backend/tests/test_sample_api.py backend/tests/test_predict_api.py
git commit -m "功能：接入预置样本与手动预测接口"
```

### Task 4: Scaffold The React + Vite Frontend

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/styles.css`
- Create: `frontend/src/types.ts`
- Create: `frontend/src/lib/api.ts`
- Create: `frontend/src/__tests__/App.test.tsx`

- [ ] **Step 1: Create the frontend scaffold**

Run: `npm create vite@latest frontend -- --template react-ts`
Expected: Vite React TypeScript scaffold created under `frontend/`

- [ ] **Step 2: Install frontend runtime and test dependencies**

Run: `npm --prefix frontend install`
Expected: frontend dependencies installed

Run: `npm --prefix frontend install -D vitest jsdom @testing-library/react @testing-library/jest-dom`
Expected: test dependencies installed

- [ ] **Step 3: Write the failing frontend shell test**

```tsx
import { render, screen } from "@testing-library/react";
import App from "../App";

test("renders thesis demo title", () => {
  render(<App />);
  expect(screen.getByText(/慢性病风险建模答辩展示台/i)).toBeInTheDocument();
});
```

- [ ] **Step 4: Run the frontend shell test to verify it fails**

Run: `npm --prefix frontend test -- --run src/__tests__/App.test.tsx`
Expected: FAIL because the scaffold still renders the default Vite screen

- [ ] **Step 5: Replace the scaffold with the dashboard shell**

```tsx
export default function App() {
  return (
    <main>
      <h1>慢性病风险建模答辩展示台</h1>
      <p>研究成果展示与交互预测系统</p>
    </main>
  );
}
```

- [ ] **Step 6: Run the frontend shell test**

Run: `npm --prefix frontend test -- --run src/__tests__/App.test.tsx`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add frontend
git commit -m "功能：初始化网页演示前端骨架"
```

### Task 5: Implement Overview Cards And Chart Navigation In The Frontend

**Files:**
- Create: `frontend/src/components/HeroSection.tsx`
- Create: `frontend/src/components/ResearchFlowSection.tsx`
- Create: `frontend/src/components/TaskOverviewSection.tsx`
- Create: `frontend/src/components/ChartsSection.tsx`
- Create: `frontend/src/components/TaskCard.tsx`
- Create: `frontend/src/components/ChartTabs.tsx`
- Create: `frontend/src/components/MetricBadge.tsx`
- Create: `frontend/src/components/LoadingBlock.tsx`
- Create: `frontend/src/components/ErrorBlock.tsx`
- Create: `frontend/src/__tests__/TaskOverviewSection.test.tsx`
- Create: `frontend/src/__tests__/ChartsSection.test.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/lib/api.ts`

- [ ] **Step 1: Write the failing task overview component test**

```tsx
test("renders five task cards from API data", async () => {
  render(<TaskOverviewSection tasks={mockTasks} />);
  expect(screen.getAllByRole("article")).toHaveLength(5);
});
```

- [ ] **Step 2: Write the failing charts section test**

```tsx
test("switches chart view when a task is selected", async () => {
  render(<ChartsSection tasks={mockTasks} />);
  expect(screen.getByText(/ROC/i)).toBeInTheDocument();
});
```

- [ ] **Step 3: Run the overview and charts tests to verify they fail**

Run: `npm --prefix frontend test -- --run src/__tests__/TaskOverviewSection.test.tsx src/__tests__/ChartsSection.test.tsx`
Expected: FAIL with missing components

- [ ] **Step 4: Implement the overview cards and chart tabs**

```tsx
export function TaskCard({ task }: { task: TaskOverview }) {
  return (
    <article>
      <h3>{task.task}</h3>
      <p>{task.bestModelName}</p>
      <span>{task.auc.toFixed(3)}</span>
    </article>
  );
}
```

- [ ] **Step 5: Wire the app to fetch `/api/overview` and `/api/tasks/{task}/artifacts`**

```ts
export async function fetchOverview(): Promise<OverviewResponse> {
  const response = await fetch("/api/overview");
  return response.json();
}
```

- [ ] **Step 6: Run the overview and charts tests**

Run: `npm --prefix frontend test -- --run src/__tests__/TaskOverviewSection.test.tsx src/__tests__/ChartsSection.test.tsx`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add frontend/src
git commit -m "功能：完成结果总览与图表切换界面"
```

### Task 6: Implement Preset Sample Demo And Manual Prediction UI

**Files:**
- Create: `frontend/src/components/SampleDemoSection.tsx`
- Create: `frontend/src/components/ManualPredictionSection.tsx`
- Create: `frontend/src/components/PredictionResults.tsx`
- Create: `frontend/src/__tests__/SampleDemoSection.test.tsx`
- Create: `frontend/src/__tests__/ManualPredictionSection.test.tsx`
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Write the failing preset sample interaction test**

```tsx
test("clicking a sample card shows prediction results", async () => {
  render(<SampleDemoSection samples={mockSamples} onSelect={mockOnSelect} />);
  await user.click(screen.getByText(/低风险样本/i));
  expect(mockOnSelect).toHaveBeenCalled();
});
```

- [ ] **Step 2: Write the failing manual prediction form test**

```tsx
test("submits manual prediction inputs to the API", async () => {
  render(<ManualPredictionSection onSubmit={mockSubmit} />);
  await user.type(screen.getByLabelText(/年龄/i), "56");
  await user.click(screen.getByRole("button", { name: /开始预测/i }));
  expect(mockSubmit).toHaveBeenCalled();
});
```

- [ ] **Step 3: Run the sample and manual prediction tests to verify they fail**

Run: `npm --prefix frontend test -- --run src/__tests__/SampleDemoSection.test.tsx src/__tests__/ManualPredictionSection.test.tsx`
Expected: FAIL with missing components

- [ ] **Step 4: Implement the sample demo and manual form components**

```tsx
const fields = ["ridageyr", "bmxbmi", "lbxglu", "lbxtr", "lbdhdd", "lbdldl", "lbxin"];
```

- [ ] **Step 5: Wire the frontend to `/api/samples` and `/api/predict`**

```ts
export async function predict(payload: PredictRequest): Promise<PredictResponse> {
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return response.json();
}
```

- [ ] **Step 6: Run the sample and manual prediction tests**

Run: `npm --prefix frontend test -- --run src/__tests__/SampleDemoSection.test.tsx src/__tests__/ManualPredictionSection.test.tsx`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add frontend/src
git commit -m "功能：完成预置样本与手动预测交互"
```

### Task 7: Finish API Integration, Styling, And End-To-End Verification

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/styles.css`
- Modify: `frontend/vite.config.ts`
- Modify: `backend/app/main.py`
- Modify: `README.md`

- [ ] **Step 1: Add a failing integration check for the frontend build**

Run: `npm --prefix frontend run build`
Expected: FAIL until API types and imports are fully wired

- [ ] **Step 2: Add Vite proxy and backend CORS**

```ts
server: {
  proxy: {
    "/api": "http://127.0.0.1:8000",
    "/static": "http://127.0.0.1:8000",
  },
}
```

- [ ] **Step 3: Add loading, empty, and error states for all API-backed sections**

```tsx
if (isLoading) return <LoadingBlock label="正在加载模型结果..." />;
if (error) return <ErrorBlock message={error.message} />;
```

- [ ] **Step 4: Verify frontend build**

Run: `npm --prefix frontend run build`
Expected: PASS

- [ ] **Step 5: Verify backend routes**

Run: `python -m pytest backend/tests -q`
Expected: PASS

- [ ] **Step 6: Update README with exact local startup steps**

```markdown
1. `python -m uvicorn backend.app.main:create_app --factory --reload`
2. `npm --prefix frontend run dev`
3. Open the Vite URL and verify `/api/overview` loads the five tasks
```

- [ ] **Step 7: Commit**

```bash
git add frontend backend README.md
git commit -m "功能：完成网页演示集成与运行说明"
```

### Task 8: Final Real-Data Verification For The Demo Stack

**Files:**
- Modify: `README.md`
- Create: `backend/tests/test_real_artifacts_smoke.py`

- [ ] **Step 1: Write the failing real-artifact smoke test**

```python
def test_existing_reports_support_demo_api():
    payload = load_task_overview("diabetes", Path("reports/tables"))
    assert payload["auc"] > 0.5
```

- [ ] **Step 2: Run the smoke test to verify it fails if paths or formats drift**

Run: `python -m pytest backend/tests/test_real_artifacts_smoke.py -q`
Expected: FAIL until the smoke utility is added

- [ ] **Step 3: Implement the smoke utility and document the full demo walkthrough**

```markdown
1. Run `python scripts/train_candidates.py`
2. Run `python scripts/export_artifacts.py`
3. Start backend
4. Start frontend
5. Click a preset sample and verify five-task output
6. Submit manual values and verify response
```

- [ ] **Step 4: Run the full Python and frontend test suites**

Run: `python -m pytest -q`
Expected: PASS

Run: `npm --prefix frontend test -- --run`
Expected: PASS

- [ ] **Step 5: Run backend locally**

Run: `python -m uvicorn backend.app.main:create_app --factory --reload`
Expected: FastAPI starts at `http://127.0.0.1:8000`

- [ ] **Step 6: Run frontend locally**

Run: `npm --prefix frontend run dev`
Expected: Vite starts and the page loads with five task cards

- [ ] **Step 7: Commit**

```bash
git add backend/tests README.md
git commit -m "功能：完成网页演示最终验收"
```

## Notes For Execution

- Do not move the current research result files out of `reports/tables` and `artifacts` in the first web iteration.
- Prediction service must reuse the existing feature logic in `src/chronic_disease_risk/features` and not re-implement formulas differently in the backend.
- Frontend 首版只做单页，不引入路由复杂度。
- All backend paths should resolve relative to the repo root so local execution and tests behave identically.
- If `npm create vite` or `npm install` requires network approval, request it before running those commands.
