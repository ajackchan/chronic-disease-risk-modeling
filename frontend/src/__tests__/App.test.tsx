import { render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import App from '../App'

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('App', () => {
  it('renders overview and loads tasks', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : (input as any).url ?? String(input)

      if (url.endsWith('/api/health')) {
        return jsonResponse({ status: 'ok' })
      }

      if (url.endsWith('/api/overview')) {
        return jsonResponse({
          title: '慢性病风险建模答辩展示台',
          tasks: [
            {
              task: 'diabetes',
              best_model_name: 'LogisticRegression',
              auc: 0.8123,
              accuracy: 0.7001,
              precision: 0.6123,
              recall: 0.5345,
              f1: 0.5702,
            },
          ],
        })
      }

      if (url.endsWith('/api/samples')) {
        return jsonResponse({
          samples: [
            {
              id: 'sample-low',
              label: '低风险样本',
              inputs: {
                ridageyr: 42,
                bmxbmi: 22.3,
                lbxglu: 92.0,
                lbxtr: 95.0,
                lbdhdd: 58.0,
                lbdldl: 102.0,
                lbxin: 6.5,
              },
            },
          ],
        })
      }

      if (url.includes('/api/tasks/') && url.endsWith('/artifacts')) {
        return jsonResponse({
          task: 'diabetes',
          roc_plot_url: '/static/reports/tables/baseline_diabetes_roc.png',
          confusion_matrix_url: '/static/reports/tables/baseline_diabetes_confusion.png',
          comparison_plot_url: '/static/reports/tables/candidate_diabetes_comparison.png',
          summary_csv_url: '/static/reports/tables/candidate_diabetes_summary.csv',
        })
      }

      return jsonResponse({ detail: 'not found' }, 404)
    })

    vi.stubGlobal('fetch', fetchMock)

    render(<App />)

    expect(screen.getByText('慢性病风险建模')).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getAllByText('糖尿病').length).toBeGreaterThan(0)
      expect(screen.getAllByText('LogisticRegression').length).toBeGreaterThan(0)
    })
  })
})