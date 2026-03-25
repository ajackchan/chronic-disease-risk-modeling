export function taskLabel(task: string): string {
  switch (task) {
    case 'cardiovascular':
      return '心血管疾病'
    case 'diabetes':
      return '糖尿病'
    case 'liver':
      return '肝病'
    case 'cancer':
      return '肿瘤'
    case 'multimorbidity_label':
      return '多病共存'
    default:
      return task
  }
}

export function riskLabelZh(riskLabel: string): string {
  switch (riskLabel) {
    case 'low':
      return '低风险'
    case 'medium':
      return '中风险'
    case 'high':
      return '高风险'
    default:
      return riskLabel
  }
}