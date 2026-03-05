// Core domain types
export interface Dataset {
  id: string
  name: string
  domain: 'tabular' | 'vision' | 'nlp' | 'audio'
  size: string
  readiness: 'ready' | 'processing' | 'error'
  lastModified: string
}

export interface Model {
  id: string
  name: string
  task: 'classification' | 'regression' | 'clustering' | 'detection' | 'unknown'
  framework: string
  accuracy?: number
  lastTrained: string
  status: 'active' | 'training' | 'failed'
}

export interface RecentItem {
  id: string
  name: string
  type: 'Dataset' | 'Model' | 'Notebook' | 'Project'
  time: string
  iconColor: string
}

export interface DatasetOption {
  value: string
  label: string
  domain: string
  rows: string
  tags: string[]
  storage: string
}
