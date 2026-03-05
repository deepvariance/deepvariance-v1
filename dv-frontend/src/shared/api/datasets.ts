/**
 * Dataset API Service
 * API calls for dataset management
 */
import { apiClient } from './client'

export interface Dataset {
  id: string
  name: string
  domain: 'tabular' | 'vision' | 'text' | 'audio'
  size: number | null // Number of files (can be null)
  readiness: 'ready' | 'profiling' | 'processing' | 'draft' | 'error'
  storage: 'local' | 'gcs' | 's3'
  path: string | null // Path can be null
  tags: string[]
  description?: string
  created_at: string
  updated_at: string
  last_modified?: string | null
  freshness?: string | null
  metadata?: {
    target_column?: string
  }
}

export interface DatasetsFilters {
  domain?: string
  readiness?: string
  search?: string
}

/**
 * Sanitize dataset data to handle null values gracefully
 */
const sanitizeDataset = (dataset: any): Dataset => {
  return {
    ...dataset,
    size: dataset.size ?? 0, // Default to 0 if null
    path: dataset.path ?? '', // Default to empty string if null
    tags: dataset.tags ?? [], // Default to empty array if null
    description: dataset.description ?? undefined,
    last_modified: dataset.last_modified ?? null,
    freshness: dataset.freshness ?? null,
  }
}

/**
 * Fetch all datasets with optional filters
 */
export const getDatasets = async (
  filters?: DatasetsFilters
): Promise<Dataset[]> => {
  const { data } = await apiClient.get<Dataset[]>('/datasets', {
    params: filters,
  })
  // Sanitize each dataset to handle null values
  return data.map(sanitizeDataset)
}

/**
 * Get a single dataset by ID
 */
export const getDataset = async (id: string): Promise<Dataset> => {
  const { data } = await apiClient.get<Dataset>(`/datasets/${id}`)
  return sanitizeDataset(data)
}

/**
 * Create/upload a new dataset
 */
export const createDataset = async (
  formData: FormData,
  onUploadProgress?: (progressEvent: any) => void
): Promise<Dataset> => {
  const { data } = await apiClient.post<Dataset>('/datasets', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress,
    timeout: 600000, // 10 minutes for large files
  })
  return data
}

/**
 * Update dataset metadata
 */
export const updateDataset = async (
  id: string,
  updates: {
    name?: string
    tags?: string[]
    description?: string
    readiness?: string
  }
): Promise<Dataset> => {
  const { data } = await apiClient.put<Dataset>(`/datasets/${id}`, updates)
  return data
}

/**
 * Update only dataset name
 */
export const updateDatasetName = async (
  id: string,
  name: string
): Promise<Dataset> => {
  const { data } = await apiClient.patch<Dataset>(
    `/datasets/${id}/name`,
    null,
    {
      params: { name },
    }
  )
  return data
}

/**
 * Delete a dataset (includes files)
 */
export const deleteDataset = async (id: string): Promise<void> => {
  await apiClient.delete(`/datasets/${id}`)
}

/**
 * Get column names and shape information from a tabular dataset
 */
export const getDatasetColumns = async (
  id: string
): Promise<{
  dataset_id: string
  columns: string[]
  total_columns: number
  total_rows: number
  shape: { rows: number; columns: number }
  dtypes: Record<string, string>
}> => {
  const { data } = await apiClient.get(`/datasets/${id}/columns`)
  return data
}
