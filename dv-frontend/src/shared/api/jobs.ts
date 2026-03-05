/**
 * Training Jobs API Service
 * API calls for training job management
 */
import { apiClient } from './client'

export interface HyperparametersConfig {
  learning_rate?: number
  batch_size?: number
  optimizer?: 'Adam' | 'SGD' | 'RMSprop'
  dropout_rate?: number
  max_iterations?: number
  target_accuracy?: number
}

export interface PipelineStage {
  stage: number
  name: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  started_at?: string
  completed_at?: string
  duration?: number
  error?: string
}

export interface TrainingRun {
  accuracy?: number
  precision?: number
  recall?: number
  f1_score?: number
  loss?: number
  duration_seconds?: number
}

export interface TrainingJob {
  id: string
  dataset_id: string
  model_id?: string
  job_type?: 'automl_training' | 'dl_training'
  status: 'pending' | 'queued' | 'running' | 'completed' | 'failed' | 'stopped'
  progress: number
  current_iteration: number
  total_iterations: number
  current_accuracy?: number
  best_accuracy?: number
  current_loss?: number // Current training loss
  best_loss?: number // Best (lowest) loss achieved
  precision?: number // Precision metric
  recall?: number // Recall metric
  f1_score?: number // F1-Score metric
  hyperparameters: HyperparametersConfig
  created_at: string
  started_at?: string
  completed_at?: string
  error_message?: string
  elapsed_time?: string // Human-readable elapsed time (e.g., "2h 15m")
  estimated_remaining?: string // Human-readable estimated time (e.g., "45m 32s")
  config?: {
    pipeline_stages?: PipelineStage[]
  }
  training_run?: TrainingRun
}

export interface TrainingJobCreate {
  dataset_id: string
  model_name?: string
  task: 'classification' | 'regression' | 'clustering' | 'detection'
  hyperparameters?: HyperparametersConfig
}

export interface JobsFilters {
  status?: string
}

export interface JobLog {
  time?: string
  timestamp?: string
  level: string
  message: string
}

export interface JobLogsResponse {
  job_id: string
  logs: JobLog[]
  total_lines: number
}

/**
 * Fetch all training jobs with optional filters
 */
export const getJobs = async (
  filters?: JobsFilters
): Promise<TrainingJob[]> => {
  const { data } = await apiClient.get<TrainingJob[]>('/jobs', {
    params: filters,
  })
  return data
}

/**
 * Get a single training job by ID
 */
export const getJob = async (id: string): Promise<TrainingJob> => {
  const { data } = await apiClient.get<TrainingJob>(`/jobs/${id}`)
  return data
}

/**
 * Create a new training job
 */
export const createTrainingJob = async (
  jobData: TrainingJobCreate
): Promise<TrainingJob> => {
  const { data } = await apiClient.post<TrainingJob>('/jobs', jobData)
  return data
}

/**
 * Cancel a training job
 */
export const cancelJob = async (id: string): Promise<void> => {
  await apiClient.post(`/jobs/${id}/cancel`)
}

/**
 * Delete a training job
 */
export const deleteJob = async (id: string): Promise<void> => {
  await apiClient.delete(`/jobs/${id}`)
}

/**
 * Restart a failed or stopped training job
 * Creates a new job with the same configuration
 */
export const restartJob = async (id: string): Promise<TrainingJob> => {
  const { data } = await apiClient.post<TrainingJob>(`/jobs/${id}/restart`)
  return data
}

/**
 * Get training job logs
 */
export const getJobLogs = async (id: string): Promise<JobLogsResponse> => {
  const { data } = await apiClient.get<JobLogsResponse>(`/jobs/${id}/logs`)
  return data
}
