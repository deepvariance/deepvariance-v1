/**
 * React Query hooks for Training Jobs
 * Provides data fetching and mutations for training job operations
 */
import {
  cancelJob,
  createTrainingJob,
  deleteJob,
  getJob,
  getJobLogs,
  getJobs,
  restartJob,
  type JobsFilters,
  type TrainingJobCreate,
} from '@/shared/api/jobs'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

/**
 * Fetch all training jobs with optional filters
 * Auto-refreshes every 5 seconds to track progress
 */
export function useJobs(filters?: JobsFilters) {
  return useQuery({
    queryKey: ['jobs', filters],
    queryFn: () => getJobs(filters),
    refetchInterval: 5000, // Refresh every 5 seconds for real-time progress
    refetchIntervalInBackground: true,
  })
}

/**
 * Fetch a single training job by ID
 * Auto-refreshes every 3 seconds for detailed progress tracking
 */
export function useJob(id: string) {
  return useQuery({
    queryKey: ['jobs', id],
    queryFn: () => getJob(id),
    enabled: !!id,
    refetchInterval: 3000, // Faster refresh for single job view
    refetchIntervalInBackground: true,
  })
}

/**
 * Create a new training job
 */
export function useCreateTrainingJob() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (jobData: TrainingJobCreate) => createTrainingJob(jobData),
    onSuccess: () => {
      // Invalidate jobs cache to trigger refresh
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
      // Also invalidate models cache since new model will be created
      queryClient.invalidateQueries({ queryKey: ['models'] })
    },
  })
}

/**
 * Cancel a running training job
 */
export function useCancelJob() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: string) => cancelJob(id),
    onSuccess: () => {
      // Invalidate jobs and models cache
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
      queryClient.invalidateQueries({ queryKey: ['models'] })
    },
  })
}

/**
 * Delete a training job
 */
export function useDeleteJob() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: string) => deleteJob(id),
    onSuccess: () => {
      // Invalidate jobs cache
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
    },
  })
}

/**
 * Restart a failed or stopped training job
 * Creates a new job with the same configuration
 */
export function useRestartJob() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: string) => restartJob(id),
    onSuccess: () => {
      // Invalidate jobs and models cache
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
      queryClient.invalidateQueries({ queryKey: ['models'] })
    },
  })
}

/**
 * Fetch training job logs
 * Auto-refreshes every 3 seconds for real-time log streaming
 */
export function useJobLogs(id: string) {
  return useQuery({
    queryKey: ['jobs', id, 'logs'],
    queryFn: () => getJobLogs(id),
    enabled: !!id,
    refetchInterval: 3000, // Refresh logs every 3 seconds
    refetchIntervalInBackground: true,
  })
}

/**
 * Get job for a specific model
 * Useful for showing training progress on model detail page
 */
export function useJobByModelId(modelId: string | undefined) {
  return useQuery({
    queryKey: ['jobs', 'model', modelId],
    queryFn: async () => {
      if (!modelId) return null
      const jobs = await getJobs()
      // Find the most recent job for this model
      return jobs.find(job => job.model_id === modelId) || null
    },
    enabled: !!modelId,
    refetchInterval: 3000, // Faster refresh for real-time updates (was 5000)
    refetchIntervalInBackground: true,
    staleTime: 0, // Always treat data as stale to force fresh fetches
    gcTime: 0, // Don't cache data (was cacheTime in older versions)
  })
}
