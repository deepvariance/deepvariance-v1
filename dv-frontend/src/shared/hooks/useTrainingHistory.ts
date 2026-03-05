import { API_BASE_URL } from '@/shared/config/constants'
import { useQuery } from '@tanstack/react-query'

export interface TrainingRun {
  id: string
  run_number: number
  status: 'pending' | 'queued' | 'running' | 'completed' | 'failed' | 'stopped'
  progress: number
  current_epoch: number | null
  total_epochs: number | null
  final_loss: number | null
  final_accuracy: number | null
  best_loss: number | null
  best_accuracy: number | null
  duration_seconds: number | null
  error_message: string | null
  config: Record<string, any>
  epoch_metrics: any[] | null
  created_at: string | null
  started_at: string | null
  completed_at: string | null
  dataset_id: string | null
}

export function useTrainingHistory(modelId: string) {
  return useQuery<TrainingRun[]>({
    queryKey: ['training-history', modelId],
    queryFn: async () => {
      const response = await fetch(
        `${API_BASE_URL}/api/models/${modelId}/training-history`
      )
      if (!response.ok) {
        throw new Error('Failed to fetch training history')
      }
      return response.json()
    },
    enabled: !!modelId,
    refetchInterval: 5000, // Refetch every 5 seconds to match auto-refresh polling
  })
}
