import type { PipelineStage } from '@/shared/api/jobs'
import { useDataset } from '@/shared/hooks/useDatasets'
import {
  useCancelJob,
  useJobByModelId,
  useJobLogs,
} from '@/shared/hooks/useJobs'
import { useModel } from '@/shared/hooks/useModels'
import { useTrainingHistory } from '@/shared/hooks/useTrainingHistory'
import {
  ActionIcon,
  Alert,
  Badge,
  Box,
  Button,
  Card,
  Center,
  Code,
  Divider,
  Group,
  Loader,
  Progress,
  ScrollArea,
  SimpleGrid,
  Stack,
  Tabs,
  Text,
  Title,
  Tooltip,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import {
  IconActivity,
  IconAlertCircle,
  IconArrowLeft,
  IconChartLine,
  IconCheck,
  IconClock,
  IconDatabase,
  IconDownload,
  IconPlayerStop,
  IconRefresh,
  IconRobot,
  IconSettings,
  IconTerminal,
  IconX,
} from '@tabler/icons-react'
import { useEffect, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  Tooltip as RechartsTooltip,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from 'recharts'

// Log type definition
interface TrainingLog {
  time?: string
  timestamp?: string
  level: string
  message: string
}

// Helper function to extract best metrics from epoch_metrics array
function getBestMetricsFromEpochs(
  epochMetrics: Array<{
    iteration: number
    accuracy?: number
    precision?: number
    recall?: number
    f1_score?: number
    loss?: number
    timestamp: string
  }>
): {
  precision: number | null
  recall: number | null
  f1_score: number | null
} {
  if (!epochMetrics || epochMetrics.length === 0) {
    return { precision: null, recall: null, f1_score: null }
  }

  // Find the epoch with the highest accuracy
  let bestEpoch = epochMetrics[0]
  let maxAccuracy = bestEpoch?.accuracy ?? -1

  for (const epoch of epochMetrics) {
    if (epoch.accuracy != null && epoch.accuracy > maxAccuracy) {
      maxAccuracy = epoch.accuracy
      bestEpoch = epoch
    }
  }

  return {
    precision: bestEpoch?.precision ?? null,
    recall: bestEpoch?.recall ?? null,
    f1_score: bestEpoch?.f1_score ?? null,
  }
}

export function TrainingRunnerPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const [autoRefresh, setAutoRefresh] = useState(true)

  // Get run ID from URL query parameter
  const runIdFromUrl = searchParams.get('run')

  // Fetch model and job data
  const {
    data: model,
    isLoading: modelLoading,
    error: modelError,
    refetch: refetchModel,
  } = useModel(id!)
  const {
    data: job,
    isLoading: jobLoading,
    error: jobError,
    refetch: refetchJob,
  } = useJobByModelId(id)
  const { data: logsData, refetch: refetchLogs } = useJobLogs(job?.id || '')
  const { data: dataset } = useDataset(job?.dataset_id || '')
  const { data: trainingHistory } = useTrainingHistory(id || '')
  const cancelJobMutation = useCancelJob()
  // const restartJobMutation = useRestartJob() // Commented out for future use

  // Get the specific training run from URL or default to latest
  const currentTrainingRun = runIdFromUrl
    ? trainingHistory?.find(run => run.id === runIdFromUrl)
    : trainingHistory?.[0]

  // Determine which data source to use:
  // - If a specific run is selected (runIdFromUrl), use currentTrainingRun data
  // - Otherwise, use the latest job data for real-time updates
  const useHistoricalRun = runIdFromUrl && currentTrainingRun
  const displayRun = useHistoricalRun ? currentTrainingRun : null

  // State for toggleable chart lines
  const [hiddenLines, setHiddenLines] = useState<Record<string, boolean>>({})

  // Control auto-refresh based on training status (only for current job, not historical)
  const isTraining =
    !useHistoricalRun &&
    (job?.status === 'pending' ||
      job?.status === 'queued' ||
      job?.status === 'running')

  // Detect ML Pipeline jobs
  const isMLPipeline = displayRun
    ? displayRun.config?.pipeline_type === 'automl'
    : job?.job_type === 'automl_training'

  // Define all 8 stages upfront
  const allStages = [
    { stage: 1, name: 'Type Conversion', status: 'pending' },
    { stage: 2, name: 'Data Sampling', status: 'pending' },
    { stage: 3, name: 'Profile Generation (Sampled)', status: 'pending' },
    { stage: 4, name: 'Preprocessing Insights Generation', status: 'pending' },
    { stage: 5, name: 'Preprocessing Code Execution', status: 'pending' },
    { stage: 6, name: 'Profile Generation (Preprocessed)', status: 'pending' },
    { stage: 7, name: 'Model Recommendation', status: 'pending' },
    { stage: 8, name: 'Model Training', status: 'pending' },
  ]

  // Merge with actual stages from job or displayRun (override with real status/timing)
  const backendStages = displayRun
    ? displayRun.config?.pipeline_stages || []
    : job?.config?.pipeline_stages || []
  const pipelineStages = allStages.map(defaultStage => {
    const backendStage = backendStages.find(
      (s: PipelineStage) => s.stage === defaultStage.stage
    )
    return backendStage || defaultStage
  })

  // Set default tab based on job type
  const [activeTab, setActiveTab] = useState<string | null>(null)

  // Set default tab when job data loads
  useEffect(() => {
    if (job && activeTab === null) {
      setActiveTab('overview')
    }
  }, [job, activeTab])

  // Auto-refresh polling
  useEffect(() => {
    if (!autoRefresh || !isTraining) return

    const interval = setInterval(() => {
      refetchModel()
      refetchJob()
      if (job?.id) refetchLogs()
    }, 5000)

    return () => clearInterval(interval)
  }, [autoRefresh, isTraining, job?.id, refetchModel, refetchJob, refetchLogs])

  const isLoading = modelLoading || jobLoading
  const error = modelError || jobError

  // Extract best metrics from epoch_metrics (from displayRun only)
  const bestMetrics = displayRun?.epoch_metrics
    ? getBestMetricsFromEpochs(displayRun.epoch_metrics)
    : { precision: null, recall: null, f1_score: null }

  // Use real data from API
  const data =
    model && (job || displayRun)
      ? {
          modelName: model.name,
          task: model.task,
          status: model.status,
          currentIteration: displayRun
            ? displayRun.current_epoch || displayRun.total_epochs || 0
            : (job?.current_iteration ?? 0),
          totalIterations: displayRun
            ? (displayRun.total_epochs ?? 0)
            : (job?.total_iterations ?? 0),
          progress: displayRun
            ? (displayRun.progress ?? 0)
            : (job?.progress ?? 0),
          currentLoss: displayRun
            ? displayRun.final_loss
            : (job?.current_loss ?? null),
          bestLoss: displayRun
            ? displayRun.best_loss
            : (job?.best_loss ?? null),
          accuracy: displayRun
            ? displayRun.final_accuracy
              ? displayRun.final_accuracy * 100
              : null
            : job?.current_accuracy
              ? job.current_accuracy * 100
              : null,
          validationAccuracy: displayRun
            ? displayRun.best_accuracy
              ? displayRun.best_accuracy * 100
              : null
            : job?.best_accuracy
              ? job.best_accuracy * 100
              : null,
          precision: bestMetrics.precision
            ? bestMetrics.precision * 100
            : (model?.metrics?.precision ?? null),
          recall: bestMetrics.recall
            ? bestMetrics.recall * 100
            : (model?.metrics?.recall ?? null),
          f1Score: bestMetrics.f1_score
            ? bestMetrics.f1_score * 100
            : (model?.metrics?.f1_score ?? null),
          learningRate: displayRun
            ? (displayRun.config?.hyperparameters?.learning_rate ?? 0.001)
            : (job?.hyperparameters?.learning_rate ?? 0.001),
          batchSize: displayRun
            ? (displayRun.config?.hyperparameters?.batch_size ?? 32)
            : (job?.hyperparameters?.batch_size ?? 32),
          optimizer: displayRun
            ? (displayRun.config?.hyperparameters?.optimizer ?? 'Adam')
            : (job?.hyperparameters?.optimizer ?? 'Adam'),
          dropoutRate: displayRun
            ? (displayRun.config?.hyperparameters?.dropout_rate ?? 0.2)
            : (job?.hyperparameters?.dropout_rate ?? 0.2),
          maxIterations: displayRun
            ? (displayRun.config?.hyperparameters?.max_iterations ?? 10)
            : (job?.hyperparameters?.max_iterations ?? 10),
          targetAccuracy: displayRun
            ? (displayRun.config?.hyperparameters?.target_accuracy ?? 1.0)
            : (job?.hyperparameters?.target_accuracy ?? 1.0),
          elapsedTime: (() => {
            // For historical runs, always use duration_seconds
            if (displayRun) {
              const durationSeconds = displayRun.duration_seconds
              if (durationSeconds) {
                const hours = Math.floor(durationSeconds / 3600)
                const minutes = Math.floor((durationSeconds % 3600) / 60)
                const seconds = durationSeconds % 60
                if (hours > 0) return `${hours}h ${minutes}m`
                if (minutes > 0) return `${minutes}m ${seconds}s`
                return `${seconds}s`
              }
              return '0s'
            }

            // For current job, use job.elapsed_time (updated frequently)
            if (job?.status === 'running' && job.elapsed_time) {
              return job.elapsed_time
            }

            // For completed current job, use currentTrainingRun duration if available
            const durationSeconds = currentTrainingRun?.duration_seconds
            if (durationSeconds) {
              const hours = Math.floor(durationSeconds / 3600)
              const minutes = Math.floor((durationSeconds % 3600) / 60)
              const seconds = durationSeconds % 60
              if (hours > 0) return `${hours}h ${minutes}m`
              if (minutes > 0) return `${minutes}m ${seconds}s`
              return `${seconds}s`
            }
            return job?.elapsed_time || '0s'
          })(),
          estimatedRemaining: displayRun
            ? 'N/A'
            : job?.estimated_remaining || '0s',
          avgTimePerIteration: (() => {
            // Use displayRun duration for historical runs
            if (displayRun) {
              const durationSeconds = displayRun.duration_seconds
              const currentIter = displayRun.current_epoch

              if (durationSeconds && currentIter && currentIter > 0) {
                const avgSeconds = durationSeconds / currentIter
                if (avgSeconds < 60) return `${avgSeconds.toFixed(1)}s`
                if (avgSeconds < 3600) return `${(avgSeconds / 60).toFixed(1)}m`
                return `${(avgSeconds / 3600).toFixed(1)}h`
              }
              return 'N/A'
            }

            // For current job, use training run duration for accurate calculation
            const durationSeconds = currentTrainingRun?.duration_seconds
            const currentIter = job?.current_iteration

            if (durationSeconds && currentIter && currentIter > 0) {
              const avgSeconds = durationSeconds / currentIter
              if (avgSeconds < 60) return `${avgSeconds.toFixed(1)}s`
              if (avgSeconds < 3600) return `${(avgSeconds / 60).toFixed(1)}m`
              return `${(avgSeconds / 3600).toFixed(1)}h`
            }

            // Fallback to parsing elapsed_time string
            if (currentIter && currentIter > 0 && job?.elapsed_time) {
              const elapsed = job.elapsed_time
              let totalSeconds = 0

              const hoursMatch = elapsed.match(/(\d+)h/)
              if (hoursMatch) totalSeconds += parseInt(hoursMatch[1]) * 3600

              const minutesMatch = elapsed.match(/(\d+)m/)
              if (minutesMatch) totalSeconds += parseInt(minutesMatch[1]) * 60

              const secondsMatch = elapsed.match(/(\d+)s/)
              if (secondsMatch) totalSeconds += parseInt(secondsMatch[1])

              if (totalSeconds === 0) return 'N/A'

              const avgSeconds = totalSeconds / currentIter
              if (avgSeconds < 60) return `${avgSeconds.toFixed(1)}s`
              if (avgSeconds < 3600) return `${(avgSeconds / 60).toFixed(1)}m`
              return `${(avgSeconds / 3600).toFixed(1)}h`
            }

            return 'N/A'
          })(),
          datasetName: model.tags.find(tag => tag !== model.task) || 'Unknown',
          datasetSize: dataset?.size
            ? `${dataset.size.toLocaleString()} samples`
            : 'N/A',
        }
      : null

  if (!data) {
    return (
      <Box p="xl">
        <Alert color="yellow" title="No Training Data">
          Training data is not available yet. Please wait for the training to
          start.
        </Alert>
      </Box>
    )
  }

  // Parse logs data - the API returns { job_id, logs: [...], total_lines }
  const logs: TrainingLog[] = logsData?.logs || []

  // Handle cancel job
  const handleCancelJob = async () => {
    if (!job?.id) return

    try {
      await cancelJobMutation.mutateAsync(job.id)
      notifications.show({
        title: 'Job Cancelled',
        message: 'Training job has been cancelled successfully',
        color: 'green',
      })
    } catch (error) {
      notifications.show({
        title: 'Cancel Failed',
        message:
          error instanceof Error ? error.message : 'Failed to cancel job',
        color: 'red',
      })
    }
  }

  // Handle restart job
  /* Commented out for future use
  const handleRestartJob = async () => {
    if (!job?.id) return

    try {
      const newJob = await restartJobMutation.mutateAsync(job.id)
      notifications.show({
        title: 'Job Restarted',
        message: 'Training job has been restarted successfully',
        color: 'green',
      })
      // Navigate to the new job's training page
      if (newJob.model_id) {
        navigate(`/training/${newJob.model_id}`)
      }
    } catch (error) {
      notifications.show({
        title: 'Restart Failed',
        message:
          error instanceof Error ? error.message : 'Failed to restart job',
        color: 'red',
      })
    }
  }
  */

  // Show loading state
  if (isLoading) {
    return (
      <Stack gap={0} style={{ minHeight: '100vh', backgroundColor: '#FAFAFA' }}>
        <Center h="100vh">
          <Loader size="lg" color="#6366F1" />
        </Center>
      </Stack>
    )
  }

  // Show error state
  if (error) {
    return (
      <Stack gap={0} style={{ minHeight: '100vh', backgroundColor: '#FAFAFA' }}>
        <Box px={32} pt={40}>
          <Alert
            icon={<IconAlertCircle size={16} />}
            title="Error loading training data"
            color="red"
            variant="light"
          >
            {error instanceof Error
              ? error.message
              : 'Failed to fetch training data'}
          </Alert>
          <Button
            mt={16}
            variant="light"
            color="gray"
            leftSection={<IconArrowLeft size={16} />}
            onClick={() => navigate('/models')}
          >
            Back to Models
          </Button>
        </Box>
      </Stack>
    )
  }

  return (
    <Stack gap={0} style={{ minHeight: '100vh', backgroundColor: '#FAFAFA' }}>
      {/* Header Section */}
      <Box px={32} pt={40} pb={24}>
        <Group gap={16} mb={20}>
          <Button
            variant="subtle"
            color="gray"
            size="sm"
            leftSection={<IconArrowLeft size={16} />}
            onClick={() => navigate(`/models/${id}?tab=history`)}
            styles={{
              root: {
                fontSize: '14px',
                fontWeight: 500,
                color: '#6B7280',
                paddingLeft: 8,
                paddingRight: 12,
              },
            }}
          >
            Back to Model
          </Button>
        </Group>

        {/* Training Header */}
        <Group justify="space-between" align="flex-start" mb={24}>
          <div>
            <Group gap={12} mb={8}>
              <Title order={1} size={32} fw={700}>
                {data.modelName}
              </Title>
              <Badge
                variant="light"
                color="blue"
                size="lg"
                styles={{
                  root: {
                    fontSize: '13px',
                    fontWeight: 500,
                    textTransform: 'capitalize',
                  },
                }}
              >
                {data.status}
              </Badge>
              {useHistoricalRun && displayRun && (
                <Badge
                  variant="light"
                  color="orange"
                  size="lg"
                  styles={{
                    root: {
                      fontSize: '13px',
                      fontWeight: 500,
                    },
                  }}
                >
                  {displayRun.run_number
                    ? `Run #${displayRun.run_number}`
                    : 'Historical Run'}
                </Badge>
              )}
            </Group>
            <Group gap={6}>
              <Text size="15px" c="dimmed">
                {data.task.charAt(0).toUpperCase() + data.task.slice(1)} •
                Dataset:
              </Text>
              <Text
                size="15px"
                c="blue"
                fw={500}
                style={{ cursor: 'pointer', textDecoration: 'underline' }}
                onClick={() =>
                  model?.dataset_id && navigate(`/datasets/${model.dataset_id}`)
                }
              >
                {data.datasetName}
              </Text>
            </Group>
          </div>

          {/* Action Buttons */}
          <Group gap={12}>
            {!useHistoricalRun &&
              job &&
              (job.status === 'pending' ||
                job.status === 'queued' ||
                job.status === 'running') && (
                <Tooltip
                  label={
                    autoRefresh ? 'Pause auto-refresh' : 'Resume auto-refresh'
                  }
                >
                  <ActionIcon
                    size="lg"
                    variant={autoRefresh ? 'filled' : 'light'}
                    color="blue"
                    onClick={() => setAutoRefresh(!autoRefresh)}
                  >
                    <IconRefresh size={18} />
                  </ActionIcon>
                </Tooltip>
              )}
            {job &&
              (job.status === 'pending' ||
                job.status === 'queued' ||
                job.status === 'running') && (
                <Button
                  variant="light"
                  color="red"
                  leftSection={<IconPlayerStop size={16} />}
                  styles={{ root: { fontSize: '14px' } }}
                  onClick={handleCancelJob}
                  loading={cancelJobMutation.isPending}
                >
                  Stop Training
                </Button>
              )}
          </Group>
        </Group>

        {/* Status Cards */}
        <SimpleGrid cols={4} spacing={16}>
          {/* Progress Card */}
          <Card
            shadow="sm"
            padding="lg"
            radius={12}
            withBorder
            style={{ borderColor: '#E5E7EB' }}
          >
            <Stack gap={8}>
              <Text size="13px" c="dimmed" fw={500}>
                Progress
              </Text>
              <Text size="28px" fw={700}>
                {isMLPipeline
                  ? `${data.currentIteration}/8`
                  : `${data.currentIteration}/${data.totalIterations}`}
              </Text>
              <Progress
                value={data.progress}
                size="md"
                color="blue"
                animated={!useHistoricalRun && job?.status === 'running'}
              />
              <Text size="12px" c="dimmed">
                {data.progress.toFixed(1)}% Complete
              </Text>
            </Stack>
          </Card>

          {/* Current Accuracy Card */}
          <Card
            shadow="sm"
            padding="lg"
            radius={12}
            withBorder
            style={{ borderColor: '#E5E7EB' }}
          >
            <Stack gap={8}>
              <Text size="13px" c="dimmed" fw={500}>
                {isMLPipeline
                  ? model?.task?.toLowerCase() === 'regression'
                    ? 'R² Score'
                    : model?.task?.toLowerCase() === 'unknown'
                      ? 'Metric'
                      : 'Accuracy'
                  : 'Current Accuracy'}
              </Text>
              {model?.task?.toLowerCase() === 'unknown' ? (
                // Show placeholder for unknown task type (being detected)
                <>
                  <Text size="28px" fw={700} c="dimmed">
                    --
                  </Text>
                  <Text size="12px" c="dimmed">
                    Detecting task type...
                  </Text>
                </>
              ) : isMLPipeline &&
                model?.task?.toLowerCase() === 'regression' ? (
                // Show R² for regression models
                model?.metrics?.r2 !== null &&
                model?.metrics?.r2 !== undefined ? (
                  <>
                    <Text
                      size="28px"
                      fw={700}
                      c={model.metrics.r2 > 0 ? 'blue' : 'red'}
                    >
                      {(model.metrics.r2 * 100).toFixed(2)}%
                    </Text>
                    <Progress
                      value={Math.max(0, model.metrics.r2 * 100)}
                      size="md"
                      color={model.metrics.r2 > 0 ? 'blue' : 'red'}
                    />
                    <Text size="12px" c="dimmed">
                      Test set
                    </Text>
                  </>
                ) : (
                  <>
                    <Text size="28px" fw={700} c="dimmed">
                      --
                    </Text>
                    <Text size="12px" c="dimmed">
                      No data yet
                    </Text>
                  </>
                )
              ) : data.accuracy !== null ? (
                <>
                  <Text size="28px" fw={700} c="blue">
                    {data.accuracy.toFixed(2)}%
                  </Text>
                  <Progress value={data.accuracy ?? 0} size="md" color="blue" />
                  <Text size="12px" c="dimmed">
                    {isMLPipeline ? 'Test set' : 'Latest iteration'}
                  </Text>
                </>
              ) : (
                <>
                  <Text size="28px" fw={700} c="dimmed">
                    --
                  </Text>
                  <Text size="12px" c="dimmed">
                    No data yet
                  </Text>
                </>
              )}
            </Stack>
          </Card>

          {/* Best Accuracy / Loss Card - conditional based on job type */}
          <Card
            shadow="sm"
            padding="lg"
            radius={12}
            withBorder
            style={{ borderColor: '#E5E7EB' }}
          >
            <Stack gap={8}>
              <Text size="13px" c="dimmed" fw={500}>
                {isMLPipeline ? 'Loss' : 'Best Accuracy'}
              </Text>
              {isMLPipeline ? (
                data.bestLoss !== null ? (
                  <>
                    <Text size="28px" fw={700} c="orange">
                      {data.bestLoss.toFixed(4)}
                    </Text>
                    <Text size="12px" c="dimmed">
                      Lower is better
                    </Text>
                  </>
                ) : (
                  <>
                    <Text size="28px" fw={700} c="dimmed">
                      --
                    </Text>
                    <Text size="12px" c="dimmed">
                      No data yet
                    </Text>
                  </>
                )
              ) : data.validationAccuracy !== null ? (
                <>
                  <Text size="28px" fw={700} c="green">
                    {data.validationAccuracy.toFixed(2)}%
                  </Text>
                  <Progress
                    value={data.validationAccuracy ?? 0}
                    size="md"
                    color="green"
                  />
                  <Text size="12px" c="dimmed">
                    All iterations
                  </Text>
                </>
              ) : (
                <>
                  <Text size="28px" fw={700} c="dimmed">
                    --
                  </Text>
                  <Text size="12px" c="dimmed">
                    No data yet
                  </Text>
                </>
              )}
            </Stack>
          </Card>

          {/* Time Card */}
          <Card
            shadow="sm"
            padding="lg"
            radius={12}
            withBorder
            style={{ borderColor: '#E5E7EB' }}
          >
            <Stack gap={8}>
              <Text size="13px" c="dimmed" fw={500}>
                Time
              </Text>
              <Text size="28px" fw={700}>
                {data.elapsedTime}
              </Text>
              {job?.status === 'running' ? (
                <Text size="12px" c="dimmed" fw={500}>
                  Remaining: {data.estimatedRemaining}
                </Text>
              ) : (
                <Text size="12px" c="dimmed" fw={500}>
                  Avg per iteration: {data.avgTimePerIteration}
                </Text>
              )}
            </Stack>
          </Card>
        </SimpleGrid>
      </Box>

      {/* Main Content */}
      <Box px={32} pb={32}>
        <Card
          shadow="sm"
          padding={0}
          radius={12}
          withBorder
          style={{ borderColor: '#E5E7EB' }}
        >
          <Tabs value={activeTab} onChange={setActiveTab}>
            <Tabs.List
              style={{
                borderBottom: '1px solid #E5E7EB',
                paddingLeft: 24,
                paddingRight: 24,
              }}
            >
              <Tabs.Tab
                value="overview"
                leftSection={<IconActivity size={16} />}
                styles={{
                  tab: {
                    fontSize: '14px',
                    fontWeight: 500,
                    paddingTop: 16,
                    paddingBottom: 16,
                  },
                }}
              >
                Overview
              </Tabs.Tab>
              <Tabs.Tab
                value="dataset"
                leftSection={<IconDatabase size={16} />}
                styles={{
                  tab: {
                    fontSize: '14px',
                    fontWeight: 500,
                    paddingTop: 16,
                    paddingBottom: 16,
                  },
                }}
              >
                Dataset
              </Tabs.Tab>
              <Tabs.Tab
                value="hyperparameters"
                leftSection={<IconSettings size={16} />}
                styles={{
                  tab: {
                    fontSize: '14px',
                    fontWeight: 500,
                    paddingTop: 16,
                    paddingBottom: 16,
                  },
                }}
              >
                Hyperparameters
              </Tabs.Tab>
              {!isMLPipeline && (
                <Tabs.Tab
                  value="metrics"
                  leftSection={<IconChartLine size={16} />}
                  styles={{
                    tab: {
                      fontSize: '14px',
                      fontWeight: 500,
                      paddingTop: 16,
                      paddingBottom: 16,
                    },
                  }}
                >
                  Metrics
                </Tabs.Tab>
              )}
              <Tabs.Tab
                value="logs"
                leftSection={<IconTerminal size={16} />}
                styles={{
                  tab: {
                    fontSize: '14px',
                    fontWeight: 500,
                    paddingTop: 16,
                    paddingBottom: 16,
                  },
                }}
              >
                Logs
              </Tabs.Tab>
            </Tabs.List>{' '}
            {/* Overview Tab */}
            <Tabs.Panel value="overview" p={24}>
              <Stack gap={24}>
                {/* ML Pipeline Stages - Show first for AutoML */}
                {isMLPipeline && (
                  <div>
                    <Text size="16px" fw={600} mb={4}>
                      AutoML Pipeline Progress
                    </Text>
                    <Text size="14px" c="dimmed" mb={20}>
                      8-stage automated machine learning pipeline
                    </Text>
                    <SimpleGrid cols={3} spacing={12}>
                      {pipelineStages.map(
                        (stage: PipelineStage, index: number) => {
                          const isCompleted = stage.status === 'completed'
                          const isRunning = stage.status === 'running'
                          const isFailed = stage.status === 'failed'

                          // Calculate duration if available
                          let duration = 'N/A'
                          if (stage.started_at && stage.completed_at) {
                            const start = new Date(stage.started_at).getTime()
                            const end = new Date(stage.completed_at).getTime()
                            const seconds = Math.round((end - start) / 1000)
                            if (seconds < 60) duration = `${seconds}s`
                            else
                              duration = `${Math.round(seconds / 60)}m ${seconds % 60}s`
                          }

                          return (
                            <Card
                              key={index}
                              padding="md"
                              radius={8}
                              withBorder
                              style={{
                                borderColor: isCompleted
                                  ? '#10B981'
                                  : isRunning
                                    ? '#3B82F6'
                                    : isFailed
                                      ? '#EF4444'
                                      : '#E5E7EB',
                                borderWidth: isRunning ? 2 : 1,
                                backgroundColor: isCompleted
                                  ? '#F0FDF4'
                                  : isRunning
                                    ? '#EFF6FF'
                                    : isFailed
                                      ? '#FEF2F2'
                                      : '#F9FAFB',
                              }}
                            >
                              <Group gap={8} mb={8}>
                                {isCompleted && (
                                  <IconCheck
                                    size={18}
                                    color="#10B981"
                                    style={{ flexShrink: 0 }}
                                  />
                                )}
                                {isRunning && <Loader size={16} color="blue" />}
                                {isFailed && (
                                  <IconX
                                    size={18}
                                    color="#EF4444"
                                    style={{ flexShrink: 0 }}
                                  />
                                )}
                                {!isCompleted && !isRunning && !isFailed && (
                                  <IconClock
                                    size={18}
                                    color="#9CA3AF"
                                    style={{ flexShrink: 0 }}
                                  />
                                )}
                                <Text
                                  size="13px"
                                  fw={600}
                                  c={
                                    isCompleted
                                      ? 'green'
                                      : isRunning
                                        ? 'blue'
                                        : isFailed
                                          ? 'red'
                                          : 'dimmed'
                                  }
                                >
                                  Stage {stage.stage}
                                </Text>
                              </Group>
                              <Text size="14px" fw={500} mb={8}>
                                {stage.name}
                              </Text>
                              <Text size="12px" c="dimmed">
                                {isCompleted
                                  ? `Completed in ${duration}`
                                  : isRunning
                                    ? 'In progress...'
                                    : isFailed
                                      ? stage.error || 'Failed'
                                      : 'Pending'}
                              </Text>
                            </Card>
                          )
                        }
                      )}
                    </SimpleGrid>
                  </div>
                )}

                {/* Training Curves - Hide for AutoML */}
                {!isMLPipeline && (
                  <div>
                    <Text size="16px" fw={600} mb={16}>
                      Training Curves
                    </Text>
                    {(() => {
                      // Filter out iteration 0 and check if we have at least 2 iterations with metrics
                      const metricsData = currentTrainingRun?.epoch_metrics
                        ? currentTrainingRun.epoch_metrics.filter(
                            m => m.iteration > 0
                          )
                        : []
                      const hasData = metricsData.length >= 2
                      const currentStatus = displayRun
                        ? displayRun.status
                        : job?.status

                      if (currentStatus === 'pending') {
                        return (
                          <Box
                            p={32}
                            style={{
                              backgroundColor: '#F9FAFB',
                              borderRadius: 8,
                              border: '1px solid #E5E7EB',
                              minHeight: 300,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                            }}
                          >
                            <Center style={{ height: '100%' }}>
                              <Stack gap={12} align="center">
                                <Text c="dimmed" size="15px">
                                  Waiting for training to start...
                                </Text>
                              </Stack>
                            </Center>
                          </Box>
                        )
                      }

                      if (
                        currentStatus === 'running' &&
                        metricsData.length < 2
                      ) {
                        return (
                          <Box
                            p={32}
                            style={{
                              backgroundColor: '#F9FAFB',
                              borderRadius: 8,
                              border: '1px solid #E5E7EB',
                              minHeight: 300,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                            }}
                          >
                            <Center style={{ height: '100%' }}>
                              <Stack gap={12} align="center">
                                <Text c="dimmed" size="15px">
                                  Collecting data... Training curves will appear
                                  after at least 2 iterations
                                </Text>
                              </Stack>
                            </Center>
                          </Box>
                        )
                      }

                      if (!hasData) {
                        return (
                          <Box
                            p={32}
                            style={{
                              backgroundColor: '#F9FAFB',
                              borderRadius: 8,
                              border: '1px solid #E5E7EB',
                              minHeight: 300,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                            }}
                          >
                            <Center style={{ height: '100%' }}>
                              <Stack gap={12} align="center">
                                <Text c="dimmed" size="15px">
                                  No training data available
                                </Text>
                              </Stack>
                            </Center>
                          </Box>
                        )
                      }

                      // Filter out iteration 0 (initial state) and prepare data
                      // Sample data for testing
                      // Filter out iteration 0 (initial state) and prepare data
                      const chartData = (
                        currentTrainingRun?.epoch_metrics || []
                      )
                        .filter(m => m.iteration > 0)
                        .map(m => ({
                          iteration: m.iteration,
                          accuracy: m.accuracy,
                          loss: m.loss,
                          precision: m.precision,
                          recall: m.recall,
                          f1_score: m.f1_score,
                        }))

                      // Find best iteration (highest accuracy)
                      const bestIteration = chartData.reduce(
                        (best, current) =>
                          current.accuracy > best.accuracy ? current : best,
                        chartData[0]
                      )

                      return (
                        <ResponsiveContainer width="100%" height={400}>
                          <LineChart
                            data={chartData}
                            margin={{
                              top: 20,
                              right: 0,
                              left: 0,
                              bottom: 0,
                            }}
                          >
                            <XAxis
                              dataKey="iteration"
                              label={{
                                value: 'Iteration',
                                position: 'insideBottom',
                                offset: -5,
                              }}
                            />
                            <YAxis
                              yAxisId="left"
                              width={60}
                              domain={[0, 1]}
                              tickFormatter={value =>
                                `${(value * 100).toFixed(0)}%`
                              }
                              label={{
                                value: 'Metrics (%)',
                                angle: -90,
                                position: 'insideLeft',
                              }}
                            />
                            <YAxis
                              yAxisId="right"
                              orientation="right"
                              label={{
                                value: 'Loss',
                                angle: 90,
                                position: 'insideRight',
                              }}
                            />
                            <CartesianGrid strokeDasharray="3 3" />
                            <RechartsTooltip
                              labelFormatter={(value: number | string) => {
                                const isBest =
                                  Number(value) === bestIteration?.iteration
                                return `Iteration ${value}${isBest ? ' (Best)' : ''}`
                              }}
                              formatter={(value: number | string) => {
                                if (typeof value === 'number') {
                                  return (value * 100).toFixed(2) + '%'
                                }
                                return value
                              }}
                            />
                            <Legend
                              onClick={data => {
                                if (data.dataKey) {
                                  setHiddenLines(prev => ({
                                    ...prev,
                                    [data.dataKey as string]:
                                      !prev[data.dataKey as string],
                                  }))
                                }
                              }}
                              wrapperStyle={{
                                cursor: 'pointer',
                                paddingTop: '20px',
                              }}
                            />
                            <Line
                              yAxisId="left"
                              type="monotone"
                              dataKey="accuracy"
                              stroke="#2563eb"
                              strokeWidth={2}
                              name="Accuracy"
                              hide={hiddenLines['accuracy']}
                              dot={(props: {
                                cx?: number
                                cy?: number
                                payload?: { iteration: number }
                              }) => {
                                if (
                                  props.payload?.iteration ===
                                  bestIteration?.iteration
                                ) {
                                  return (
                                    <circle
                                      cx={props.cx}
                                      cy={props.cy}
                                      r={6}
                                      fill="#2563eb"
                                      stroke="black"
                                      strokeWidth={2}
                                    />
                                  )
                                }
                                return (
                                  <circle
                                    cx={props.cx}
                                    cy={props.cy}
                                    r={3}
                                    fill="#2563eb"
                                  />
                                )
                              }}
                            />
                            <Line
                              yAxisId="left"
                              type="monotone"
                              dataKey="precision"
                              stroke="#059669"
                              strokeWidth={2}
                              name="Precision"
                              hide={hiddenLines['precision']}
                            />
                            <Line
                              yAxisId="left"
                              type="monotone"
                              dataKey="recall"
                              stroke="#ea580c"
                              strokeWidth={2}
                              name="Recall"
                              hide={hiddenLines['recall']}
                            />
                            <Line
                              yAxisId="left"
                              type="monotone"
                              dataKey="f1_score"
                              stroke="#7c3aed"
                              strokeWidth={2}
                              name="F1 Score"
                              hide={hiddenLines['f1_score']}
                            />
                            <Line
                              yAxisId="right"
                              type="monotone"
                              dataKey="loss"
                              stroke="#dc2626"
                              strokeWidth={2}
                              name="Loss"
                              hide={hiddenLines['loss']}
                              dot={(props: {
                                cx?: number
                                cy?: number
                                payload?: { iteration: number }
                              }) => {
                                if (
                                  props.payload?.iteration ===
                                  bestIteration?.iteration
                                ) {
                                  return (
                                    <circle
                                      cx={props.cx}
                                      cy={props.cy}
                                      r={6}
                                      fill="#dc2626"
                                      stroke="black"
                                      strokeWidth={2}
                                    />
                                  )
                                }
                                return (
                                  <circle
                                    cx={props.cx}
                                    cy={props.cy}
                                    r={3}
                                    fill="#dc2626"
                                  />
                                )
                              }}
                            />
                            <ReferenceLine
                              x={bestIteration?.iteration}
                              yAxisId="left"
                              stroke="green"
                              strokeWidth={1}
                              label={{
                                value: 'Best',
                                position: 'top',
                                fill: 'green',
                              }}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      )
                    })()}
                  </div>
                )}

                {/* Performance Metrics */}
                <div>
                  <Text size="16px" fw={600} mb={16}>
                    Performance Metrics
                  </Text>
                  <SimpleGrid cols={isMLPipeline ? 2 : 4} spacing={16}>
                    <Card
                      padding="md"
                      radius={8}
                      style={{ backgroundColor: '#F9FAFB' }}
                    >
                      <Text size="13px" c="dimmed" mb={4}>
                        {isMLPipeline ? 'Accuracy' : 'Current Accuracy'}
                      </Text>
                      <Text size="24px" fw={700} c="blue">
                        {data.accuracy !== null
                          ? `${data.accuracy.toFixed(2)}%`
                          : 'N/A'}
                      </Text>
                    </Card>
                    {!isMLPipeline && (
                      <Card
                        padding="md"
                        radius={8}
                        style={{ backgroundColor: '#F9FAFB' }}
                      >
                        <Text size="13px" c="dimmed" mb={4}>
                          Best Accuracy
                        </Text>
                        <Text size="24px" fw={700} c="green">
                          {data.validationAccuracy !== null
                            ? `${data.validationAccuracy.toFixed(2)}%`
                            : 'N/A'}
                        </Text>
                      </Card>
                    )}
                    <Card
                      padding="md"
                      radius={8}
                      style={{ backgroundColor: '#F9FAFB' }}
                    >
                      <Text size="13px" c="dimmed" mb={4}>
                        {isMLPipeline ? 'Loss' : 'Current Loss'}
                      </Text>
                      <Text size="24px" fw={700} c="orange">
                        {data.currentLoss !== null
                          ? data.currentLoss.toFixed(4)
                          : 'N/A'}
                      </Text>
                    </Card>
                    {!isMLPipeline && (
                      <Card
                        padding="md"
                        radius={8}
                        style={{ backgroundColor: '#F9FAFB' }}
                      >
                        <Text size="13px" c="dimmed" mb={4}>
                          Best Loss
                        </Text>
                        <Text size="24px" fw={700} c="teal">
                          {data.bestLoss !== null && data.bestLoss !== undefined
                            ? data.bestLoss.toFixed(4)
                            : 'N/A'}
                        </Text>
                      </Card>
                    )}
                  </SimpleGrid>
                </div>

                {/* Classification Metrics */}
                <div>
                  <Text size="16px" fw={600} mb={16}>
                    Classification Metrics
                  </Text>
                  <SimpleGrid cols={3} spacing={16}>
                    <Card
                      padding="md"
                      radius={8}
                      style={{ backgroundColor: '#F9FAFB' }}
                    >
                      <Text size="13px" c="dimmed" mb={4}>
                        Precision
                      </Text>
                      <Text size="24px" fw={700} c="violet">
                        {data.precision !== null
                          ? `${data.precision.toFixed(2)}%`
                          : 'N/A'}
                      </Text>
                    </Card>
                    <Card
                      padding="md"
                      radius={8}
                      style={{ backgroundColor: '#F9FAFB' }}
                    >
                      <Text size="13px" c="dimmed" mb={4}>
                        Recall
                      </Text>
                      <Text size="24px" fw={700} c="grape">
                        {data.recall !== null
                          ? `${data.recall.toFixed(2)}%`
                          : 'N/A'}
                      </Text>
                    </Card>
                    <Card
                      padding="md"
                      radius={8}
                      style={{ backgroundColor: '#F9FAFB' }}
                    >
                      <Text size="13px" c="dimmed" mb={4}>
                        F1-Score
                      </Text>
                      <Text size="24px" fw={700} c="indigo">
                        {data.f1Score !== null
                          ? `${data.f1Score.toFixed(2)}%`
                          : 'N/A'}
                      </Text>
                    </Card>
                  </SimpleGrid>

                  {data.precision === null &&
                    data.recall === null &&
                    data.f1Score === null && (
                      <Text
                        size="13px"
                        c="dimmed"
                        mt={12}
                        style={{ fontStyle: 'italic' }}
                      >
                        Classification metrics will appear here once training
                        begins.
                      </Text>
                    )}
                </div>
              </Stack>
            </Tabs.Panel>
            {/* Logs Tab */}
            <Tabs.Panel value="logs" p={0}>
              <Box p={16} style={{ borderBottom: '1px solid #E5E7EB' }}>
                <Group justify="space-between">
                  <Text size="14px" fw={600}>
                    Training Logs {logs.length > 0 && `(${logs.length} lines)`}
                  </Text>
                  <Group gap={8}>
                    <Button
                      size="xs"
                      variant="light"
                      color="gray"
                      leftSection={<IconDownload size={14} />}
                      disabled={logs.length === 0}
                    >
                      Download
                    </Button>
                  </Group>
                </Group>
              </Box>
              <ScrollArea h={500} style={{ backgroundColor: '#1e1e1e' }}>
                <Box p={16}>
                  {logs.length > 0 ? (
                    logs.map((log: TrainingLog, index: number) => (
                      <Group
                        key={index}
                        gap={12}
                        mb={4}
                        style={{
                          fontFamily: 'monospace',
                          fontSize: '13px',
                        }}
                      >
                        <Text c="gray.5" style={{ minWidth: 80 }}>
                          {log.time || log.timestamp || ''}
                        </Text>
                        <Badge
                          size="xs"
                          color={
                            log.level === 'WARNING'
                              ? 'yellow'
                              : log.level === 'ERROR'
                                ? 'red'
                                : log.level === 'INFO'
                                  ? 'blue'
                                  : 'gray'
                          }
                          variant="light"
                          style={{ minWidth: 60 }}
                        >
                          {log.level}
                        </Badge>
                        <Text c="gray.3">{log.message}</Text>
                      </Group>
                    ))
                  ) : (
                    <Center py={40}>
                      <Text c="gray.5" size="14px">
                        {job?.status === 'pending' || job?.status === 'queued'
                          ? 'Waiting for training to start...'
                          : job?.status === 'running'
                            ? 'Loading logs...'
                            : 'No logs available for this training run'}
                      </Text>
                    </Center>
                  )}
                </Box>
              </ScrollArea>
            </Tabs.Panel>
            {/* Metrics Tab */}
            <Tabs.Panel value="metrics" p={24}>
              {isMLPipeline ? (
                <Center py={60}>
                  <Stack gap={12} align="center">
                    <IconRobot size={48} color="#9CA3AF" />
                    <Text size="18px" fw={600} c="dimmed">
                      Coming Soon
                    </Text>
                    <Text size="14px" c="dimmed">
                      ML Pipeline metrics visualization in development
                    </Text>
                  </Stack>
                </Center>
              ) : (
                <Stack gap={24}>
                  {/* Performance Metrics */}
                  <div>
                    <Text size="16px" fw={600} mb={16}>
                      Performance Metrics
                    </Text>
                    <SimpleGrid cols={4} spacing={16}>
                      <Card
                        padding="md"
                        radius={8}
                        style={{
                          backgroundColor: '#F0FDF4',
                          border: '1px solid #BBF7D0',
                        }}
                      >
                        <Stack gap={8}>
                          <Text size="12px" c="dimmed" tt="uppercase" fw={600}>
                            Accuracy
                          </Text>
                          <Text size="24px" fw={700} c="green">
                            {data.validationAccuracy !== null
                              ? `${data.validationAccuracy.toFixed(2)}%`
                              : 'N/A'}
                          </Text>
                        </Stack>
                      </Card>
                      <Card
                        padding="md"
                        radius={8}
                        style={{
                          backgroundColor: '#EFF6FF',
                          border: '1px solid #BFDBFE',
                        }}
                      >
                        <Stack gap={8}>
                          <Text size="12px" c="dimmed" tt="uppercase" fw={600}>
                            Precision
                          </Text>
                          <Text size="24px" fw={700} c="blue">
                            {data.precision !== null
                              ? `${data.precision.toFixed(2)}%`
                              : 'N/A'}
                          </Text>
                        </Stack>
                      </Card>
                      <Card
                        padding="md"
                        radius={8}
                        style={{
                          backgroundColor: '#FEF3C7',
                          border: '1px solid #FDE68A',
                        }}
                      >
                        <Stack gap={8}>
                          <Text size="12px" c="dimmed" tt="uppercase" fw={600}>
                            Recall
                          </Text>
                          <Text size="24px" fw={700} c="yellow">
                            {data.recall !== null
                              ? `${data.recall.toFixed(2)}%`
                              : 'N/A'}
                          </Text>
                        </Stack>
                      </Card>
                      <Card
                        padding="md"
                        radius={8}
                        style={{
                          backgroundColor: '#FEF2F2',
                          border: '1px solid #FECACA',
                        }}
                      >
                        <Stack gap={8}>
                          <Text size="12px" c="dimmed" tt="uppercase" fw={600}>
                            F1 Score
                          </Text>
                          <Text size="24px" fw={700} c="red">
                            {data.f1Score !== null
                              ? `${data.f1Score.toFixed(2)}%`
                              : 'N/A'}
                          </Text>
                        </Stack>
                      </Card>
                    </SimpleGrid>
                  </div>

                  {/* Loss Metrics */}
                  {job?.training_run?.loss !== undefined && (
                    <div>
                      <Text size="16px" fw={600} mb={16}>
                        Training Loss
                      </Text>
                      <Card
                        padding="lg"
                        radius={8}
                        style={{ backgroundColor: '#F9FAFB' }}
                      >
                        <Stack gap={8}>
                          <Text size="14px" c="dimmed">
                            Final Loss
                          </Text>
                          <Text size="28px" fw={700}>
                            {job.training_run.loss.toFixed(4)}
                          </Text>
                        </Stack>
                      </Card>
                    </div>
                  )}
                </Stack>
              )}
            </Tabs.Panel>
            {/* Hyperparameters Tab */}
            <Tabs.Panel value="hyperparameters" p={24}>
              <Stack gap={32}>
                {/* Initial Configuration */}
                <div>
                  <Text size="16px" fw={600} mb={16}>
                    Initial Training Configuration
                  </Text>
                  <SimpleGrid cols={2} spacing={16}>
                    <Box>
                      <Text size="13px" c="dimmed" mb={4}>
                        Learning Rate
                      </Text>
                      <Code block p={8} style={{ fontSize: '14px' }}>
                        {data.learningRate}
                      </Code>
                    </Box>
                    <Box>
                      <Text size="13px" c="dimmed" mb={4}>
                        Batch Size
                      </Text>
                      <Code block p={8} style={{ fontSize: '14px' }}>
                        {data.batchSize}
                      </Code>
                    </Box>
                    <Box>
                      <Text size="13px" c="dimmed" mb={4}>
                        Optimizer
                      </Text>
                      <Code block p={8} style={{ fontSize: '14px' }}>
                        {data.optimizer}
                      </Code>
                    </Box>
                    <Box>
                      <Text size="13px" c="dimmed" mb={4}>
                        Dropout Rate
                      </Text>
                      <Code block p={8} style={{ fontSize: '14px' }}>
                        {data.dropoutRate}
                      </Code>
                    </Box>
                    <Box>
                      <Text size="13px" c="dimmed" mb={4}>
                        Max Iterations
                      </Text>
                      <Code block p={8} style={{ fontSize: '14px' }}>
                        {data.maxIterations}
                      </Code>
                    </Box>
                    <Box>
                      <Text size="13px" c="dimmed" mb={4}>
                        Target Accuracy
                      </Text>
                      <Code block p={8} style={{ fontSize: '14px' }}>
                        {data.targetAccuracy}
                      </Code>
                    </Box>
                    <Box>
                      <Text size="13px" c="dimmed" mb={4}>
                        Task
                      </Text>
                      <Code
                        block
                        p={8}
                        style={{
                          fontSize: '14px',
                          textTransform: 'capitalize',
                        }}
                      >
                        {data.task}
                      </Code>
                    </Box>
                  </SimpleGrid>
                </div>
              </Stack>
            </Tabs.Panel>
            {/* Dataset Tab */}
            <Tabs.Panel value="dataset" p={24}>
              <Stack gap={24}>
                <div>
                  <Text size="16px" fw={600} mb={16}>
                    Dataset Information
                  </Text>
                  <Card
                    padding="lg"
                    radius={8}
                    style={{
                      backgroundColor: '#F9FAFB',
                      cursor: model?.dataset_id ? 'pointer' : 'default',
                      transition: 'background-color 0.2s ease',
                    }}
                    onClick={() =>
                      model?.dataset_id &&
                      navigate(`/datasets/${model.dataset_id}`)
                    }
                    onMouseEnter={e => {
                      if (model?.dataset_id) {
                        e.currentTarget.style.backgroundColor = '#F3F4F6'
                      }
                    }}
                    onMouseLeave={e => {
                      e.currentTarget.style.backgroundColor = '#F9FAFB'
                    }}
                  >
                    <Stack gap={12}>
                      <Group justify="space-between">
                        <Text size="14px" c="dimmed">
                          Dataset Name
                        </Text>
                        <Group gap={8}>
                          <Text size="14px" fw={500} c="blue">
                            {data.datasetName}
                          </Text>
                          <Text size="12px" c="dimmed">
                            (click to view)
                          </Text>
                        </Group>
                      </Group>
                      <Divider />
                      <Group justify="space-between">
                        <Text size="14px" c="dimmed">
                          Total Samples
                        </Text>
                        <Text size="14px" fw={500}>
                          {data.datasetSize}
                        </Text>
                      </Group>
                    </Stack>
                  </Card>
                </div>
              </Stack>
            </Tabs.Panel>
          </Tabs>
        </Card>
      </Box>
    </Stack>
  )
}
