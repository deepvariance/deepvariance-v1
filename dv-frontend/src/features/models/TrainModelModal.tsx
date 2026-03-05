import { getDatasetColumns } from '@/shared/api/datasets'
import { createTrainingJob } from '@/shared/api/jobs'
import { DatasetSelector } from '@/shared/components'
import { useDatasets } from '@/shared/hooks/useDatasets'
import type { DatasetOption } from '@/shared/types'
import { formatters } from '@/shared/utils/formatters'
import {
  Alert,
  Badge,
  Box,
  Button,
  Group,
  Loader,
  Modal,
  Select,
  Stack,
  Text,
  TextInput,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import {
  IconAlertCircle,
  IconBolt,
  IconCheck,
  IconSparkles,
} from '@tabler/icons-react'
import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'

interface TrainModelModalProps {
  opened: boolean
  onClose: () => void
}

export function TrainModelModal({ opened, onClose }: TrainModelModalProps) {
  const queryClient = useQueryClient()
  const [modelName, setModelName] = useState('')
  const [selectedDataset, setSelectedDataset] = useState<string>('')
  const [task, setTask] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [availableTasks, setAvailableTasks] = useState<string[]>([
    'classification',
    'regression',
    'detection',
    'clustering',
  ])
  const [isAutoSelected, setIsAutoSelected] = useState(false)
  const [targetColumn, setTargetColumn] = useState('')
  const [columns, setColumns] = useState<string[]>([])
  const [loadingColumns, setLoadingColumns] = useState(false)

  // Fetch datasets
  const {
    data: apiDatasets = [],
    isLoading,
    error,
  } = useDatasets({ readiness: 'ready' })

  // Transform API datasets to DatasetOption format
  const datasets: DatasetOption[] = useMemo(() => {
    return apiDatasets.map(dataset => ({
      value: dataset.id,
      label: dataset.name,
      domain: formatters.domain(dataset.domain),
      rows: `${dataset.size} files`,
      tags: dataset.tags || [],
      storage: formatters.storage(dataset.storage),
    }))
  }, [apiDatasets])

  // Auto-select task when dataset is selected
  useEffect(() => {
    if (!selectedDataset || !apiDatasets.length) return

    const dataset = apiDatasets.find(d => d.id === selectedDataset)
    if (!dataset) return

    // Check if dataset has task recommendations in structure
    const structure = dataset.structure as any
    const supportedTasks = structure?.supported_tasks
    const recommendedTask = structure?.recommended_task

    if (supportedTasks && Array.isArray(supportedTasks)) {
      setAvailableTasks(supportedTasks)

      // Auto-select recommended task if available
      if (recommendedTask && supportedTasks.includes(recommendedTask)) {
        setTask(recommendedTask)
        setIsAutoSelected(true)
      } else {
        // Default to first available task
        setTask(supportedTasks[0])
        setIsAutoSelected(true)
      }
    } else {
      // Fallback: all tasks available
      setAvailableTasks([
        'classification',
        'regression',
        'detection',
        'clustering',
      ])
      setTask('classification')
      setIsAutoSelected(false)
    }
  }, [selectedDataset, apiDatasets])

  // Fetch columns when a tabular dataset is selected
  useEffect(() => {
    if (!selectedDataset || !apiDatasets.length) return

    const dataset = apiDatasets.find(d => d.id === selectedDataset)
    if (!dataset || dataset.domain !== 'tabular') {
      setColumns([])
      return
    }

    // Check if dataset has target column in metadata
    const savedTargetColumn = dataset.metadata?.target_column

    // Fetch columns for tabular datasets
    setLoadingColumns(true)
    getDatasetColumns(selectedDataset)
      .then(response => {
        setColumns(response.columns)
        // Priority 1: Use saved target column from dataset metadata
        if (savedTargetColumn && response.columns.includes(savedTargetColumn)) {
          setTargetColumn(savedTargetColumn)
        }
        // Priority 2: Auto-select first column if no column selected
        else if (!targetColumn && response.columns.length > 0) {
          setTargetColumn(response.columns[0])
        }
      })
      .catch(error => {
        console.error('Failed to fetch columns:', error)
        setColumns([])
      })
      .finally(() => {
        setLoadingColumns(false)
      })
  }, [selectedDataset, apiDatasets])

  const handleStartTraining = async () => {
    if (!task && trainingType === 'deep_learning') return
    if (!targetColumn && trainingType === 'automl') return

    setIsSubmitting(true)
    try {
      if (trainingType === 'automl') {
        // AutoML training via /api/jobs/train-automl
        const response = await fetch(
          'http://localhost:8000/api/jobs/train-automl',
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              dataset_id: selectedDataset,
              target_column: targetColumn,
              model_name: modelName.trim() || undefined,
            }),
          }
        )

        if (!response.ok) {
          const error = await response.json()
          throw new Error(error.detail || 'Failed to start AutoML training')
        }
      } else {
        // Deep Learning training via existing endpoint
        await createTrainingJob({
          dataset_id: selectedDataset,
          model_name: modelName.trim() || undefined,
          task: task as
            | 'classification'
            | 'regression'
            | 'clustering'
            | 'detection',
        })
      }

      notifications.show({
        title: 'Training Job Queued',
        message: `${trainingType === 'automl' ? 'AutoML' : 'Model'} "${modelName}" has been queued for training`,
        color: 'green',
        icon: <IconCheck size={16} />,
      })

      // Invalidate models cache to trigger immediate refresh
      queryClient.invalidateQueries({ queryKey: ['models'] })

      // Reset form and close modal
      setModelName('')
      setSelectedDataset('')
      setTask(null)
      setTargetColumn('')
      onClose()
    } catch (error) {
      console.error('Failed to start training:', error)
      notifications.show({
        title: 'Failed to Start Training',
        message: error instanceof Error ? error.message : 'An error occurred',
        color: 'red',
        icon: <IconAlertCircle size={16} />,
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  // Auto-detect training type based on dataset domain
  const selectedDatasetObj = apiDatasets.find(d => d.id === selectedDataset)
  const domain = selectedDatasetObj?.domain
  const isTabular = domain === 'tabular'
  const trainingType = isTabular ? 'automl' : 'deep_learning'
  const pipelineInfo = isTabular
    ? {
        type: 'AutoML',
        icon: '🤖',
        description: '8-stage LLM-driven pipeline',
      }
    : {
        type: 'Deep Learning',
        icon: '🧠',
        description: 'CNN-based neural network',
      }

  const isFormValid =
    modelName.trim() !== '' &&
    selectedDataset !== '' &&
    (trainingType === 'deep_learning'
      ? task !== null
      : targetColumn.trim() !== '')

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title="Train New Model"
      size="xl"
      centered
      styles={{
        title: {
          fontSize: '20px',
          fontWeight: 600,
        },
      }}
    >
      <Stack gap="lg">
        <Text size="15px" c="dimmed">
          Configure and start a new model training job
        </Text>

        {/* Model Name */}
        <div>
          <Text size="14px" fw={600} mb={8}>
            Model Name
          </Text>
          <TextInput
            placeholder="e.g., credit-risk-classifier"
            value={modelName}
            onChange={e => setModelName(e.currentTarget.value)}
            styles={{
              input: {
                fontSize: '15px',
              },
            }}
          />
          <Text size="13px" c="dimmed" mt={6}>
            Choose a descriptive name for your model
          </Text>
        </div>

        {/* Dataset Selection */}
        <div>
          <Text size="14px" fw={600} mb={8}>
            Training Dataset
          </Text>
          {isLoading ? (
            <Box
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: 110,
                border: '1px solid #E5E7EB',
                borderRadius: 6,
              }}
            >
              <Loader size="sm" color="#6366F1" />
            </Box>
          ) : error ? (
            <Alert
              icon={<IconAlertCircle size={16} />}
              title="Error loading datasets"
              color="red"
              variant="light"
            >
              {error instanceof Error
                ? error.message
                : 'Failed to fetch datasets'}
            </Alert>
          ) : datasets.length === 0 ? (
            <Alert
              icon={<IconAlertCircle size={16} />}
              color="indigo"
              variant="light"
            >
              No ready datasets available. Please upload a dataset first.
            </Alert>
          ) : (
            <DatasetSelector
              datasets={datasets}
              value={selectedDataset}
              onChange={setSelectedDataset}
            />
          )}
          <Text size="13px" c="dimmed" mt={6}>
            Choose a dataset from your library
          </Text>
          {selectedDataset && (
            <Alert
              icon={<Text size="20px">{pipelineInfo.icon}</Text>}
              color={trainingType === 'automl' ? 'indigo' : 'blue'}
              variant="light"
              mt={12}
            >
              <Group gap={8}>
                <Text size="14px" fw={600}>
                  {pipelineInfo.type} Pipeline
                </Text>
                <Badge
                  size="sm"
                  variant="light"
                  color={trainingType === 'automl' ? 'indigo' : 'blue'}
                >
                  Auto-detected
                </Badge>
              </Group>
              <Text size="13px" mt={4}>
                {pipelineInfo.description}
              </Text>
            </Alert>
          )}
        </div>

        {/* Task Type (Deep Learning Only) */}
        {trainingType === 'deep_learning' && (
          <div>
            <Group gap={8} mb={8}>
              <Text size="14px" fw={600}>
                Task Type
              </Text>
              {isAutoSelected && (
                <Badge
                  size="sm"
                  variant="light"
                  color="blue"
                  leftSection={<IconSparkles size={12} />}
                >
                  Auto-selected
                </Badge>
              )}
            </Group>
            <Select
              placeholder="Select task type"
              data={availableTasks.map(t => ({
                value: t,
                label: t.charAt(0).toUpperCase() + t.slice(1),
              }))}
              value={task}
              onChange={value => {
                setTask(value)
                setIsAutoSelected(false)
              }}
              styles={{
                input: {
                  fontSize: '15px',
                },
              }}
              disabled={!selectedDataset}
            />
            <Text size="13px" c="dimmed" mt={6}>
              {selectedDataset
                ? availableTasks.length < 4
                  ? `This dataset supports: ${availableTasks.join(', ')}`
                  : 'Select the type of machine learning task'
                : 'Select a dataset first'}
            </Text>
          </div>
        )}

        {/* Target Column (AutoML Only) */}
        {trainingType === 'automl' && (
          <div>
            <Text size="14px" fw={600} mb={8}>
              Target Column
            </Text>
            {!isTabular && selectedDataset ? (
              <Alert
                icon={<IconAlertCircle size={16} />}
                color="orange"
                variant="light"
              >
                AutoML requires tabular dataset. Selected dataset is not
                tabular.
              </Alert>
            ) : (
              <>
                <Select
                  placeholder={
                    loadingColumns
                      ? 'Loading columns...'
                      : 'Select target column'
                  }
                  data={columns.map(col => ({ value: col, label: col }))}
                  value={targetColumn}
                  onChange={value => setTargetColumn(value || '')}
                  disabled={
                    !selectedDataset ||
                    !isTabular ||
                    loadingColumns ||
                    columns.length === 0
                  }
                  searchable
                  styles={{
                    input: {
                      fontSize: '15px',
                    },
                  }}
                />
                <Text size="13px" c="dimmed" mt={6}>
                  {selectedDataset
                    ? columns.length > 0
                      ? `Select the column you want to predict (${columns.length} columns available)`
                      : loadingColumns
                        ? 'Loading columns...'
                        : 'No columns found in dataset'
                    : 'Select a tabular dataset first'}
                </Text>
              </>
            )}
          </div>
        )}

        {/* Action Buttons */}
        <Group justify="flex-end" gap={12} mt="md">
          <Button
            variant="light"
            color="gray"
            onClick={onClose}
            disabled={isSubmitting}
            styles={{
              root: {
                fontSize: '15px',
              },
            }}
          >
            Cancel
          </Button>
          <Button
            leftSection={!isSubmitting && <IconBolt size={18} />}
            color="indigo"
            disabled={!isFormValid || isSubmitting}
            loading={isSubmitting}
            onClick={handleStartTraining}
            styles={{
              root: {
                backgroundColor: '#6366F1',
                fontSize: '15px',
              },
            }}
          >
            {isSubmitting ? 'Starting...' : 'Start Training'}
          </Button>
        </Group>
      </Stack>
    </Modal>
  )
}
