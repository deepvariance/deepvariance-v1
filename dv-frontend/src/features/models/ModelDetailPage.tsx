import { COLORS, ROUTES } from '@/shared/config/constants'
import { useDataset } from '@/shared/hooks/useDatasets'
import { useJobByModelId, useRestartJob } from '@/shared/hooks/useJobs'
import { useModel, useUpdateModel } from '@/shared/hooks/useModels'
import { useTrainingHistory } from '@/shared/hooks/useTrainingHistory'
import { formatDate } from '@/shared/utils/formatters'
import {
  ActionIcon,
  Alert,
  Badge,
  Box,
  Button,
  Card,
  Center,
  Code,
  CopyButton,
  Divider,
  Group,
  Loader,
  Stack,
  Table,
  Tabs,
  Text,
  Textarea,
  Title,
  Tooltip,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import {
  IconAlertCircle,
  IconArrowLeft,
  IconChartBar,
  IconCheck,
  IconCircleCheck,
  IconCircleMinus,
  IconCircleX,
  IconClock,
  IconCopy,
  IconDownload,
  IconEdit,
  IconExternalLink,
  IconHistory,
  IconRefresh,
  IconRocket,
  IconX,
} from '@tabler/icons-react'
import { useEffect, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'

const taskColors: Record<string, string> = {
  classification: 'blue',
  regression: 'purple',
  clustering: 'teal',
  detection: 'grape',
}

const statusColors: Record<string, string> = {
  active: 'green',
  ready: 'green',
  training: 'blue',
  queued: 'cyan',
  draft: 'orange',
  failed: 'red',
}

const trainingStatusColors: Record<string, string> = {
  completed: 'green',
  running: 'blue',
  failed: 'red',
  stopped: 'orange',
  pending: 'gray',
  queued: 'cyan',
}

const trainingStatusIcons: Record<
  string,
  React.ComponentType<Record<string, unknown>>
> = {
  completed: IconCircleCheck,
  running: IconClock,
  failed: IconCircleX,
  stopped: IconCircleMinus,
  pending: IconClock,
  queued: IconClock,
}

function formatDuration(seconds: number | null): string {
  if (!seconds) return 'N/A'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = seconds % 60

  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`
  } else {
    return `${secs}s`
  }
}

export function ModelDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const [activeTab, setActiveTab] = useState<string | null>('overview')
  const [isEditingSummary, setIsEditingSummary] = useState(false)
  const [summaryValue, setSummaryValue] = useState('')

  // Fetch model data
  const { data: model, isLoading, error } = useModel(id || '')

  // Fetch dataset data if model has a dataset_id
  const { data: dataset } = useDataset(model?.dataset_id || '')

  // Fetch training history
  const {
    data: trainingHistory,
    isLoading: isLoadingHistory,
    refetch: refetchHistory,
  } = useTrainingHistory(id || '')

  // Fetch job for this model
  const { data: job } = useJobByModelId(id)

  // Restart job mutation
  const restartJobMutation = useRestartJob()

  // Update model mutation
  const updateModelMutation = useUpdateModel()

  // Handle restart job
  const handleRestartJob = async () => {
    if (!job?.id) return

    try {
      const newJob = await restartJobMutation.mutateAsync(job.id)
      notifications.show({
        title: 'Training Restarted',
        message: 'Training job has been restarted successfully',
        color: 'green',
      })
      // Navigate to the training page for the new job
      if (newJob.model_id) {
        navigate(`/models/${newJob.model_id}/training`)
      }
    } catch (error) {
      notifications.show({
        title: 'Restart Failed',
        message:
          error instanceof Error ? error.message : 'Failed to restart training',
        color: 'red',
      })
    }
  }

  // Set active tab from URL query parameter
  useEffect(() => {
    const tabParam = searchParams.get('tab')
    if (tabParam) {
      setActiveTab(tabParam)
    }
  }, [searchParams])

  // Initialize summary value when model loads
  useEffect(() => {
    if (model?.description) {
      setSummaryValue(model.description)
    }
  }, [model?.description])

  if (isLoading) {
    return (
      <Center style={{ height: '100vh' }}>
        <Loader size="lg" color={COLORS.PRIMARY} />
      </Center>
    )
  }

  if (error || !model) {
    return (
      <Box p={32}>
        <Alert
          icon={<IconAlertCircle size={16} />}
          title="Error loading model"
          color="red"
          variant="light"
        >
          {error instanceof Error ? error.message : 'Model not found'}
        </Alert>
      </Box>
    )
  }

  return (
    <Stack gap={0} style={{ height: '100vh', backgroundColor: '#FAFAFA' }}>
      {/* Header Section */}
      <Box px={32} pt={40} pb={24}>
        <Group gap={16} mb={20}>
          <Button
            variant="subtle"
            color="gray"
            size="sm"
            leftSection={<IconArrowLeft size={16} />}
            onClick={() => navigate(ROUTES.MODELS)}
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
            Back to Models
          </Button>
        </Group>

        {/* Model Header */}
        <Group justify="space-between" align="flex-start" mb={16}>
          <div>
            <Group gap={12} mb={8}>
              <Title order={1} size={32} fw={700}>
                {model.name}
              </Title>
              <Badge
                variant="light"
                color={taskColors[model.task] || 'gray'}
                styles={{
                  root: {
                    fontSize: '13px',
                    fontWeight: 500,
                    textTransform: 'capitalize',
                    paddingLeft: 10,
                    paddingRight: 10,
                  },
                }}
              >
                {model.task}
              </Badge>
              <Badge
                variant={
                  model.status === 'ready' || model.status === 'active'
                    ? 'light'
                    : 'outline'
                }
                color={statusColors[model.status] || 'gray'}
                styles={{
                  root: {
                    fontSize: '13px',
                    fontWeight: 500,
                    textTransform: 'capitalize',
                    paddingLeft: 10,
                    paddingRight: 10,
                  },
                }}
              >
                {model.status}
              </Badge>
            </Group>
            <Text size="15px" c="dimmed">
              {model.description || 'No description provided'}
            </Text>
          </div>

          <Group gap={12}>
            {(model.status === 'training' || model.status === 'queued') && (
              <Button
                variant="light"
                color="blue"
                leftSection={<IconChartBar size={18} />}
                onClick={() => navigate(`/models/${model.id}/training`)}
                styles={{
                  root: {
                    fontSize: '15px',
                    fontWeight: 500,
                  },
                }}
              >
                View Training
              </Button>
            )}
            {(model.status === 'failed' || model.status === 'ready') && job && (
              <Button
                variant="light"
                color="blue"
                leftSection={<IconRefresh size={18} />}
                onClick={handleRestartJob}
                loading={restartJobMutation.isPending}
                styles={{
                  root: {
                    fontSize: '15px',
                    fontWeight: 500,
                  },
                }}
              >
                Restart Training
              </Button>
            )}
            {model.status === 'ready' && (
              <>
                <Button
                  variant="light"
                  color="indigo"
                  leftSection={<IconDownload size={18} />}
                  styles={{
                    root: {
                      fontSize: '15px',
                      fontWeight: 500,
                    },
                  }}
                >
                  Download
                </Button>
                <Button
                  color="indigo"
                  leftSection={<IconRocket size={18} />}
                  styles={{
                    root: {
                      backgroundColor: COLORS.PRIMARY,
                      fontSize: '15px',
                      fontWeight: 500,
                    },
                  }}
                >
                  Deploy
                </Button>
              </>
            )}
          </Group>
        </Group>
      </Box>

      {/* Tabs Section */}
      <Box px={32} style={{ flex: 1, overflow: 'auto' }}>
        <Tabs value={activeTab} onChange={setActiveTab}>
          <Tabs.List
            styles={{
              list: {
                borderBottom: '1px solid #E5E7EB',
              },
            }}
          >
            <Tabs.Tab value="overview">Overview</Tabs.Tab>
            <Tabs.Tab value="evaluations">Evaluations</Tabs.Tab>
            <Tabs.Tab value="api">API Usage</Tabs.Tab>
            <Tabs.Tab value="versions">Versions</Tabs.Tab>
            <Tabs.Tab value="history">Training History</Tabs.Tab>
          </Tabs.List>

          {/* Overview Tab */}
          <Tabs.Panel value="overview" pt={24} pb={32}>
            <Stack gap={24}>
              {/* Summary Card */}
              <Card
                shadow="none"
                padding={24}
                radius={12}
                withBorder
                style={{
                  borderColor: '#E5E7EB',
                  backgroundColor: 'white',
                }}
              >
                <Group justify="space-between" mb={16}>
                  <Text size="16px" fw={600}>
                    Summary
                  </Text>
                  {!isEditingSummary && (
                    <Button
                      variant="subtle"
                      color="gray"
                      size="sm"
                      leftSection={<IconEdit size={16} />}
                      onClick={() => setIsEditingSummary(true)}
                      styles={{
                        root: {
                          fontSize: '14px',
                        },
                      }}
                    >
                      Edit
                    </Button>
                  )}
                </Group>

                {isEditingSummary ? (
                  <>
                    <Textarea
                      placeholder="Add a summary describing this model..."
                      value={summaryValue}
                      onChange={e => setSummaryValue(e.currentTarget.value)}
                      minRows={4}
                      autoFocus
                      styles={{
                        input: {
                          fontSize: '15px',
                          borderColor: '#E5E7EB',
                        },
                      }}
                    />
                    <Group justify="flex-end" mt={12} gap={8}>
                      <Button
                        variant="subtle"
                        color="gray"
                        size="sm"
                        leftSection={<IconX size={16} />}
                        onClick={() => {
                          setSummaryValue(model.description || '')
                          setIsEditingSummary(false)
                        }}
                        styles={{
                          root: {
                            fontSize: '14px',
                          },
                        }}
                      >
                        Cancel
                      </Button>
                      <Button
                        variant="light"
                        color="indigo"
                        size="sm"
                        leftSection={<IconCheck size={16} />}
                        loading={updateModelMutation.isPending}
                        onClick={async () => {
                          try {
                            await updateModelMutation.mutateAsync({
                              id: id!,
                              updates: { description: summaryValue },
                            })
                            notifications.show({
                              title: 'Success',
                              message: 'Model description updated successfully',
                              color: 'green',
                            })
                            setIsEditingSummary(false)
                          } catch (error) {
                            notifications.show({
                              title: 'Error',
                              message: 'Failed to update model description',
                              color: 'red',
                            })
                          }
                        }}
                        styles={{
                          root: {
                            fontSize: '14px',
                          },
                        }}
                      >
                        Save
                      </Button>
                    </Group>
                  </>
                ) : (
                  <Text size="15px" c={summaryValue ? 'dark' : 'dimmed'}>
                    {summaryValue ||
                      'No description provided. Click Edit to add one.'}
                  </Text>
                )}
              </Card>

              {/* Model Details Grid */}
              <Group
                align="flex-start"
                gap={24}
                style={{ alignItems: 'stretch' }}
              >
                {/* Model Information */}
                <Card
                  shadow="none"
                  padding={24}
                  radius={12}
                  withBorder
                  style={{
                    borderColor: '#E5E7EB',
                    backgroundColor: 'white',
                    flex: 1,
                  }}
                >
                  <Text size="16px" fw={600} mb={16}>
                    Model Information
                  </Text>
                  <Stack gap={12}>
                    <Box>
                      <Text size="13px" c="dimmed" mb={4}>
                        Framework
                      </Text>
                      <Text size="15px" fw={500}>
                        {model.framework}
                      </Text>
                    </Box>
                    <Divider />
                    <Box>
                      <Text size="13px" c="dimmed" mb={4}>
                        Version
                      </Text>
                      <Code
                        style={{
                          fontSize: '14px',
                          padding: '4px 8px',
                        }}
                      >
                        {model.version}
                      </Code>
                    </Box>
                    <Divider />
                    <Box>
                      <Text size="13px" c="dimmed" mb={4}>
                        Task Type
                      </Text>
                      <Text size="15px" fw={500} tt="capitalize">
                        {model.task}
                      </Text>
                    </Box>
                    <Divider />
                    <Box>
                      <Text size="13px" c="dimmed" mb={4}>
                        Accuracy
                      </Text>
                      <Text size="15px" fw={500}>
                        {model.accuracy
                          ? `${model.accuracy.toFixed(2)}%`
                          : 'N/A'}
                      </Text>
                    </Box>
                  </Stack>
                </Card>

                {/* Training Details */}
                <Card
                  shadow="none"
                  padding={24}
                  radius={12}
                  withBorder
                  style={{
                    borderColor: '#E5E7EB',
                    backgroundColor: 'white',
                    flex: 1,
                  }}
                >
                  <Text size="16px" fw={600} mb={16}>
                    Training Details
                  </Text>
                  <Stack gap={12}>
                    <Box>
                      <Text size="13px" c="dimmed" mb={4}>
                        Training Dataset
                      </Text>
                      <Group gap={8}>
                        <Text
                          size="15px"
                          fw={500}
                          style={{ color: COLORS.PRIMARY, cursor: 'pointer' }}
                          onClick={() => {
                            if (model.dataset_id) {
                              navigate(`/datasets/${model.dataset_id}`)
                            }
                          }}
                        >
                          {dataset?.name || model.dataset_id || 'Not linked'}
                        </Text>
                        {model.dataset_id && (
                          <ActionIcon
                            size="sm"
                            variant="subtle"
                            color="indigo"
                            onClick={() =>
                              navigate(`/datasets/${model.dataset_id}`)
                            }
                          >
                            <IconExternalLink size={14} />
                          </ActionIcon>
                        )}
                      </Group>
                    </Box>
                    <Divider />
                    <Box>
                      <Text size="13px" c="dimmed" mb={4}>
                        Created At
                      </Text>
                      <Text size="15px" fw={500}>
                        {formatDate(model.created_at)}
                      </Text>
                    </Box>
                    <Divider />
                    <Box>
                      <Text size="13px" c="dimmed" mb={4}>
                        Last Trained
                      </Text>
                      <Text size="15px" fw={500}>
                        {model.last_trained
                          ? formatDate(model.last_trained)
                          : 'Never'}
                      </Text>
                    </Box>
                    <Divider />
                    <Box>
                      <Text size="13px" c="dimmed" mb={4}>
                        Model Path
                      </Text>
                      <Code
                        style={{
                          fontSize: '13px',
                          padding: '4px 8px',
                          wordBreak: 'break-all',
                        }}
                      >
                        {model.model_path || 'Not available'}
                      </Code>
                    </Box>
                  </Stack>
                </Card>
              </Group>

              {/* Tags */}
              <Card
                shadow="none"
                padding={24}
                radius={12}
                withBorder
                style={{
                  borderColor: '#E5E7EB',
                  backgroundColor: 'white',
                }}
              >
                <Text size="16px" fw={600} mb={16}>
                  Tags
                </Text>
                <Group gap={8}>
                  {model.tags.length > 0 ? (
                    model.tags.map(tag => (
                      <Badge
                        key={tag}
                        variant="light"
                        color="gray"
                        styles={{
                          root: {
                            fontSize: '13px',
                            fontWeight: 400,
                            textTransform: 'none',
                            backgroundColor: '#F3F4F6',
                            color: '#6B7280',
                          },
                        }}
                      >
                        {tag}
                      </Badge>
                    ))
                  ) : (
                    <Text size="14px" c="dimmed">
                      No tags added
                    </Text>
                  )}
                </Group>
              </Card>
            </Stack>
          </Tabs.Panel>

          {/* Evaluations Tab */}
          <Tabs.Panel value="evaluations" pt={24} pb={32}>
            <Card
              shadow="none"
              padding={24}
              radius={12}
              withBorder
              style={{
                borderColor: '#E5E7EB',
                backgroundColor: 'white',
              }}
            >
              <Text size="16px" fw={600} mb={16}>
                Model Evaluations
              </Text>
              <Stack gap={24}>
                {/* Metrics Summary */}
                <Group gap={24}>
                  <Box style={{ flex: 1 }}>
                    <Card
                      padding={16}
                      radius={8}
                      style={{ backgroundColor: '#F9FAFB' }}
                    >
                      <Text size="13px" c="dimmed" mb={4}>
                        Accuracy
                      </Text>
                      <Text size="24px" fw={700}>
                        {model.accuracy
                          ? `${model.accuracy.toFixed(2)}%`
                          : 'N/A'}
                      </Text>
                    </Card>
                  </Box>
                  <Box style={{ flex: 1 }}>
                    <Card
                      padding={16}
                      radius={8}
                      style={{ backgroundColor: '#F9FAFB' }}
                    >
                      <Text size="13px" c="dimmed" mb={4}>
                        Precision
                      </Text>
                      <Text size="24px" fw={700}>
                        {model.metrics?.precision
                          ? `${model.metrics.precision.toFixed(2)}%`
                          : 'N/A'}
                      </Text>
                    </Card>
                  </Box>
                  <Box style={{ flex: 1 }}>
                    <Card
                      padding={16}
                      radius={8}
                      style={{ backgroundColor: '#F9FAFB' }}
                    >
                      <Text size="13px" c="dimmed" mb={4}>
                        Recall
                      </Text>
                      <Text size="24px" fw={700}>
                        {model.metrics?.recall
                          ? `${model.metrics.recall.toFixed(2)}%`
                          : 'N/A'}
                      </Text>
                    </Card>
                  </Box>
                  <Box style={{ flex: 1 }}>
                    <Card
                      padding={16}
                      radius={8}
                      style={{ backgroundColor: '#F9FAFB' }}
                    >
                      <Text size="13px" c="dimmed" mb={4}>
                        F1 Score
                      </Text>
                      <Text size="24px" fw={700}>
                        {model.metrics?.f1_score
                          ? `${model.metrics.f1_score.toFixed(2)}%`
                          : 'N/A'}
                      </Text>
                    </Card>
                  </Box>
                </Group>

                <Divider />

                {/* Evaluation History */}
                <Box>
                  <Text size="15px" fw={600} mb={12}>
                    Evaluation History
                  </Text>
                  <Text size="14px" c="dimmed">
                    No evaluation history available yet. Run evaluations to see
                    results here.
                  </Text>
                </Box>
              </Stack>
            </Card>
          </Tabs.Panel>

          {/* API Usage Tab */}
          <Tabs.Panel value="api" pt={24} pb={32}>
            <Card
              shadow="none"
              padding={24}
              radius={12}
              withBorder
              style={{
                borderColor: '#E5E7EB',
                backgroundColor: 'white',
              }}
            >
              <Text size="16px" fw={600} mb={16}>
                API Usage
              </Text>
              <Stack gap={20}>
                <Box>
                  <Group justify="space-between" mb={8}>
                    <Text size="14px" fw={500}>
                      REST API Endpoint
                    </Text>
                    <CopyButton
                      value={`http://localhost:8000/api/models/${model.id}/predict`}
                    >
                      {({ copied, copy }) => (
                        <Tooltip label={copied ? 'Copied!' : 'Copy'}>
                          <ActionIcon
                            variant="subtle"
                            color={copied ? 'green' : 'gray'}
                            onClick={copy}
                          >
                            {copied ? (
                              <IconCheck size={16} />
                            ) : (
                              <IconCopy size={16} />
                            )}
                          </ActionIcon>
                        </Tooltip>
                      )}
                    </CopyButton>
                  </Group>
                  <Code
                    block
                    style={{
                      fontSize: '13px',
                      padding: '12px',
                      backgroundColor: '#F9FAFB',
                    }}
                  >
                    {`POST http://localhost:8000/api/models/${model.id}/predict`}
                  </Code>
                </Box>

                <Box>
                  <Group justify="space-between" mb={8}>
                    <Text size="14px" fw={500}>
                      Python Example
                    </Text>
                    <CopyButton
                      value={`import requests\n\nresponse = requests.post(\n    "http://localhost:8000/api/models/${model.id}/predict",\n    json={"data": your_data}\n)\nprint(response.json())`}
                    >
                      {({ copied, copy }) => (
                        <Tooltip label={copied ? 'Copied!' : 'Copy'}>
                          <ActionIcon
                            variant="subtle"
                            color={copied ? 'green' : 'gray'}
                            onClick={copy}
                          >
                            {copied ? (
                              <IconCheck size={16} />
                            ) : (
                              <IconCopy size={16} />
                            )}
                          </ActionIcon>
                        </Tooltip>
                      )}
                    </CopyButton>
                  </Group>
                  <Code
                    block
                    style={{
                      fontSize: '13px',
                      padding: '12px',
                      backgroundColor: '#F9FAFB',
                    }}
                  >
                    {`import requests

response = requests.post(
    "http://localhost:8000/api/models/${model.id}/predict",
    json={"data": your_data}
)
print(response.json())`}
                  </Code>
                </Box>

                <Divider />

                <Box>
                  <Text size="15px" fw={600} mb={12}>
                    Recent API Calls
                  </Text>
                  <Text size="14px" c="dimmed">
                    No API usage history available yet.
                  </Text>
                </Box>
              </Stack>
            </Card>
          </Tabs.Panel>

          {/* Versions Tab */}
          <Tabs.Panel value="versions" pt={24} pb={32}>
            <Card
              shadow="none"
              padding={24}
              radius={12}
              withBorder
              style={{
                borderColor: '#E5E7EB',
                backgroundColor: 'white',
              }}
            >
              <Text size="16px" fw={600} mb={16}>
                Version History
              </Text>
              <Table
                horizontalSpacing="lg"
                verticalSpacing="md"
                styles={{
                  th: {
                    fontSize: '12px',
                    fontWeight: 600,
                    color: '#6B7280',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    borderBottom: '1px solid #E5E7EB',
                    paddingTop: 12,
                    paddingBottom: 12,
                    backgroundColor: '#F9FAFB',
                  },
                  td: {
                    fontSize: '14px',
                    borderBottom: '1px solid #F3F4F6',
                    paddingTop: 12,
                    paddingBottom: 12,
                  },
                }}
              >
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>VERSION</Table.Th>
                    <Table.Th>STATUS</Table.Th>
                    <Table.Th>ACCURACY</Table.Th>
                    <Table.Th>CREATED</Table.Th>
                    <Table.Th>ACTIONS</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  <Table.Tr>
                    <Table.Td>
                      <Code>{model.version}</Code>
                    </Table.Td>
                    <Table.Td>
                      <Badge variant="light" color="green" size="sm">
                        Current
                      </Badge>
                    </Table.Td>
                    <Table.Td>
                      {model.accuracy ? `${model.accuracy.toFixed(2)}%` : 'N/A'}
                    </Table.Td>
                    <Table.Td>{formatDate(model.created_at)}</Table.Td>
                    <Table.Td>
                      <Button variant="subtle" size="xs" color="gray">
                        View
                      </Button>
                    </Table.Td>
                  </Table.Tr>
                </Table.Tbody>
              </Table>
            </Card>
          </Tabs.Panel>

          {/* Training History Tab */}
          <Tabs.Panel value="history" pt={24} pb={32}>
            <Card
              shadow="none"
              padding={24}
              radius={12}
              withBorder
              style={{
                borderColor: '#E5E7EB',
                backgroundColor: 'white',
              }}
            >
              <Group justify="space-between" mb={16}>
                <Group gap={8}>
                  <IconHistory size={20} color="#374151" />
                  <Text size="16px" fw={600}>
                    Training History
                  </Text>
                </Group>
                <Button
                  variant="subtle"
                  size="sm"
                  color="gray"
                  onClick={() => refetchHistory()}
                >
                  Refresh
                </Button>
              </Group>
              <Table
                horizontalSpacing="lg"
                verticalSpacing="md"
                styles={{
                  th: {
                    fontSize: '12px',
                    fontWeight: 600,
                    color: '#6B7280',
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    borderBottom: '1px solid #E5E7EB',
                    paddingTop: 12,
                    paddingBottom: 12,
                    backgroundColor: '#F9FAFB',
                  },
                  td: {
                    fontSize: '14px',
                    borderBottom: '1px solid #F3F4F6',
                    paddingTop: 12,
                    paddingBottom: 12,
                  },
                  tr: {
                    cursor: 'pointer',
                    transition: 'background-color 0.2s ease',
                    '&:hover': {
                      backgroundColor: '#F9FAFB',
                    },
                  },
                }}
              >
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>RUN ID</Table.Th>
                    <Table.Th>STATUS</Table.Th>
                    <Table.Th>DURATION</Table.Th>
                    <Table.Th>FINAL LOSS</Table.Th>
                    <Table.Th>FINAL ACCURACY</Table.Th>
                    <Table.Th>DATASET</Table.Th>
                    <Table.Th>STARTED</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {isLoadingHistory ? (
                    <Table.Tr>
                      <Table.Td colSpan={7}>
                        <Center py={40}>
                          <Loader size="md" color={COLORS.PRIMARY} />
                        </Center>
                      </Table.Td>
                    </Table.Tr>
                  ) : !trainingHistory || trainingHistory.length === 0 ? (
                    <Table.Tr>
                      <Table.Td colSpan={7}>
                        <Center py={40}>
                          <Stack gap={12} align="center">
                            <IconHistory size={32} color="#9CA3AF" />
                            <Text size="14px" c="dimmed">
                              No training history available
                            </Text>
                          </Stack>
                        </Center>
                      </Table.Td>
                    </Table.Tr>
                  ) : (
                    trainingHistory.map(run => {
                      const StatusIcon =
                        trainingStatusIcons[run.status] || IconClock
                      const statusColor =
                        trainingStatusColors[run.status] || 'gray'

                      return (
                        <Table.Tr
                          key={run.id}
                          onClick={() =>
                            navigate(
                              `/models/${model.id}/training?run=${run.id}`
                            )
                          }
                        >
                          <Table.Td>
                            <Code style={{ fontSize: '13px' }}>
                              {run.run_number
                                ? `run_${run.run_number.toString().padStart(3, '0')}`
                                : run.id.slice(0, 8)}
                            </Code>
                          </Table.Td>
                          <Table.Td>
                            <Group gap={8}>
                              <StatusIcon
                                size={16}
                                color={
                                  COLORS[
                                    statusColor.toUpperCase() as keyof typeof COLORS
                                  ] || '#6B7280'
                                }
                              />
                              <Badge
                                variant="light"
                                color={statusColor}
                                size="sm"
                              >
                                {run.status.charAt(0).toUpperCase() +
                                  run.status.slice(1)}
                              </Badge>
                            </Group>
                          </Table.Td>
                          <Table.Td>
                            <Group gap={6}>
                              <IconClock size={14} color="#6B7280" />
                              <Text size="14px">
                                {formatDuration(run.duration_seconds)}
                              </Text>
                            </Group>
                          </Table.Td>
                          <Table.Td>
                            <Text
                              size="14px"
                              fw={500}
                              c={run.final_loss ? undefined : 'dimmed'}
                            >
                              {run.final_loss != null
                                ? run.final_loss.toFixed(4)
                                : 'N/A'}
                            </Text>
                          </Table.Td>
                          <Table.Td>
                            <Text
                              size="14px"
                              fw={600}
                              c={
                                run.final_accuracy != null
                                  ? run.final_accuracy * 100 > 80
                                    ? COLORS.SUCCESS
                                    : undefined
                                  : 'dimmed'
                              }
                            >
                              {run.final_accuracy != null
                                ? `${(run.final_accuracy * 100).toFixed(2)}%`
                                : 'N/A'}
                            </Text>
                          </Table.Td>
                          <Table.Td>
                            <Text
                              size="14px"
                              style={{
                                color: COLORS.PRIMARY,
                                cursor: 'pointer',
                              }}
                              onClick={e => {
                                e.stopPropagation()
                                if (run.dataset_id) {
                                  navigate(`/datasets/${run.dataset_id}`)
                                }
                              }}
                            >
                              {dataset?.name ||
                                run.dataset_id?.slice(0, 8) ||
                                'N/A'}
                            </Text>
                          </Table.Td>
                          <Table.Td>
                            <Text size="14px">
                              {run.started_at
                                ? formatDate(run.started_at)
                                : run.created_at
                                  ? formatDate(run.created_at)
                                  : 'N/A'}
                            </Text>
                          </Table.Td>
                        </Table.Tr>
                      )
                    })
                  )}
                </Table.Tbody>
              </Table>
            </Card>
          </Tabs.Panel>
        </Tabs>
      </Box>
    </Stack>
  )
}
