import type { Model } from '@/shared/api/models'
import { BrainIcon } from '@/shared/components/BrainIcon'
import { useDeleteModel, useModels } from '@/shared/hooks/useModels'
import { formatDate } from '@/shared/utils/formatters'
import {
  ActionIcon,
  Alert,
  Badge,
  Box,
  Button,
  Center,
  Group,
  Loader,
  Modal,
  Select,
  Stack,
  Table,
  Text,
  TextInput,
  Title,
  Tooltip,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import {
  IconAlertCircle,
  IconPlus,
  IconSearch,
  IconTrash,
} from '@tabler/icons-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { TrainModelModal } from './TrainModelModal'
import { TrainingProgress } from './TrainingProgress'

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

export function ModelsPage() {
  const navigate = useNavigate()
  const [searchQuery, setSearchQuery] = useState('')
  const [taskFilter, setTaskFilter] = useState<string | null>(null)
  const [statusFilter, setStatusFilter] = useState<string | null>(null)
  const [deleteModalOpen, setDeleteModalOpen] = useState(false)
  const [modelToDelete, setModelToDelete] = useState<Model | null>(null)
  const [trainModalOpen, setTrainModalOpen] = useState(false)

  // Fetch models with filters
  const {
    data: models = [],
    isLoading,
    error,
  } = useModels({
    search: searchQuery || undefined,
    task:
      taskFilter && taskFilter !== 'All Tasks'
        ? taskFilter.toLowerCase()
        : undefined,
    status:
      statusFilter && statusFilter !== 'All Status'
        ? statusFilter.toLowerCase()
        : undefined,
  })

  // Delete model mutation
  const deleteModelMutation = useDeleteModel()

  // Helper function to check if model can be deleted
  const canDeleteModel = (status: string) => {
    return ['queued', 'ready', 'draft', 'failed', 'stopped'].includes(status)
  }

  // Handle delete button click
  const handleDeleteClick = (model: Model, e: React.MouseEvent) => {
    e.stopPropagation()
    if (canDeleteModel(model.status)) {
      setModelToDelete(model)
      setDeleteModalOpen(true)
    }
  }

  // Confirm delete
  const handleConfirmDelete = async () => {
    if (!modelToDelete) return

    try {
      await deleteModelMutation.mutateAsync({
        id: modelToDelete.id,
        deleteFiles: true,
      })

      notifications.show({
        title: 'Model Deleted',
        message: `Successfully deleted model "${modelToDelete.name}"`,
        color: 'green',
      })

      setDeleteModalOpen(false)
      setModelToDelete(null)
    } catch (error) {
      notifications.show({
        title: 'Delete Failed',
        message:
          error instanceof Error ? error.message : 'Failed to delete model',
        color: 'red',
      })
    }
  }

  return (
    <Stack gap={0} style={{ height: '100vh', backgroundColor: '#FAFAFA' }}>
      {/* Header Section */}
      <Box px={32} pt={40} pb={24}>
        <Group justify="space-between" align="flex-start">
          <div>
            <Title order={1} size={32} fw={700} mb={8}>
              Models
            </Title>
            <Text size="15px" c="dimmed">
              Registry of trainable and evaluable models, experiments, and
              versions
            </Text>
          </div>
          <Button
            leftSection={<IconPlus size={18} />}
            color="indigo"
            size="md"
            onClick={() => setTrainModalOpen(true)}
            styles={{
              root: {
                backgroundColor: '#6366F1',
                fontSize: '15px',
                fontWeight: 500,
                paddingLeft: 20,
                paddingRight: 24,
                height: 40,
              },
            }}
          >
            Train Model
          </Button>
        </Group>
      </Box>

      {/* Filters */}
      <Box px={32} pb={24}>
        <Group gap={12}>
          <TextInput
            placeholder="Search models..."
            leftSection={<IconSearch size={16} />}
            value={searchQuery}
            onChange={event => setSearchQuery(event.currentTarget.value)}
            style={{ flex: 1 }}
            styles={{
              input: {
                fontSize: '15px',
                borderColor: '#E5E7EB',
                backgroundColor: 'white',
              },
            }}
          />
          <Select
            placeholder="All Tasks"
            data={[
              'All Tasks',
              'Classification',
              'Regression',
              'Clustering',
              'Detection',
            ]}
            value={taskFilter}
            onChange={setTaskFilter}
            clearable
            styles={{
              input: {
                fontSize: '15px',
                borderColor: '#E5E7EB',
                backgroundColor: 'white',
                minWidth: 160,
              },
            }}
          />
          <Select
            placeholder="All Status"
            data={[
              'All Status',
              'Active',
              'Ready',
              'Training',
              'Queued',
              'Draft',
              'Failed',
            ]}
            value={statusFilter}
            onChange={setStatusFilter}
            clearable
            styles={{
              input: {
                fontSize: '15px',
                borderColor: '#E5E7EB',
                backgroundColor: 'white',
                minWidth: 160,
              },
            }}
          />
        </Group>
      </Box>

      {/* Table */}
      <Box px={32}>
        <Box
          style={{
            backgroundColor: 'white',
            borderRadius: 12,
            border: '1px solid #E5E7EB',
            overflow: 'hidden',
          }}
        >
          <Table
            horizontalSpacing="xl"
            verticalSpacing="md"
            styles={{
              th: {
                fontSize: '12px',
                fontWeight: 600,
                color: '#6B7280',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
                borderBottom: '1px solid #E5E7EB',
                paddingTop: 16,
                paddingBottom: 16,
                backgroundColor: '#F9FAFB',
              },
              td: {
                fontSize: '15px',
                borderBottom: '1px solid #F3F4F6',
                paddingTop: 16,
                paddingBottom: 16,
              },
              tr: {
                transition: 'background-color 0.15s ease',
              },
            }}
          >
            <Table.Thead>
              <Table.Tr>
                <Table.Th>NAME</Table.Th>
                <Table.Th>TASK</Table.Th>
                <Table.Th>LATEST VERSION</Table.Th>
                <Table.Th>STATUS</Table.Th>
                <Table.Th>LAST UPDATED</Table.Th>
                <Table.Th>TAGS</Table.Th>
                <Table.Th>ACTIONS</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {isLoading ? (
                <Table.Tr>
                  <Table.Td colSpan={7}>
                    <Center py="xl">
                      <Loader size="md" color="#6366F1" />
                    </Center>
                  </Table.Td>
                </Table.Tr>
              ) : error ? (
                <Table.Tr>
                  <Table.Td colSpan={7}>
                    <Center py="xl">
                      <Alert
                        icon={<IconAlertCircle size={16} />}
                        title="Error loading models"
                        color="red"
                        variant="light"
                      >
                        {error instanceof Error
                          ? error.message
                          : 'Failed to fetch models'}
                      </Alert>
                    </Center>
                  </Table.Td>
                </Table.Tr>
              ) : models.length > 0 ? (
                models.map(model => (
                  <Table.Tr
                    key={model.id}
                    style={{ cursor: 'pointer' }}
                    onClick={() => navigate(`/models/${model.id}`)}
                    onMouseEnter={e => {
                      e.currentTarget.style.backgroundColor = '#FAFAFA'
                    }}
                    onMouseLeave={e => {
                      e.currentTarget.style.backgroundColor = 'transparent'
                    }}
                  >
                    <Table.Td>
                      <Group gap={12}>
                        <Box
                          style={{
                            width: 32,
                            height: 32,
                            borderRadius: 6,
                            backgroundColor: '#EEF2FF',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                          }}
                        >
                          <BrainIcon size={16} />
                        </Box>
                        <Text size="15px" fw={500}>
                          {model.name}
                        </Text>
                      </Group>
                    </Table.Td>
                    <Table.Td>
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
                    </Table.Td>
                    <Table.Td>
                      <Text
                        size="15px"
                        fw={500}
                        style={{ fontFamily: 'monospace' }}
                      >
                        {model.version}
                      </Text>
                    </Table.Td>
                    <Table.Td>
                      <TrainingProgress model={model} />
                    </Table.Td>
                    <Table.Td>
                      <Text size="15px" c="dimmed">
                        {model.last_trained
                          ? formatDate(model.last_trained)
                          : formatDate(model.updated_at)}
                      </Text>
                    </Table.Td>
                    <Table.Td>
                      <Group gap={6}>
                        {model.tags.map(tag => (
                          <Badge
                            key={tag}
                            variant="light"
                            color="gray"
                            styles={{
                              root: {
                                fontSize: '12px',
                                fontWeight: 400,
                                textTransform: 'none',
                                backgroundColor: '#F3F4F6',
                                color: '#6B7280',
                              },
                            }}
                          >
                            {tag}
                          </Badge>
                        ))}
                      </Group>
                    </Table.Td>
                    <Table.Td>
                      <Tooltip
                        label={
                          canDeleteModel(model.status)
                            ? 'Delete model'
                            : 'Cannot delete model while active or training'
                        }
                        position="left"
                      >
                        <ActionIcon
                          variant="subtle"
                          color="red"
                          size="lg"
                          disabled={!canDeleteModel(model.status)}
                          onClick={e => handleDeleteClick(model, e)}
                          styles={{
                            root: {
                              '&:disabled': {
                                backgroundColor: 'transparent',
                                opacity: 0.4,
                              },
                            },
                          }}
                        >
                          <IconTrash size={18} />
                        </ActionIcon>
                      </Tooltip>
                    </Table.Td>
                  </Table.Tr>
                ))
              ) : (
                <Table.Tr>
                  <Table.Td colSpan={7}>
                    <Text ta="center" c="dimmed" size="15px" py="xl">
                      No models found
                    </Text>
                  </Table.Td>
                </Table.Tr>
              )}
            </Table.Tbody>
          </Table>
        </Box>
      </Box>

      {/* Train Model Modal */}
      <TrainModelModal
        opened={trainModalOpen}
        onClose={() => setTrainModalOpen(false)}
      />

      {/* Delete Confirmation Modal */}
      <Modal
        opened={deleteModalOpen}
        onClose={() => {
          setDeleteModalOpen(false)
          setModelToDelete(null)
        }}
        title="Delete Model"
        centered
        styles={{
          title: {
            fontSize: '18px',
            fontWeight: 600,
          },
        }}
      >
        <Stack gap={16}>
          <Text size="15px">
            Are you sure you want to delete the model{' '}
            <Text component="span" fw={600}>
              {modelToDelete?.name}
            </Text>
            ? This action cannot be undone.
          </Text>
          <Group justify="flex-end" gap={12}>
            <Button
              variant="light"
              color="gray"
              onClick={() => {
                setDeleteModalOpen(false)
                setModelToDelete(null)
              }}
            >
              Cancel
            </Button>
            <Button
              color="red"
              loading={deleteModelMutation.isPending}
              onClick={handleConfirmDelete}
            >
              Delete
            </Button>
          </Group>
        </Stack>
      </Modal>
    </Stack>
  )
}
