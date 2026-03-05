import {
  Alert,
  Anchor,
  Badge,
  Box,
  Breadcrumbs,
  Button,
  Card,
  Collapse,
  Group,
  Loader,
  Stack,
  Text,
  Title,
} from '@mantine/core'
import {
  IconAlertCircle,
  IconChevronDown,
  IconChevronRight,
  IconChevronUp,
  IconDatabase,
} from '@tabler/icons-react'
import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { getDatasetColumns } from '../../shared/api/datasets'
import {
  BADGE_STYLES,
  DOMAIN_COLORS,
  READINESS_COLORS,
} from '../../shared/constants/theme'
import { useDataset } from '../../shared/hooks/useDatasets'
import { formatters } from '../../shared/utils/formatters'

interface BreadcrumbItem {
  title: string
  onClick?: () => void
}

function DatasetBreadcrumbs({ items }: { items: BreadcrumbItem[] }) {
  return (
    <Breadcrumbs separator={<IconChevronRight size={14} color="#9CA3AF" />}>
      {items.map((item, index) => {
        const isCurrentPage = !item.onClick

        if (isCurrentPage) {
          return (
            <Text key={index} size="14px" c="#6B7280" fw={500}>
              {item.title}
            </Text>
          )
        }

        return (
          <Anchor
            key={index}
            onClick={item.onClick}
            style={{
              cursor: 'pointer',
              color: '#6366F1',
              textDecoration: 'none',
              fontSize: '14px',
            }}
          >
            {item.title}
          </Anchor>
        )
      })}
    </Breadcrumbs>
  )
}

function LoadingState() {
  return (
    <Stack gap={0} style={{ height: '100vh', backgroundColor: '#FAFAFA' }}>
      <Box px={32} pt={40}>
        <Box
          style={{
            backgroundColor: 'white',
            borderRadius: 12,
            border: '1px solid #E5E7EB',
            padding: 60,
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
          }}
        >
          <Stack align="center" gap={16}>
            <Loader size="lg" color="indigo" />
            <Text size="15px" c="dimmed">
              Loading dataset details...
            </Text>
          </Stack>
        </Box>
      </Box>
    </Stack>
  )
}

function ErrorState({
  error,
  onBack,
}: {
  error: Error | null
  onBack: () => void
}) {
  return (
    <Stack gap={0} style={{ height: '100vh', backgroundColor: '#FAFAFA' }}>
      <Box px={32} pt={40}>
        <Alert
          icon={<IconAlertCircle size={16} />}
          title="Error loading dataset"
          color="red"
        >
          {error instanceof Error ? error.message : 'Dataset not found'}
        </Alert>
        <Button mt={16} onClick={onBack}>
          Back to Datasets
        </Button>
      </Box>
    </Stack>
  )
}

interface InfoRowProps {
  label: string
  value: React.ReactNode
}

function InfoRow({ label, value }: InfoRowProps) {
  return (
    <Group justify="space-between">
      <Text size="14px" c="dimmed">
        {label}
      </Text>
      {typeof value === 'string' ? (
        <Text size="14px" fw={500}>
          {value}
        </Text>
      ) : (
        value
      )}
    </Group>
  )
}

interface InfoCardProps {
  title: string
  children: React.ReactNode
}

function InfoCard({ title, children }: InfoCardProps) {
  return (
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
      <Title order={3} size={18} fw={600} mb={20}>
        {title}
      </Title>
      {children}
    </Card>
  )
}

export function DatasetDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const { data: dataset, isLoading, isError, error } = useDataset(id || '')

  // State for tabular dataset schema
  const [schemaData, setSchemaData] = useState<{
    columns: string[]
    total_rows: number
    shape: { rows: number; columns: number }
    dtypes: Record<string, string>
  } | null>(null)
  const [loadingSchema, setLoadingSchema] = useState(false)
  const [showAllColumns, setShowAllColumns] = useState(false)

  // Fetch schema for tabular datasets
  useEffect(() => {
    if (dataset && dataset.domain === 'tabular' && id) {
      setLoadingSchema(true)
      getDatasetColumns(id)
        .then(data => {
          setSchemaData({
            columns: data.columns,
            total_rows: data.total_rows,
            shape: data.shape,
            dtypes: data.dtypes,
          })
        })
        .catch(err => {
          console.error('Failed to load schema:', err)
        })
        .finally(() => {
          setLoadingSchema(false)
        })
    }
  }, [dataset, id])

  const breadcrumbItems: BreadcrumbItem[] = [
    { title: 'Datasets', onClick: () => navigate('/datasets') },
    { title: dataset?.name || 'Loading...' },
  ]

  if (isLoading) {
    return <LoadingState />
  }

  if (isError || !dataset) {
    return <ErrorState error={error} onBack={() => navigate('/datasets')} />
  }

  return (
    <Stack gap={0} style={{ minHeight: '100vh', backgroundColor: '#FAFAFA' }}>
      {/* Breadcrumbs */}
      <Box px={32} pt={24} pb={16}>
        <DatasetBreadcrumbs items={breadcrumbItems} />
      </Box>

      {/* Header */}
      <Box px={32} pb={24}>
        <Group gap={16} align="flex-start">
          <Box
            style={{
              width: 48,
              height: 48,
              borderRadius: 8,
              backgroundColor: '#D4F4DD',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <IconDatabase size={24} color="#22C55E" />
          </Box>
          <Stack gap={4} style={{ flex: 1 }}>
            <Title order={1} size={28} fw={700}>
              {dataset.name}
            </Title>
            {dataset.description && (
              <Text size="15px" c="dimmed">
                {dataset.description}
              </Text>
            )}
          </Stack>
        </Group>
      </Box>

      {/* Content */}
      <Box px={32} pb={32}>
        <Stack gap={24}>
          {/* Overview Card */}
          <InfoCard title="Overview">
            <Stack gap={16}>
              <InfoRow
                label="Domain"
                value={
                  <Badge
                    variant="light"
                    color={DOMAIN_COLORS[dataset.domain] || 'gray'}
                    styles={BADGE_STYLES.domain}
                  >
                    {formatters.domain(dataset.domain)}
                  </Badge>
                }
              />
              <InfoRow
                label="Readiness"
                value={
                  <Badge
                    variant="light"
                    color={READINESS_COLORS[dataset.readiness] || 'gray'}
                    styles={BADGE_STYLES.readiness}
                  >
                    {formatters.readiness(dataset.readiness)}
                  </Badge>
                }
              />
              <InfoRow
                label="Storage"
                value={
                  <Badge
                    variant="light"
                    color="gray"
                    styles={BADGE_STYLES.storage}
                  >
                    {formatters.storage(dataset.storage)}
                  </Badge>
                }
              />
              <InfoRow label="Files" value={formatters.number(dataset.size)} />

              {/* Data Shape for Tabular Datasets */}
              {dataset.domain === 'tabular' && schemaData && (
                <InfoRow
                  label="Data Shape"
                  value={
                    <Text size="14px" fw={500}>
                      {formatters.number(schemaData.shape.rows)} rows ×{' '}
                      {schemaData.shape.columns} columns
                    </Text>
                  }
                />
              )}

              <InfoRow
                label="Path"
                value={
                  <Text
                    size="14px"
                    fw={500}
                    style={{ fontFamily: 'monospace' }}
                  >
                    {dataset.path}
                  </Text>
                }
              />
            </Stack>
          </InfoCard>

          {/* Data Schema Card for Tabular Datasets */}
          {dataset.domain === 'tabular' && (
            <InfoCard title="Data Schema">
              {loadingSchema ? (
                <Group justify="center" py="md">
                  <Loader size="sm" />
                  <Text size="sm" c="dimmed">
                    Loading schema...
                  </Text>
                </Group>
              ) : schemaData ? (
                <Stack gap={16}>
                  {/* Target Column */}
                  {dataset.metadata?.target_column && (
                    <InfoRow
                      label="Target Column"
                      value={
                        <Badge variant="light" color="indigo" size="md">
                          {dataset.metadata.target_column}
                        </Badge>
                      }
                    />
                  )}

                  <InfoRow
                    label="Total Columns"
                    value={
                      <Group gap={8}>
                        <Text size="14px" fw={500}>
                          {schemaData.columns.length}
                        </Text>
                        <Button
                          variant="subtle"
                          size="xs"
                          rightSection={
                            showAllColumns ? (
                              <IconChevronUp size={14} />
                            ) : (
                              <IconChevronDown size={14} />
                            )
                          }
                          onClick={() => setShowAllColumns(!showAllColumns)}
                        >
                          {showAllColumns ? 'Hide' : 'Show all'}
                        </Button>
                      </Group>
                    }
                  />

                  <Collapse in={showAllColumns}>
                    <Box
                      p={12}
                      style={{
                        backgroundColor: '#F9FAFB',
                        borderRadius: 8,
                        border: '1px solid #E5E7EB',
                      }}
                    >
                      <Stack gap={8}>
                        {schemaData.columns.map((col, idx) => (
                          <Group key={idx} justify="space-between">
                            <Text
                              size="13px"
                              style={{ fontFamily: 'monospace' }}
                            >
                              {col}
                            </Text>
                            <Badge size="xs" variant="light" color="gray">
                              {schemaData.dtypes[col]}
                            </Badge>
                          </Group>
                        ))}
                      </Stack>
                    </Box>
                  </Collapse>

                  <InfoRow
                    label="Total Rows"
                    value={formatters.number(schemaData.total_rows)}
                  />
                </Stack>
              ) : (
                <Text size="sm" c="dimmed">
                  Failed to load schema information
                </Text>
              )}
            </InfoCard>
          )}

          {/* Tags Card */}
          {dataset.tags?.length > 0 && (
            <InfoCard title="Tags">
              <Group gap={8}>
                {dataset.tags.map(tag => (
                  <Badge
                    key={tag}
                    variant="light"
                    color="gray"
                    styles={BADGE_STYLES.tag}
                  >
                    {tag}
                  </Badge>
                ))}
              </Group>
            </InfoCard>
          )}

          {/* Metadata Card */}
          <InfoCard title="Metadata">
            <Stack gap={16}>
              <InfoRow
                label="Created"
                value={formatters.dateTime(dataset.created_at)}
              />
              <InfoRow
                label="Last Updated"
                value={formatters.dateTime(dataset.updated_at)}
              />
              {dataset.freshness && (
                <InfoRow
                  label="Freshness"
                  value={formatters.dateTime(dataset.freshness)}
                />
              )}
              <InfoRow
                label="Dataset ID"
                value={
                  <Text
                    size="13px"
                    fw={500}
                    c="dimmed"
                    style={{ fontFamily: 'monospace' }}
                  >
                    {dataset.id}
                  </Text>
                }
              />
            </Stack>
          </InfoCard>
        </Stack>
      </Box>
    </Stack>
  )
}
