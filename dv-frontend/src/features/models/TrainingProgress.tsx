import type { Model } from '@/shared/api/models'
import { Badge, Group, Progress, Stack, Text } from '@mantine/core'

interface TrainingProgressProps {
  model: Model
}

const statusColors: Record<string, string> = {
  active: 'green',
  ready: 'green',
  training: 'blue',
  queued: 'cyan',
  draft: 'orange',
  failed: 'red',
  stopped: 'orange',
}

export function TrainingProgress({ model }: TrainingProgressProps) {
  const {
    status,
    current_iteration,
    total_iterations,
    current_accuracy,
    best_accuracy,
  } = model

  // Show progress bar for training models
  if (
    status === 'training' &&
    current_iteration !== undefined &&
    total_iterations !== undefined
  ) {
    const progressPercent = (current_iteration / total_iterations) * 100

    return (
      <Stack gap={6}>
        <Group gap={8}>
          <Badge
            variant="outline"
            color="blue"
            styles={{
              root: {
                fontSize: '13px',
                fontWeight: 500,
                textTransform: 'capitalize',
              },
            }}
          >
            Training
          </Badge>
          <Text size="13px" c="dimmed">
            {current_iteration}/{total_iterations}
          </Text>
        </Group>
        <div style={{ width: '100%' }}>
          <Progress
            value={progressPercent}
            size="sm"
            radius="sm"
            color="blue"
            styles={{
              root: {
                width: 150,
              },
            }}
          />
        </div>
        {(current_accuracy !== undefined || best_accuracy !== undefined) && (
          <Group gap={8}>
            {current_accuracy !== undefined && (
              <Text size="12px" c="dimmed">
                Current: {(current_accuracy * 100).toFixed(1)}%
              </Text>
            )}
            {best_accuracy !== undefined && (
              <Text size="12px" c="blue" fw={500}>
                Best: {(best_accuracy * 100).toFixed(1)}%
              </Text>
            )}
          </Group>
        )}
      </Stack>
    )
  }

  // Show queued status
  if (status === 'queued') {
    return (
      <Badge
        variant="outline"
        color="cyan"
        styles={{
          root: {
            fontSize: '13px',
            fontWeight: 500,
            textTransform: 'capitalize',
          },
        }}
      >
        Queued
      </Badge>
    )
  }

  // Show regular status badges for other states
  return (
    <Badge
      variant={status === 'ready' || status === 'active' ? 'light' : 'outline'}
      color={statusColors[status] || 'gray'}
      styles={{
        root: {
          fontSize: '13px',
          fontWeight: 500,
          textTransform: 'capitalize',
          paddingLeft: 10,
          paddingRight: 10,
          backgroundColor:
            status === 'ready' || status === 'active'
              ? '#D4F4DD'
              : 'transparent',
          color:
            status === 'ready' || status === 'active'
              ? '#16A34A'
              : status === 'training'
                ? '#3B82F6'
                : status === 'queued'
                  ? '#0891B2'
                  : status === 'failed'
                    ? '#EF4444'
                    : '#F97316',
          borderColor:
            status === 'ready' || status === 'active'
              ? 'transparent'
              : statusColors[status] || '#F97316',
        },
      }}
    >
      {status}
    </Badge>
  )
}
