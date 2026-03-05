import { BADGE_STYLES, COLORS } from '@/shared/config/colors'
import type { DatasetOption } from '@/shared/types'
import {
  Badge,
  Box,
  Combobox,
  Group,
  InputBase,
  ScrollArea,
  Text,
  useCombobox,
} from '@mantine/core'
import { forwardRef, useMemo } from 'react'

interface DatasetSelectorProps {
  datasets: DatasetOption[]
  value: string
  onChange: (value: string) => void
}

// Custom select item component for rich dropdown
interface SelectItemProps extends React.ComponentPropsWithoutRef<'div'> {
  label: string
  domain: string
  rows: string
  tags: string[]
  storage: string
}

const SelectItem = forwardRef<HTMLDivElement, SelectItemProps>(
  ({ label, domain, rows, tags, storage, ...others }: SelectItemProps, ref) => (
    <div ref={ref} {...others}>
      <Group gap={8} mb={6}>
        <Text size="15px" fw={500}>
          {label}
        </Text>
        <Badge
          size="sm"
          variant="light"
          color="gray"
          styles={BADGE_STYLES.gcsLabel}
        >
          {storage}
        </Badge>
      </Group>
      <Text size="13px" c="dimmed" mb={6}>
        {domain} • {rows} • Ready
      </Text>
      <Group gap={6}>
        {tags.map(tag => (
          <Badge
            key={tag}
            size="sm"
            variant="light"
            color="gray"
            styles={BADGE_STYLES.tag}
          >
            {tag}
          </Badge>
        ))}
      </Group>
    </div>
  )
)

SelectItem.displayName = 'SelectItem'

export function DatasetSelector({
  datasets,
  value,
  onChange,
}: DatasetSelectorProps) {
  const combobox = useCombobox({
    onDropdownClose: () => combobox.resetSelectedOption(),
  })

  const selectedData = useMemo(
    () => datasets.find(d => d.value === value),
    [datasets, value]
  )

  return (
    <Combobox
      store={combobox}
      onOptionSubmit={val => {
        onChange(val)
        combobox.closeDropdown()
      }}
      transitionProps={{ duration: 200, transition: 'pop' }}
    >
      <Combobox.Target>
        <InputBase
          component="button"
          type="button"
          pointer
          rightSection={<Combobox.Chevron />}
          rightSectionPointerEvents="none"
          onClick={() => combobox.toggleDropdown()}
          style={{
            minHeight: 110,
            cursor: 'pointer',
            alignItems: 'flex-start',
          }}
          styles={{
            input: {
              fontSize: '15px',
              fontWeight: 500,
              borderColor: combobox.dropdownOpened
                ? COLORS.PRIMARY
                : COLORS.GRAY_200,
              borderWidth: combobox.dropdownOpened ? 2 : 1,
              textAlign: 'left',
              height: 'auto',
              paddingTop: 16,
              paddingBottom: 16,
              paddingLeft: 16,
              paddingRight: 16,
              transition: 'border-color 0.2s ease, border-width 0.2s ease',
            },
          }}
        >
          {selectedData ? (
            <Box>
              <Group gap={8} mb={4}>
                <Text size="15px" fw={600} c={COLORS.PRIMARY}>
                  {selectedData.label}
                </Text>
                <Badge
                  size="sm"
                  variant="light"
                  color="indigo"
                  styles={BADGE_STYLES.gcsLabel}
                >
                  {selectedData.storage}
                </Badge>
              </Group>
              <Text size="13px" c="dimmed">
                {selectedData.domain} • {selectedData.rows} • Ready
              </Text>
            </Box>
          ) : (
            <Box>
              <Text size="15px" fw={500} c="dark" mb={4}>
                Select a dataset
              </Text>
              <Text size="13px" c="dimmed">
                Choose from your library to begin training
              </Text>
            </Box>
          )}
        </InputBase>
      </Combobox.Target>

      <Combobox.Dropdown p={8}>
        <ScrollArea.Autosize mah={250} type="scroll">
          <Combobox.Options>
            {datasets.map(dataset => (
              <Combobox.Option
                key={dataset.value}
                value={dataset.value}
                p={12}
                style={{
                  borderRadius: 6,
                }}
              >
                <SelectItem
                  label={dataset.label}
                  domain={dataset.domain}
                  rows={dataset.rows}
                  tags={dataset.tags}
                  storage={dataset.storage}
                />
              </Combobox.Option>
            ))}
          </Combobox.Options>
        </ScrollArea.Autosize>
      </Combobox.Dropdown>
    </Combobox>
  )
}
