import { useCreateDataset } from '@/shared/hooks/useDatasets'
import {
  Alert,
  Button,
  FileInput,
  Group,
  Modal,
  Progress,
  Select,
  Stack,
  Text,
  Textarea,
  TextInput,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { IconAlertCircle, IconUpload } from '@tabler/icons-react'
import axios from 'axios'
import { useEffect, useState } from 'react'

interface ImportDatasetModalProps {
  opened: boolean
  onClose: () => void
}

export function ImportDatasetModal({
  opened,
  onClose,
}: ImportDatasetModalProps) {
  const [importName, setImportName] = useState('')
  const [importDomain, setImportDomain] = useState<string | null>(null)
  const [importFile, setImportFile] = useState<File | null>(null)
  const [importTags, setImportTags] = useState('')
  const [importDescription, setImportDescription] = useState('')
  const [uploadProgress, setUploadProgress] = useState(0)
  const [errorMessage, setErrorMessage] = useState('')
  const [targetColumn, setTargetColumn] = useState<string>('')
  const [columns, setColumns] = useState<string[]>([])
  const [loadingColumns, setLoadingColumns] = useState(false)

  const handleUploadProgress = (progressEvent: any) => {
    if (progressEvent.total) {
      const progress = Math.round(
        (progressEvent.loaded * 100) / progressEvent.total
      )
      setUploadProgress(progress)
    }
  }

  const { mutate: createDataset, isPending: isCreating } =
    useCreateDataset(handleUploadProgress)

  // Extract columns from CSV file when tabular domain and CSV file are selected
  // Note: Parquet files are binary and can't be parsed in browser, so column extraction is skipped
  useEffect(() => {
    const fileName = importFile?.name.toLowerCase() || ''
    const isCSV = fileName.endsWith('.csv')
    const isParquet = fileName.endsWith('.parquet') || fileName.endsWith('.pq')

    if (importDomain === 'tabular' && importFile && isCSV) {
      setLoadingColumns(true)
      setColumns([])
      setTargetColumn('')

      const reader = new FileReader()
      reader.onload = e => {
        try {
          const text = e.target?.result as string
          const firstLine = text.split('\n')[0]
          const cols = firstLine
            .split(',')
            .map(col => col.trim().replace(/^"|"$/g, ''))
          setColumns(cols)
          // Auto-select last column as default target
          if (cols.length > 0) {
            setTargetColumn(cols[cols.length - 1])
          }
        } catch (error) {
          console.error('Failed to parse CSV columns:', error)
        } finally {
          setLoadingColumns(false)
        }
      }
      reader.onerror = () => {
        setLoadingColumns(false)
      }
      // Read only first 1KB to extract headers
      reader.readAsText(importFile.slice(0, 1024))
    } else if (importDomain === 'tabular' && importFile && isParquet) {
      // Parquet files are binary, cannot extract columns in browser
      // User will need to specify target column manually after upload
      setColumns([])
      setTargetColumn('')
      setLoadingColumns(false)
    } else {
      setColumns([])
      setTargetColumn('')
    }
  }, [importDomain, importFile])

  const handleImport = () => {
    // Clear previous errors
    setErrorMessage('')
    setUploadProgress(0)

    if (!importName || !importDomain || !importFile) {
      setErrorMessage('Please fill in all required fields')
      return
    }

    // Validate target column for tabular datasets (CSV files only - Parquet can be set later)
    const isCSV = importFile?.name.toLowerCase().endsWith('.csv') || false
    if (importDomain === 'tabular' && isCSV && !targetColumn) {
      setErrorMessage('Please select a target column for CSV datasets')
      return
    }

    const formData = new FormData()
    formData.append('name', importName)
    formData.append('domain', importDomain)
    formData.append('file', importFile)
    if (importTags) {
      formData.append('tags', importTags)
    }
    if (importDescription) {
      formData.append('description', importDescription)
    }
    if (importDomain === 'tabular' && targetColumn) {
      formData.append('target_column', targetColumn)
    }

    createDataset(formData, {
      onSuccess: () => {
        notifications.show({
          title: 'Dataset imported',
          message: `"${importName}" has been imported successfully`,
          color: 'green',
        })
        // Reset form
        setImportName('')
        setImportDomain(null)
        setImportFile(null)
        setImportTags('')
        setImportDescription('')
        setTargetColumn('')
        setColumns([])
        setUploadProgress(0)
        setErrorMessage('')
        onClose()
      },
      onError: error => {
        // Extract error message from axios error
        let errorMsg = 'Failed to import dataset'
        if (axios.isAxiosError(error) && error.response?.data?.detail) {
          errorMsg = error.response.data.detail
        } else if (error instanceof Error) {
          errorMsg = error.message
        }
        setErrorMessage(errorMsg)
        setUploadProgress(0)
      },
    })
  }

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title="Import Dataset"
      size="lg"
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
          Upload your dataset files (CSV, Parquet, ZIP). Large files (up to
          100GB) are supported.
        </Text>

        {errorMessage && (
          <Alert
            icon={<IconAlertCircle size={16} />}
            title="Import Failed"
            color="red"
            variant="light"
            withCloseButton
            onClose={() => setErrorMessage('')}
          >
            {errorMessage}
          </Alert>
        )}

        {isCreating && uploadProgress > 0 && (
          <Stack gap="xs">
            <Group justify="space-between">
              <Text size="sm" fw={500}>
                Uploading dataset...
              </Text>
              <Text size="sm" c="dimmed">
                {uploadProgress}%
              </Text>
            </Group>
            <Progress
              value={uploadProgress}
              size="sm"
              color="indigo"
              animated
            />
          </Stack>
        )}

        <TextInput
          label="Dataset Name"
          placeholder="Enter dataset name"
          required
          value={importName}
          onChange={e => setImportName(e.currentTarget.value)}
          styles={{
            label: { fontSize: '14px', fontWeight: 500, marginBottom: 8 },
            input: { fontSize: '15px' },
          }}
        />

        <Select
          label="Domain"
          placeholder="Select domain"
          required
          value={importDomain}
          onChange={setImportDomain}
          data={[
            { value: 'vision', label: 'Vision' },
            { value: 'tabular', label: 'Tabular' },
          ]}
          description="Only Vision and Tabular datasets are currently supported"
          styles={{
            label: { fontSize: '14px', fontWeight: 500, marginBottom: 8 },
            input: { fontSize: '15px' },
          }}
        />

        <FileInput
          label="Dataset File"
          placeholder="Upload file or ZIP archive"
          required
          value={importFile}
          onChange={setImportFile}
          leftSection={<IconUpload size={16} />}
          accept=".zip,.csv,.parquet,.pq,.json,.txt,image/*"
          description="Supports CSV, Parquet (.parquet, .pq), ZIP archives, and images"
          styles={{
            label: { fontSize: '14px', fontWeight: 500, marginBottom: 8 },
            input: { fontSize: '15px' },
          }}
        />

        {/* Target Column for Tabular Datasets */}
        {importDomain === 'tabular' && columns.length > 0 && (
          <Select
            label="Target Column"
            placeholder="Select target column"
            required
            value={targetColumn}
            onChange={value => setTargetColumn(value || '')}
            data={columns.map(col => ({ value: col, label: col }))}
            searchable
            description={
              loadingColumns
                ? 'Loading columns...'
                : `${columns.length} columns available`
            }
            disabled={loadingColumns}
            styles={{
              label: { fontSize: '14px', fontWeight: 500, marginBottom: 8 },
              input: { fontSize: '15px' },
            }}
          />
        )}

        {/* Info for Parquet files - target column will be set after upload */}
        {importDomain === 'tabular' &&
          importFile &&
          (importFile.name.toLowerCase().endsWith('.parquet') ||
            importFile.name.toLowerCase().endsWith('.pq')) && (
            <Alert color="blue" variant="light">
              <Text size="sm">
                For Parquet files, the target column must be configured after
                upload before training. You can set it via the dataset details
                page.
              </Text>
            </Alert>
          )}

        <TextInput
          label="Tags"
          placeholder="Enter comma-separated tags (optional)"
          value={importTags}
          onChange={e => setImportTags(e.currentTarget.value)}
          styles={{
            label: { fontSize: '14px', fontWeight: 500, marginBottom: 8 },
            input: { fontSize: '15px' },
          }}
        />

        <Textarea
          label="Description"
          placeholder="Enter dataset description (optional)"
          value={importDescription}
          onChange={e => setImportDescription(e.currentTarget.value)}
          minRows={3}
          styles={{
            label: { fontSize: '14px', fontWeight: 500, marginBottom: 8 },
            input: { fontSize: '15px' },
          }}
        />

        <Group justify="flex-end" gap="sm" mt="md">
          <Button
            variant="light"
            color="gray"
            onClick={onClose}
            disabled={isCreating}
            styles={{
              root: {
                fontSize: '15px',
              },
            }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleImport}
            loading={isCreating}
            leftSection={!isCreating && <IconUpload size={16} />}
            color="indigo"
            styles={{
              root: {
                backgroundColor: '#6366F1',
                fontSize: '15px',
              },
            }}
          >
            Import Dataset
          </Button>
        </Group>
      </Stack>
    </Modal>
  )
}
