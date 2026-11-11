import { useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { FileText } from 'lucide-react'
import { useGraphStore } from '@/stores/graph'
import { useSettingsStore } from '@/stores/settings'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/Select'

// Radix Select forbids empty string values, so "whole knowledge base" uses a sentinel.
const ALL_FILES = '__all_files__'

/**
 * File filter for the knowledge graph view, backed by /graphs `file_path` +
 * the response's `file_paths` list (all processed documents of the KB).
 */
const GraphFileFilter = () => {
  const { t } = useTranslation()
  const filePaths = useGraphStore.use.graphFilePaths()
  const filter = useSettingsStore.use.graphFileFilter()

  const handleSelect = useCallback((value: string) => {
    useSettingsStore.getState().setGraphFileFilter(value === ALL_FILES ? '' : value)
    // Re-trigger the graph fetch, following the GraphLabels refresh pattern
    const graphState = useGraphStore.getState()
    graphState.setGraphDataFetchAttempted(false)
    graphState.setLastSuccessfulQueryLabel('')
    graphState.incrementGraphDataVersion()
  }, [])

  // A single file offers nothing to filter on
  if (filePaths.length < 2) {
    return null
  }

  return (
    <Select value={filter || ALL_FILES} onValueChange={handleSelect}>
      <SelectTrigger
        className="max-h-8 w-44 bg-background shadow-md sm:w-56"
        title={t('graphPanel.graphFileFilter.placeholder')}
      >
        <FileText className="mr-1 size-4 shrink-0 opacity-70" />
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={ALL_FILES}>{t('graphPanel.graphFileFilter.allFiles')}</SelectItem>
        {filePaths.map((path) => (
          <SelectItem key={path} value={path} title={path}>
            <span className="max-w-52 truncate">{path}</span>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

export default GraphFileFilter
