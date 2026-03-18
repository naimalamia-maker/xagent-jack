"use client"

import { useEffect } from "react"
import { useApp } from "@/contexts/app-context-chat"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useI18n } from "@/contexts/i18n-context"
import { Loader2, XIcon } from "lucide-react"
import { DocxPreviewRenderer } from "@/components/docx-preview-renderer"

interface FilePreviewContentProps {
  open: boolean
}

export function FilePreviewContent({ open }: FilePreviewContentProps) {
  const { state, dispatch } = useApp()
  const { filePreview } = state
  const { t } = useI18n()

  useEffect(() => {
    if (open && filePreview.fileId && !filePreview.content && !filePreview.error) {
      const loadFileContent = async () => {
        try {
          const apiUrl = getApiUrl()
          const isPptxFile = !!filePreview.fileName.match(/\.pptx?$/i)

          const url = isPptxFile
            ? `${apiUrl}/api/files/preview/${encodeURIComponent(filePreview.fileId)}`
            : `${apiUrl}/api/files/download/${encodeURIComponent(filePreview.fileId)}`

          const response = await apiRequest(url, {
            cache: 'no-cache',
            headers: { 'Cache-Control': 'no-cache', 'Pragma': 'no-cache' }
          })

          if (!response.ok) {
            dispatch({ type: "SET_FILE_PREVIEW_CONTENT", payload: { content: "", error: t('files.previewDialog.errors.loadFailed') } })
            return
          }

          const contentType = (response.headers.get('content-type') || '').split(';')[0].trim()
          let fileContent: string
          let mimeType = contentType

          if (isPptxFile) {
            if (contentType === 'application/pdf') {
              // PDF from LibreOffice — binary to base64
              const buf = await response.arrayBuffer()
              fileContent = btoa(Array.from(new Uint8Array(buf), b => String.fromCharCode(b)).join(''))
              mimeType = 'application/pdf'
            } else {
              // HTML fallback
              fileContent = await response.text()
              mimeType = 'text/html'
            }
          } else if (filePreview.fileName.match(/\.(docx|pdf|jpg|jpeg|png|gif|webp|svg)$/i)) {
            const buf = await response.arrayBuffer()
            fileContent = btoa(Array.from(new Uint8Array(buf), b => String.fromCharCode(b)).join(''))
          } else {
            fileContent = await response.text()
          }

          dispatch({ type: "SET_FILE_PREVIEW_CONTENT", payload: { content: fileContent, mimeType, error: null } })
        } catch (error) {
          const msg = (error as any)?.message || t('common.errors.unknown')
          dispatch({ type: "SET_FILE_PREVIEW_CONTENT", payload: { content: "", error: t('files.previewDialog.errors.networkErrorWithMsg', { msg }) } })
        }
      }
      loadFileContent()
    }
  }, [open, filePreview.fileId, filePreview.content, filePreview.error, dispatch, t, filePreview.fileName])

  return (
    <div className="w-full h-full">
      <div className="flex-1 overflow-hidden flex flex-col min-h-0 h-full">
        {filePreview.isLoading ? (
          <div className="flex items-center justify-center h-full">
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
              <span className="text-sm text-muted-foreground">{t('files.previewDialog.loading')}</span>
            </div>
          </div>
        ) : filePreview.error ? (
          <div className="flex items-center justify-center h-full">
            <div className="flex flex-col items-center gap-2 text-center">
              <XIcon className="h-8 w-8 text-destructive" />
              <span className="text-sm text-muted-foreground">{filePreview.error}</span>
            </div>
          </div>
        ) : (
          <div className="flex-1 overflow-auto bg-muted/30 rounded border">
            {/* PPTX as PDF (best quality) */}
            {(filePreview.fileName.match(/\.pptx?$/i)) && filePreview.mimeType === 'application/pdf' && filePreview.content ? (
              <iframe
                src={`data:application/pdf;base64,${filePreview.content}`}
                className="w-full h-full border-0"
                title={filePreview.fileName}
              />
            /* PPTX as HTML fallback */
            ) : (filePreview.fileName.match(/\.pptx?$/i)) && filePreview.mimeType === 'text/html' ? (
              <iframe
                srcDoc={filePreview.content || ''}
                className="w-full h-full border-0"
                sandbox="allow-same-origin allow-scripts"
                title={filePreview.fileName}
              />
            ) : filePreview.fileName.match(/\.(jpg|jpeg|png|gif|webp|svg)$/i) && filePreview.content ? (
              <div className="flex items-center justify-center h-full p-4">
                <img
                  src={`data:image/${filePreview.fileName.split('.').pop()};base64,${filePreview.content}`}
                  alt={filePreview.fileName}
                  className="max-w-full max-h-full object-contain"
                />
              </div>
            ) : filePreview.fileName.toLowerCase().endsWith('.pdf') && filePreview.content ? (
              <div className="flex items-center justify-center h-full p-4">
                <iframe
                  src={`data:application/pdf;base64,${filePreview.content}`}
                  className="w-full h-full border-0"
                  title={filePreview.fileName}
                />
              </div>
            ) : filePreview.fileName.toLowerCase().endsWith('.docx') ? (
              <DocxPreviewRenderer base64Content={filePreview.content || ''} />
            ) : filePreview.fileName.match(/\.html?$/i) ? (
              <iframe
                srcDoc={filePreview.content || ''}
                className="w-full h-full border-0"
                sandbox="allow-same-origin allow-scripts"
                title={filePreview.fileName}
              />
            ) : (
              <pre className="p-4 text-sm font-mono whitespace-pre-wrap break-words">
                {filePreview.content || t('files.previewDialog.emptyContent')}
              </pre>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
