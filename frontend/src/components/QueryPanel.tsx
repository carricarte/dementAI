import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import type { QueryResponse, VisitRecord } from '../types'
import StageBadge from './StageBadge'
import CitationList from './CitationList'

interface Props {
  onQuery: (query: string) => Promise<void>
  isLoading: boolean
  isStreaming: boolean
  queryIntent: string | null
  response: QueryResponse | null
  selectedVisit: VisitRecord | null
  error: string | null
}

export default function QueryPanel({ onQuery, isLoading, isStreaming, queryIntent, response, selectedVisit, error }: Props) {
  const [query, setQuery] = useState('')

  const handleSubmit = async () => {
    if (!query.trim() || isLoading || isStreaming) return
    await onQuery(query.trim())
    setQuery('')
  }

  const displayed = selectedVisit
    ? { stage: selectedVisit.stage, response: selectedVisit.specialist_response, citations: selectedVisit.citations }
    : response
      ? { stage: response.stage, response: response.response, citations: response.citations }
      : null

  return (
    <div className="h-full flex flex-col">

      {/* ── Scrollable response area ───────────────────────────── */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-6 pt-6 pb-4 flex flex-col gap-5">

          {/* Error */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          )}

          {/* Loading skeleton */}
          {isLoading && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
              <div className="animate-pulse space-y-3">
                <div className="h-5 bg-slate-100 rounded w-24" />
                <div className="h-3 bg-slate-100 rounded w-full mt-4" />
                <div className="h-3 bg-slate-100 rounded w-5/6" />
                <div className="h-3 bg-slate-100 rounded w-4/5" />
                <div className="h-3 bg-slate-100 rounded w-full" />
                <div className="h-3 bg-slate-100 rounded w-3/5" />
              </div>
              {queryIntent === 'patient_specific' && (
                <p className="mt-4 text-xs text-blue-500 flex items-center gap-1.5">
                  <span className="inline-block w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
                  Analyzing patient record…
                </p>
              )}
            </div>
          )}

          {/* Response / selected visit */}
          {(isStreaming || (!isLoading && displayed)) && displayed && (
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <StageBadge stage={displayed.stage} />
                  {!selectedVisit && response?.personalized && (
                    <span className="text-xs font-medium text-blue-600 bg-blue-50 border border-blue-200 rounded px-2 py-0.5">
                      Personalized
                    </span>
                  )}
                </div>
                {selectedVisit && (
                  <span className="text-xs text-slate-400 italic">
                    Visit · {new Date(selectedVisit.timestamp).toLocaleDateString()}
                  </span>
                )}
              </div>
              <ReactMarkdown
                components={{
                  h1: ({ children }) => <h1 className="text-lg font-bold text-slate-900 mt-5 mb-2 first:mt-0">{children}</h1>,
                  h2: ({ children }) => <h2 className="text-base font-semibold text-slate-800 mt-4 mb-1.5 first:mt-0">{children}</h2>,
                  h3: ({ children }) => <h3 className="text-sm font-semibold text-slate-700 mt-3 mb-1 first:mt-0">{children}</h3>,
                  p: ({ children }) => <p className="text-sm text-slate-700 leading-relaxed mb-2">{children}</p>,
                  strong: ({ children }) => <strong className="font-semibold text-slate-900">{children}</strong>,
                  em: ({ children }) => <em className="italic">{children}</em>,
                  ul: ({ children }) => <ul className="text-sm text-slate-700 list-disc list-outside ml-4 mb-2 space-y-0.5">{children}</ul>,
                  ol: ({ children }) => <ol className="text-sm text-slate-700 list-decimal list-outside ml-4 mb-2 space-y-0.5">{children}</ol>,
                  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
                  hr: () => <hr className="my-4 border-slate-200" />,
                  code: ({ children }) => {
                    const text = typeof children === 'string' ? children : ''
                    if (/^\[\d+\]$/.test(text)) {
                      const n = text.slice(1, -1)
                      return (
                        <a
                          href={`#ref-${n}`}
                          className="inline-block text-xs font-semibold text-blue-600 bg-blue-50 border border-blue-200 rounded px-1 py-0.5 hover:bg-blue-100 transition-colors mx-0.5 no-underline"
                        >
                          {text}
                        </a>
                      )
                    }
                    return <code className="bg-slate-100 text-slate-800 text-xs font-mono px-1 py-0.5 rounded">{children}</code>
                  },
                  blockquote: ({ children }) => <blockquote className="border-l-2 border-slate-300 pl-3 text-slate-500 italic my-2">{children}</blockquote>,
                  a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">{children}</a>,
                }}
              >
                {injectCitationMarkers(displayed.response)}
              </ReactMarkdown>
              {isStreaming && !selectedVisit && (
                <span className="inline-block w-0.5 h-4 bg-slate-500 animate-pulse align-middle ml-0.5" />
              )}
              <CitationList citations={displayed.citations} />
            </div>
          )}
        </div>
      </div>

      {/* ── Query input — pinned to bottom center ──────────────── */}
      <div className="shrink-0 border-t border-slate-200 bg-slate-100 px-6 py-4">
        <div className="max-w-3xl mx-auto bg-white rounded-xl border border-slate-200 shadow-sm">
          <div className="px-5 pt-4 pb-3">
            <textarea
              className="w-full resize-none text-sm text-slate-800 placeholder-slate-400 focus:outline-none leading-relaxed"
              rows={3}
              placeholder="Type your clinical question…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSubmit()
              }}
            />
          </div>
          <div className="px-5 py-3 border-t border-slate-100 flex items-center justify-between">
            <span className="text-xs text-slate-400">⌘ Enter to submit</span>
            <button
              disabled={isLoading || isStreaming || !query.trim()}
              onClick={handleSubmit}
              className="bg-slate-900 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm px-5 py-2 rounded-lg transition-colors font-medium"
            >
              {isLoading ? (
                <span className="flex items-center gap-2">
                  <Spinner />
                  Thinking…
                </span>
              ) : isStreaming ? (
                <span className="flex items-center gap-2">
                  <Spinner />
                  Writing…
                </span>
              ) : (
                'Ask'
              )}
            </button>
          </div>
        </div>
      </div>

    </div>
  )
}

// Wrap [N] citation markers in backtick code spans so the `code` component
// can intercept them as anchor badges. Backtick spans are used instead of
// **bold** because [N] inside **[N]** is parsed as a markdown link reference,
// producing complex React children that fail the single-citation string check.
function injectCitationMarkers(text: string): string {
  return text
    .replace(/\*{3,}/g, '')             // strip PDF extraction artifacts
    .replace(/\](?=\[\d+\])/g, '] ')   // space before each adjacent [N] — lookahead avoids consuming the ] needed by the next match
    .replace(/\[(\d+)\]/g, '`[$1]`')   // wrap each [N] in a code span
}

function Spinner() {
  return (
    <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
    </svg>
  )
}
