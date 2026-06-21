import { useState, useCallback } from 'react'
import type { PatientRecord, QueryResponse, ResearchSummary, VisitRecord } from './types'
import { fetchPatient, fetchResearch, streamQuery } from './api/client'
import VisitHistory from './components/VisitHistory'
import QueryPanel from './components/QueryPanel'
import PatientProfile from './components/PatientProfile'

export default function App() {
  const [patientIdInput, setPatientIdInput] = useState('')
  const [activeId, setActiveId] = useState('')
  const [record, setRecord] = useState<PatientRecord | null>(null)
  const [research, setResearch] = useState<ResearchSummary | null>(null)
  const [response, setResponse] = useState<QueryResponse | null>(null)
  const [selectedVisit, setSelectedVisit] = useState<VisitRecord | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [queryIntent, setQueryIntent] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const clearPatient = useCallback(() => {
    setActiveId('')
    setPatientIdInput('')
    setRecord(null)
    setResearch(null)
    setResponse(null)
    setSelectedVisit(null)
    setQueryIntent(null)
    setIsLoading(false)
    setIsStreaming(false)
    setError(null)
  }, [])

  const loadPatient = useCallback(async (id: string) => {
    const trimmed = id.trim()
    if (!trimmed) return
    setError(null)
    setResponse(null)
    setSelectedVisit(null)
    setResearch(null)
    setActiveId(trimmed)
    try {
      const [r, rs] = await Promise.all([
        fetchPatient(trimmed).catch(() => null),
        fetchResearch(trimmed),
      ])
      setRecord(r)
      setResearch(rs)
      if (!r && !rs) setError('Backend unreachable — start uvicorn to enable queries.')
    } catch {
      setRecord(null)
      setResearch(null)
    }
  }, [])

  const handleQuery = async (query: string) => {
    const patientId = activeId || undefined
    setIsLoading(true)
    setIsStreaming(false)
    setQueryIntent(null)
    setError(null)
    setSelectedVisit(null)
    setResponse(null)

    try {
      await streamQuery(
        { patient_id: patientId, query },
        (stage) => {
          setResponse({ patient_id: patientId, stage, response: '', citations: [] })
        },
        (intent) => {
          setQueryIntent(intent)
        },
        (text) => {
          setIsLoading(false)
          setIsStreaming(true)
          setResponse(prev =>
            prev ? { ...prev, response: prev.response + text } : null
          )
        },
        (citations, personalized, renumberedResponse) => {
          setIsStreaming(false)
          setQueryIntent(null)
          setResponse(prev => prev ? {
            ...prev,
            response: renumberedResponse ?? prev.response,
            citations,
            personalized,
          } : null)
          if (patientId) fetchPatient(patientId).then(setRecord).catch(() => {})
        },
      )
    } catch (e) {
      setIsStreaming(false)
      setIsLoading(false)
      setError(e instanceof Error ? e.message : 'Query failed — is the backend running?')
    }
  }

  const handleVisitSelect = (visit: VisitRecord) => {
    setSelectedVisit(visit)
    setResponse(null)
  }

  return (
    <div className="h-screen flex flex-col bg-slate-100">

      {/* ── Header ─────────────────────────────────────────────── */}
      <header className="bg-slate-900 text-white px-6 py-3 flex items-center gap-8 shadow-lg shrink-0">
        <div className="shrink-0">
          <p className="text-base font-semibold tracking-tight leading-none">Dement<span className="text-blue-400">IA</span></p>
          <p className="text-xs text-slate-400 mt-0.5">Clinical Decision Support</p>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-400 shrink-0">Patient ID <span className="text-slate-600">(optional)</span></label>
          <input
            className="bg-slate-800 border border-slate-600 rounded-lg px-3 py-1.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-400 w-48 transition-colors"
            placeholder="Enter ID…"
            value={patientIdInput}
            onChange={(e) => setPatientIdInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && loadPatient(patientIdInput)}
          />
          <button
            className="bg-blue-600 hover:bg-blue-500 text-white text-sm px-4 py-1.5 rounded-lg transition-colors font-medium"
            onClick={() => loadPatient(patientIdInput)}
          >
            Load
          </button>
        </div>

        {activeId && (
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <span className="text-slate-200 font-medium">{activeId}</span>
            {!record && (
              <span className="text-xs bg-slate-700 rounded px-2 py-0.5">new patient</span>
            )}
            <button
              onClick={clearPatient}
              className="text-xs text-slate-400 hover:text-white border border-slate-600 hover:border-slate-400 rounded px-2 py-0.5 transition-colors"
            >
              Clear
            </button>
          </div>
        )}
      </header>

      {/* ── Body ───────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">
        {activeId && (
          <VisitHistory
            visits={record?.visits ?? []}
            onSelect={handleVisitSelect}
          />
        )}
        <main className="flex-1 overflow-hidden">
          <QueryPanel
            onQuery={handleQuery}
            isLoading={isLoading}
            isStreaming={isStreaming}
            queryIntent={queryIntent}
            response={response}
            selectedVisit={selectedVisit}
            error={error}
          />
        </main>
        {activeId && <PatientProfile record={record} research={research} />}
      </div>
    </div>
  )
}
