import type { ClinicalStage } from '../types'

const STYLES: Record<ClinicalStage, string> = {
  screening:  'bg-blue-100   text-blue-800   border-blue-200',
  diagnosis:  'bg-violet-100 text-violet-800 border-violet-200',
  prevention: 'bg-emerald-100 text-emerald-800 border-emerald-200',
  treatment:  'bg-amber-100  text-amber-800  border-amber-200',
  care:       'bg-teal-100   text-teal-800   border-teal-200',
  off_topic:  'bg-slate-100  text-slate-500  border-slate-200',
}

const LABELS: Record<ClinicalStage, string> = {
  screening:  'Screening',
  diagnosis:  'Diagnosis',
  prevention: 'Prevention',
  treatment:  'Treatment',
  care:       'Care',
  off_topic:  'Outside Scope',
}

export default function StageBadge({ stage }: { stage: ClinicalStage }) {
  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${STYLES[stage]}`}
    >
      {LABELS[stage]}
    </span>
  )
}
