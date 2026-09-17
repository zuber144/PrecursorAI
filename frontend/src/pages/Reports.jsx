import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { listReports, submitReport, submitClarification } from '../services/api'
import { Activity, Plus, ChevronRight, AlertTriangle, Info } from 'lucide-react'
import heroImage from '../assets/image1.png'
import '../ops/styles/reports.css'

const REPORT_TYPES = ['UNSAFE_ACT', 'UNSAFE_CONDITION', 'NEAR_MISS']

const STATUS_CLASS = {
  RESOLVED: 'badge-resolved',
  ANALYZED: 'badge-analyzed',
  REVIEW:   'badge-review',
  PENDING:  'badge-error',
  ERROR:    'badge-error',
}

const TYPE_ICON = {
  UNSAFE_ACT:       <Activity size={14} />,
  UNSAFE_CONDITION: <Info size={14} />,
  NEAR_MISS:        <AlertTriangle size={14} className="type-icon" />,
}

export default function Reports({ role }) {
  const [reports, setReports]       = useState([])
  const [loading, setLoading]       = useState(role === 'ADMIN')
  const [submitting, setSubmitting] = useState(false)
  const [form, setForm]             = useState({
    report_type: 'NEAR_MISS',
    report_text: '',
    location:    '',
    asset_id:    '',
  })
  const [clarification, setClarification] = useState({
    reportId: null,
    question: null,
    answer: ''
  })

  useEffect(() => {
    if (role === 'ADMIN') {
      listReports().then((r) => setReports(r.data)).finally(() => setLoading(false))
    }
  }, [role])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setSubmitting(true)
    try {
      const res = await submitReport(form)
      if (res.data.status === 'NEEDS_CLARIFICATION') {
        setClarification({ 
          reportId: res.data.report_id, 
          question: res.data.followup_question, 
          answer: '' 
        })
      } else {
        setForm({ report_type: 'NEAR_MISS', report_text: '', location: '', asset_id: '' })
        alert("Report successfully submitted to the AI analysis queue.")
      }
    } catch (err) {
      alert("Failed to submit report. Backend may be offline.")
    } finally {
      setSubmitting(false)
    }
  }

  const handleClarificationSubmit = async (e) => {
    e.preventDefault()
    setSubmitting(true)
    try {
      await submitClarification(clarification.reportId, { answer: clarification.answer })
      setClarification({ reportId: null, question: null, answer: '' })
      setForm({ report_type: 'NEAR_MISS', report_text: '', location: '', asset_id: '' })
      alert("Clarification submitted. AI analysis complete.")
    } catch (err) {
      alert("Failed to submit clarification.")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="reports-page-wrap">
      {/* Hero banner */}
      <div className="page-hero">
        <img src={heroImage} alt="" className="page-hero-img" />
        <div className="page-hero-overlay" />
        <div className="page-hero-content">
          <div className="page-hero-label">Operations Intelligence</div>
          <div className="page-hero-title">{role === 'USER' ? 'File a Safety Report' : 'Reports Log'}</div>
          <div className="page-hero-subtitle">
            {role === 'USER' ? 'Submit an HSE safety report to the AI processing queue.' : 'Review AI-analyzed HSE safety reports from all assets.'}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="reports-content">
        {role === 'USER' && !clarification.reportId && (
          <div className="report-form-panel">
            <h2>Submit Safety Report</h2>
            <form onSubmit={handleSubmit}>
              <div className="form-grid">
                <div className="form-field">
                  <label>Report Type</label>
                  <select
                    className="form-input"
                    value={form.report_type}
                    onChange={(e) => setForm({ ...form, report_type: e.target.value })}
                  >
                    {REPORT_TYPES.map((t) => (
                      <option key={t} value={t}>{t.replace(/_/g, ' ')}</option>
                    ))}
                  </select>
                </div>
                <div className="form-field">
                  <label>Location</label>
                  <input
                    className="form-input"
                    value={form.location}
                    onChange={(e) => setForm({ ...form, location: e.target.value })}
                    placeholder="e.g. Unit 4"
                  />
                </div>
                <div className="form-field">
                  <label>Asset ID (Optional)</label>
                  <input
                    className="form-input"
                    value={form.asset_id}
                    onChange={(e) => setForm({ ...form, asset_id: e.target.value })}
                    placeholder="e.g. TANK-17"
                  />
                </div>
              </div>
              <div className="form-field">
                <label>Description</label>
                <textarea
                  className="form-textarea"
                  rows={4}
                  value={form.report_text}
                  onChange={(e) => setForm({ ...form, report_text: e.target.value })}
                  placeholder="Describe the sequence of events, hazards observed, and actions taken..."
                  required
                />
              </div>
              <div className="form-actions">
                <button type="submit" className="btn-primary" disabled={submitting}>
                  {submitting ? <><Activity size={14} className="spin" /> Analyzing...</> : 'Submit to AI'}
                </button>
              </div>
            </form>
          </div>
        )}

        {role === 'USER' && clarification.reportId && (
          <div className="report-form-panel">
            <h2>Additional Information Required</h2>
            <div style={{ backgroundColor: '#FFFBEB', borderLeft: '4px solid #F59E0B', padding: '1rem', marginBottom: '1.5rem', borderRadius: '4px' }}>
              <p style={{ margin: 0, color: '#92400E', fontWeight: 500, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <AlertTriangle size={16} />
                AI Clarification Needed
              </p>
              <p style={{ margin: '0.5rem 0 0 0', color: '#92400E' }}>
                {clarification.question}
              </p>
            </div>
            <form onSubmit={handleClarificationSubmit}>
              <div className="form-field">
                <label>Your Answer</label>
                <textarea
                  className="form-textarea"
                  rows={3}
                  value={clarification.answer}
                  onChange={(e) => setClarification({ ...clarification, answer: e.target.value })}
                  placeholder="Type your answer here..."
                  required
                  autoFocus
                />
              </div>
              <div className="form-actions">
                <button type="submit" className="btn-primary" style={{ backgroundColor: '#F59E0B', color: 'white' }} disabled={submitting}>
                  {submitting ? <><Activity size={14} className="spin" /> Analyzing...</> : 'Submit Answer'}
                </button>
              </div>
            </form>
          </div>
        )}

        {role === 'ADMIN' && (
          loading ? (
            <div className="loading-screen" style={{ minHeight: '40vh' }}>
              <Activity size={24} className="spin" style={{ color: '#F59E0B' }} />
              <span>LOADING REPORTS...</span>
            </div>
          ) : (
          <div className="reports-table-container">
            <table className="reports-table">
              <thead>
                <tr>
                  <th>Type</th>
                  <th>Asset / Location</th>
                  <th>Date Submitted</th>
                  <th>AI Status</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {reports.map((r) => (
                  <tr key={r.id || r.report_id}>
                    <td>
                      <div className="type-cell">
                        {TYPE_ICON[r.report_type] || <Info size={14} />}
                        {r.report_type.replace(/_/g, ' ')}
                      </div>
                    </td>
                    <td>
                      <span className="asset-main">{r.asset_id || 'Unknown Asset'}</span>
                      <span className="asset-sub">{r.location || 'Unknown Location'}</span>
                    </td>
                    <td style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.78rem', color: '#8892A4' }}>
                      {new Date(r.created_at).toLocaleDateString(undefined, {
                        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
                      })}
                    </td>
                    <td>
                      <span className={`badge ${STATUS_CLASS[r.status] || 'badge-error'}`}>
                        {r.status}
                      </span>
                    </td>
                    <td>
                      <Link to={`/reports/${r.id || r.report_id}`}>
                        <ChevronRight size={17} className="chevron-icon" />
                      </Link>
                    </td>
                  </tr>
                ))}
                {reports.length === 0 && (
                  <tr>
                    <td colSpan="5" style={{ textAlign: 'center', padding: '3rem', color: '#5C6478', fontFamily: 'JetBrains Mono, monospace', fontSize: '0.75rem', letterSpacing: '0.08em' }}>
                      NO REPORTS FOUND
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </div>
  )
}
