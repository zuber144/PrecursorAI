/**
 * services/api.js — Axios-based API client for PrecursorAI backend.
 *
 * Base URL proxied through Vite dev server to http://localhost:8000
 * Frontend uses REST polling (no WebSockets) for dashboard/alert updates.
 */
import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  headers: { 'Content-Type': 'application/json' },
})

// ── Reports ──────────────────────────────────────────────────────────────────

export const submitReport = (payload) => api.post('/reports', payload)
export const listReports = (params = {}) => api.get('/reports', { params })
export const getReport = (id) => api.get(`/reports/${id}`)
export const submitClarification = (id, payload) => api.post(`/reports/${id}/clarify`, payload)

// ── Dashboard ─────────────────────────────────────────────────────────────────

export const getDashboardSummary = () => api.get('/dashboard/summary')

// ── Alerts ────────────────────────────────────────────────────────────────────

export const listAlerts = (params = {}) => api.get('/alerts', { params })
export const markAlertRead = (id) => api.patch(`/alerts/${id}/read`)

// ── Patterns ──────────────────────────────────────────────────────────────────

export const listPatterns = (params = {}) => api.get('/patterns', { params })
export const getPatternDetail = (id) => api.get(`/patterns/${id}`)
export const triggerPatternSweep = () => api.post('/patterns/sweep')

// ── Cognition (legacy stub) ────────────────────────────────────────────────────

export const triggerCognitionSweep = () => api.post('/cognition/sweep')
export const getCognitionStatus = () => api.get('/cognition/status')

export default api

