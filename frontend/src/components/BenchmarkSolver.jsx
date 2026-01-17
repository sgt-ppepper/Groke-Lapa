import React, { useState } from 'react'
import axios from 'axios'
import './BenchmarkSolver.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const BenchmarkSolver = () => {
  const [questions, setQuestions] = useState([])
  const [loading, setLoading] = useState(false)
  const [response, setResponse] = useState(null)
  const [error, setError] = useState(null)
  
  const [newQuestion, setNewQuestion] = useState({
    question_id: '',
    question_text: '',
    answers: ['', '', '', ''],
    subject: 'Українська мова',
    grade: 9
  })

  const subjects = ['Українська мова', 'Алгебра', 'Історія України']
  const grades = [8, 9]

  const addQuestion = () => {
    if (newQuestion.question_id && newQuestion.question_text && 
        newQuestion.answers.every(a => a.trim())) {
      setQuestions(prev => [...prev, { ...newQuestion }])
      setNewQuestion({
        question_id: '',
        question_text: '',
        answers: ['', '', '', ''],
        subject: 'Українська мова',
        grade: 9
      })
    }
  }

  const removeQuestion = (index) => {
    setQuestions(prev => prev.filter((_, i) => i !== index))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (questions.length === 0) {
      setError('Додайте хоча б одне питання')
      return
    }

    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      const payload = { questions }
      const res = await axios.post(`${API_BASE}/benchmark/solve`, payload)
      setResponse(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Помилка при розв\'язанні питань')
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleNewQuestionChange = (field, value) => {
    if (field.startsWith('answer_')) {
      const idx = parseInt(field.split('_')[1])
      setNewQuestion(prev => ({
        ...prev,
        answers: prev.answers.map((a, i) => i === idx ? value : a)
      }))
    } else {
      setNewQuestion(prev => ({ ...prev, [field]: value }))
    }
  }

  return (
    <div className="benchmark-solver-container">
      <div className="form-card">
        <h2>🎯 Benchmark Solver</h2>
        <p className="form-description">
          Додайте питання для автоматичного розв'язання через AI
        </p>

        <div className="question-builder">
          <h3>Створити нове питання</h3>
          <div className="new-question-form">
            <div className="form-group">
              <label>ID питання *</label>
              <input
                type="text"
                value={newQuestion.question_id}
                onChange={(e) => handleNewQuestionChange('question_id', e.target.value)}
                placeholder="question_1"
              />
            </div>

            <div className="form-group">
              <label>Текст питання *</label>
              <textarea
                value={newQuestion.question_text}
                onChange={(e) => handleNewQuestionChange('question_text', e.target.value)}
                rows={3}
                placeholder="Текст питання..."
              />
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Предмет</label>
                <select
                  value={newQuestion.subject}
                  onChange={(e) => handleNewQuestionChange('subject', e.target.value)}
                >
                  {subjects.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>

              <div className="form-group">
                <label>Клас</label>
                <select
                  value={newQuestion.grade}
                  onChange={(e) => handleNewQuestionChange('grade', parseInt(e.target.value))}
                >
                  {grades.map(g => <option key={g} value={g}>{g}</option>)}
                </select>
              </div>
            </div>

            <div className="form-group">
              <label>Варіанти відповідей *</label>
              {newQuestion.answers.map((ans, idx) => (
                <input
                  key={idx}
                  type="text"
                  value={ans}
                  onChange={(e) => handleNewQuestionChange(`answer_${idx}`, e.target.value)}
                  placeholder={`Варіант ${String.fromCharCode(65 + idx)}`}
                  className="answer-option-input"
                />
              ))}
            </div>

            <button
              type="button"
              onClick={addQuestion}
              className="add-question-btn"
            >
              + Додати питання
            </button>
          </div>
        </div>

        {questions.length > 0 && (
          <div className="questions-preview">
            <h3>Додані питання ({questions.length})</h3>
            {questions.map((q, idx) => (
              <div key={idx} className="question-preview-item">
                <div className="preview-header">
                  <span className="preview-id">{q.question_id}</span>
                  <button
                    type="button"
                    onClick={() => removeQuestion(idx)}
                    className="remove-btn"
                  >
                    ×
                  </button>
                </div>
                <p className="preview-text">{q.question_text}</p>
                <div className="preview-options">
                  {q.answers.map((ans, aidx) => (
                    <span key={aidx} className="preview-option">
                      {String.fromCharCode(65 + aidx)}. {ans}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <button
            type="submit"
            className="submit-btn"
            disabled={loading || questions.length === 0}
          >
            {loading ? '⏳ Розв\'язання...' : '🚀 Розв\'язати всі питання'}
          </button>
        </form>

        {error && (
          <div className="error-message">
            <strong>Помилка:</strong> {error}
          </div>
        )}

        {response && (
          <div className="response-container">
            <h3>✅ Результати розв'язання</h3>
            <div className="solver-results">
              {response.answers?.map((answer, idx) => {
                const question = questions[idx]
                const answerLetter = question?.answers?.[answer.answer_index] 
                  ? String.fromCharCode(65 + answer.answer_index) 
                  : 'N/A'
                
                return (
                  <div key={idx} className="solver-result-item">
                    <div className="result-header">
                      <span className="result-id">ID: {answer.question_id}</span>
                      <span className="result-answer">
                        Відповідь: <strong>{answerLetter}</strong> (індекс: {answer.answer_index})
                      </span>
                    </div>
                    {answer.answer_text && (
                      <p className="result-text">{answer.answer_text}</p>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default BenchmarkSolver
