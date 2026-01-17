import React, { useState } from 'react'
import axios from 'axios'
import './AnswerCheck.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const AnswerCheck = () => {
  const [formData, setFormData] = useState({
    query: '',
    grade: 9,
    subject: 'Українська мова',
    student_id: '',
    student_answers: []
  })
  
  const [loading, setLoading] = useState(false)
  const [response, setResponse] = useState(null)
  const [error, setError] = useState(null)
  const [answerInput, setAnswerInput] = useState('')

  const subjects = ['Українська мова', 'Алгебра', 'Історія України']
  const grades = [8, 9]

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      const payload = {
        query: formData.query,
        grade: formData.grade,
        subject: formData.subject,
        student_id: formData.student_id ? parseInt(formData.student_id) : null,
        student_answers: formData.student_answers
      }

      const res = await axios.post(`${API_BASE}/tutor/check-answers`, payload)
      setResponse(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Помилка при перевірці відповідей')
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const addAnswer = () => {
    if (answerInput.trim()) {
      setFormData(prev => ({
        ...prev,
        student_answers: [...prev.student_answers, answerInput.trim()]
      }))
      setAnswerInput('')
    }
  }

  const removeAnswer = (index) => {
    setFormData(prev => ({
      ...prev,
      student_answers: prev.student_answers.filter((_, i) => i !== index)
    }))
  }

  return (
    <div className="answer-check-container">
      <div className="form-card">
        <h2>✅ Перевірка відповідей учня</h2>
        <p className="form-description">
          Введіть запит та відповіді учня для автоматичної перевірки та отримання рекомендацій
        </p>

        <form onSubmit={handleSubmit} className="check-form">
          <div className="form-group">
            <label htmlFor="query">Запит *</label>
            <textarea
              id="query"
              name="query"
              value={formData.query}
              onChange={handleChange}
              placeholder="Наприклад: Складні речення та їх ознаки"
              required
              rows={4}
            />
          </div>

          <div className="form-row">
            <div className="form-group">
              <label htmlFor="grade">Клас *</label>
              <select
                id="grade"
                name="grade"
                value={formData.grade}
                onChange={handleChange}
                required
              >
                {grades.map(grade => (
                  <option key={grade} value={grade}>{grade}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="subject">Предмет *</label>
              <select
                id="subject"
                name="subject"
                value={formData.subject}
                onChange={handleChange}
                required
              >
                {subjects.map(subject => (
                  <option key={subject} value={subject}>{subject}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="student_id">ID учня (опціонально)</label>
              <input
                type="number"
                id="student_id"
                name="student_id"
                value={formData.student_id}
                onChange={handleChange}
                placeholder="Для персоналізації"
              />
            </div>
          </div>

          <div className="form-group">
            <label>Відповіді учня *</label>
            <div className="answers-input-group">
              <input
                type="text"
                value={answerInput}
                onChange={(e) => setAnswerInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addAnswer())}
                placeholder="Введіть відповідь і натисніть Enter або кнопку 'Додати'"
                className="answer-input"
              />
              <button
                type="button"
                onClick={addAnswer}
                className="add-btn"
              >
                Додати
              </button>
            </div>
            {formData.student_answers.length > 0 && (
              <div className="answers-list">
                {formData.student_answers.map((answer, idx) => (
                  <div key={idx} className="answer-item">
                    <span className="answer-number">{idx + 1}.</span>
                    <span className="answer-text">{answer}</span>
                    <button
                      type="button"
                      onClick={() => removeAnswer(idx)}
                      className="remove-btn"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <button 
            type="submit" 
            className="submit-btn"
            disabled={loading || formData.student_answers.length === 0}
          >
            {loading ? '⏳ Перевірка...' : '🔍 Перевірити відповіді'}
          </button>
        </form>

        {error && (
          <div className="error-message">
            <strong>Помилка:</strong> {error}
          </div>
        )}

        {response && (
          <div className="response-container">
            <h3>📊 Результати перевірки</h3>

            {response.evaluation_results && response.evaluation_results.length > 0 && (
              <section className="response-section">
                <h4>Результати оцінювання</h4>
                <div className="evaluation-results">
                  {response.evaluation_results.map((result, idx) => (
                    <div key={idx} className={`evaluation-item ${result.is_correct ? 'correct' : 'incorrect'}`}>
                      <div className="evaluation-header">
                        <span className="evaluation-number">Питання {idx + 1}</span>
                        <span className={`evaluation-status ${result.is_correct ? 'correct' : 'incorrect'}`}>
                          {result.is_correct ? '✓ Правильно' : '✗ Неправильно'}
                        </span>
                      </div>
                      <div className="evaluation-details">
                        <p><strong>Відповідь учня:</strong> {result.student_answer}</p>
                        <p><strong>Правильна відповідь:</strong> {result.correct_answer}</p>
                        {result.explanation && (
                          <p className="explanation-text"><strong>Пояснення:</strong> {result.explanation}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {response.recommendations && (
              <section className="response-section">
                <h4>💡 Рекомендації</h4>
                <div className="content-box">{response.recommendations}</div>
              </section>
            )}

            {response.next_topics && response.next_topics.length > 0 && (
              <section className="response-section">
                <h4>📚 Наступні теми для вивчення</h4>
                <ul className="topics-list">
                  {response.next_topics.map((topic, idx) => (
                    <li key={idx}>{topic}</li>
                  ))}
                </ul>
              </section>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default AnswerCheck
