import React, { useState, useEffect } from 'react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import './QueryForm.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Component for individual practice question with answer reveal
const PracticeQuestion = ({ question, questionIndex }) => {
  const [showAnswer, setShowAnswer] = useState(false)
  
  return (
    <div className="practice-question">
      <div className="question-number">Питання {questionIndex + 1}</div>
      <div className="question-text">
        <ReactMarkdown
          remarkPlugins={[remarkMath]}
          rehypePlugins={[rehypeKatex]}
        >
          {question.question}
        </ReactMarkdown>
      </div>
      <div className="options">
        {question.options?.map((option, optIdx) => {
          const letter = String.fromCharCode(65 + optIdx)
          const isCorrect = letter === question.correct_answer
          const showCorrectness = showAnswer && isCorrect
          return (
            <div 
              key={optIdx} 
              className={`option ${showCorrectness ? 'correct' : ''}`}
            >
              <strong>{letter}.</strong> 
              <ReactMarkdown
                remarkPlugins={[remarkMath]}
                rehypePlugins={[rehypeKatex]}
              >
                {option}
              </ReactMarkdown>
              {showCorrectness && <span className="correct-badge">✓ Правильно</span>}
            </div>
          )
        })}
      </div>
      <button
        type="button"
        onClick={() => setShowAnswer(!showAnswer)}
        className="reveal-answer-btn"
      >
        {showAnswer ? '🙈 Сховати відповідь' : '👁️ Показати правильну відповідь'}
      </button>
      {showAnswer && question.explanation && (
        <div className="explanation">
          <strong>Пояснення:</strong>
          <ReactMarkdown
            remarkPlugins={[remarkMath]}
            rehypePlugins={[rehypeKatex]}
          >
            {question.explanation}
          </ReactMarkdown>
        </div>
      )}
    </div>
  )
}

const QueryForm = () => {
  const [formData, setFormData] = useState({
    query: '',
    grade: 9,
    subject: 'Українська мова',
    student_id: ''
  })
  
  const [loading, setLoading] = useState(false)
  const [response, setResponse] = useState(null)
  const [error, setError] = useState(null)
  
  // Student dropdown state
  const [students, setStudents] = useState([])
  const [loadingStudents, setLoadingStudents] = useState(false)
  const [studentInfo, setStudentInfo] = useState(null)
  const [loadingStudentInfo, setLoadingStudentInfo] = useState(false)

  const subjects = ['Українська мова', 'Алгебра', 'Історія України']
  const grades = [8, 9]

  // Fetch available students when subject/grade changes
  useEffect(() => {
    fetchStudents()
  }, [formData.subject, formData.grade])

  // Fetch student info when student_id changes
  useEffect(() => {
    if (formData.student_id) {
      fetchStudentInfo(formData.student_id)
    } else {
      setStudentInfo(null)
    }
  }, [formData.student_id, formData.subject])

  const fetchStudents = async () => {
    setLoadingStudents(true)
    try {
      const res = await axios.get(`${API_BASE}/students/list`, {
        params: {
          subject: formData.subject,
          grade: formData.grade
        }
      })
      setStudents(res.data.students || [])
    } catch (err) {
      console.error('Error fetching students:', err)
      setStudents([])
    } finally {
      setLoadingStudents(false)
    }
  }

  const fetchStudentInfo = async (studentId) => {
    setLoadingStudentInfo(true)
    try {
      const res = await axios.get(`${API_BASE}/students/${studentId}/info`, {
        params: {
          subject: formData.subject
        }
      })
      setStudentInfo(res.data)
    } catch (err) {
      console.error('Error fetching student info:', err)
      setStudentInfo(null)
    } finally {
      setLoadingStudentInfo(false)
    }
  }

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
        student_id: formData.student_id ? parseInt(formData.student_id) : null
      }

      const res = await axios.post(`${API_BASE}/tutor/query`, payload)
      console.log('API Response:', res.data)
      console.log('lecture_content:', res.data.lecture_content)
      console.log('matched_topics:', res.data.matched_topics)
      setResponse(res.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Помилка при відправці запиту')
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

  return (
    <div className="query-form-container">
      <div className="form-card">
        <h2>📚 Запит до AI Tutor</h2>
        <p className="form-description">
          Введіть запит для генерації лекційного матеріалу та практичних питань
        </p>

        <form onSubmit={handleSubmit} className="query-form">
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
              <label htmlFor="student_id">Учень (опціонально)</label>
              <select
                id="student_id"
                name="student_id"
                value={formData.student_id}
                onChange={handleChange}
              >
                <option value="">Оберіть учня...</option>
                {loadingStudents ? (
                  <option disabled>Завантаження...</option>
                ) : students.length === 0 ? (
                  <option disabled>Немає учнів для цього предмету/класу</option>
                ) : (
                  students.map(student => (
                    <option key={student.student_id} value={student.student_id}>
                      ID: {student.student_id} - Середній бал: {student.average_score.toFixed(1)} ({student.total_lessons} уроків)
                    </option>
                  ))
                )}
              </select>
            </div>
          </div>

          {studentInfo && (
            <div className="student-info-panel">
              <h3>📊 Інформація про учня</h3>
              {loadingStudentInfo ? (
                <div className="loading-text">Завантаження...</div>
              ) : (
                <div className="student-details">
                  <div className="student-summary">
                    <div className="summary-item">
                      <strong>ID учня:</strong> {studentInfo.student_id}
                    </div>
                    <div className="summary-item">
                      <strong>Загальний середній бал:</strong> {studentInfo.overall_average_score.toFixed(1)}/12
                    </div>
                    <div className="summary-item">
                      <strong>Всього уроків:</strong> {studentInfo.total_lessons}
                    </div>
                    <div className="summary-item">
                      <strong>Пропущено уроків:</strong> {studentInfo.total_absences}
                    </div>
                  </div>
                  
                  {studentInfo.subjects && studentInfo.subjects.length > 0 && (
                    <div className="subject-details">
                      {studentInfo.subjects.map((subject, idx) => (
                        <div key={idx} className="subject-item">
                          <h4>{subject.subject}</h4>
                          <div className="subject-stats">
                            <span className="stat-badge">
                              Середній бал: <strong>{subject.average_score.toFixed(1)}</strong>
                            </span>
                            <span className="stat-badge">
                              Уроків: {subject.total_lessons}
                            </span>
                            <span className="stat-badge">
                              Пропусків: {subject.total_absences}
                            </span>
                          </div>
                          
                          {subject.weak_topics && subject.weak_topics.length > 0 && (
                            <div className="topics-section">
                              <strong className="weak-topics">⚠️ Слабкі теми:</strong>
                              <ul className="topics-list">
                                {subject.weak_topics.slice(0, 5).map((topic, tidx) => (
                                  <li key={tidx}>{topic} ({subject.topic_breakdown[topic]?.toFixed(1) || 'N/A'})</li>
                                ))}
                              </ul>
                            </div>
                          )}
                          
                          {subject.strong_topics && subject.strong_topics.length > 0 && (
                            <div className="topics-section">
                              <strong className="strong-topics">✅ Сильні теми:</strong>
                              <ul className="topics-list">
                                {subject.strong_topics.slice(0, 5).map((topic, tidx) => (
                                  <li key={tidx}>{topic} ({subject.topic_breakdown[topic]?.toFixed(1) || 'N/A'})</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          <button 
            type="submit" 
            className="submit-btn"
            disabled={loading}
          >
            {loading ? '⏳ Завантаження...' : '🚀 Надіслати запит'}
          </button>
        </form>

        {error && (
          <div className="error-message">
            <strong>Помилка:</strong> {error}
          </div>
        )}

        {response && (
          <div className="response-container">
            <h3>✅ Результат</h3>
            
            {response.error && (
              <div className="error-message">
                <strong>Помилка:</strong> {response.error}
              </div>
            )}

            {/* Always show matched topics if they exist */}
            {response.matched_topics && Array.isArray(response.matched_topics) && response.matched_topics.length > 0 ? (
              <section className="response-section">
                <h4>🎯 Тема з RAG</h4>
                <div className="content-box">
                  {response.matched_topics.map((topic, idx) => (
                    <div key={idx} className="matched-topic">
                      <strong>{topic.topic || topic.topic || 'Невідома тема'}</strong>
                      {topic.grade && <span className="topic-meta"> (Клас: {topic.grade})</span>}
                      {topic.subject && <span className="topic-meta"> • {topic.subject}</span>}
                    </div>
                  ))}
                </div>
              </section>
            ) : (
              <section className="response-section">
                <div className="content-box" style={{color: '#999', fontStyle: 'italic'}}>
                  Тема з RAG не знайдена
                </div>
              </section>
            )}

            {/* Always show lecture content if it exists */}
            {response.lecture_content && response.lecture_content.trim() !== '' ? (
              <section className="response-section lecture-section">
                <h4>📝 Конспект</h4>
                <div className="content-box markdown-content lecture-content">
                  <ReactMarkdown
                    remarkPlugins={[remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                  >
                    {response.lecture_content}
                  </ReactMarkdown>
                </div>
              </section>
            ) : (
              <section className="response-section">
                <div className="content-box" style={{color: '#999', fontStyle: 'italic'}}>
                  Конспект не згенеровано (lecture_content: {response.lecture_content ? `"${response.lecture_content.substring(0, 50)}..."` : 'undefined'})
                </div>
              </section>
            )}

            {response.control_questions && response.control_questions.length > 0 && (
              <section className="response-section">
                <h4>❓ Контрольні питання</h4>
                <ul className="questions-list">
                  {response.control_questions.map((q, idx) => (
                    <li key={idx}>{q}</li>
                  ))}
                </ul>
              </section>
            )}

            {response.practice_questions && response.practice_questions.length > 0 && (
              <section className="response-section">
                <h4>✍️ Практичні питання</h4>
                <div className="practice-questions">
                  {response.practice_questions.map((q, idx) => (
                    <PracticeQuestion key={idx} question={q} questionIndex={idx} />
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

            {response.sources && response.sources.length > 0 && (
              <section className="response-section">
                <h4>📚 Джерела</h4>
                <ul className="sources-list">
                  {response.sources.map((source, idx) => (
                    <li key={idx}>{source}</li>
                  ))}
                </ul>
              </section>
            )}

            {(response.matched_pages && response.matched_pages.length > 0) && (
              <section className="response-section debug-section">
                <details>
                  <summary className="debug-summary">
                    🔍 Debug: Тексти сторінок з бази ({response.matched_pages.length})
                  </summary>
                  <div className="debug-content">
                    {response.matched_pages.map((page, idx) => (
                      <div key={idx} className="page-content">
                        <div className="page-header">
                          <strong>Сторінка {idx + 1}</strong>
                        </div>
                        <div className="page-text">
                          {typeof page === 'object' && page.content ? page.content : String(page)}
                        </div>
                      </div>
                    ))}
                  </div>
                </details>
              </section>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default QueryForm
