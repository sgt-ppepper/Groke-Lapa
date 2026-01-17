import React, { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import './AnswerCheck.css'

const AnswerCheck = () => {
  // Load questions from localStorage (saved from QueryForm)
  const [questions, setQuestions] = useState([])
  const [userAnswers, setUserAnswers] = useState({})
  const [submitted, setSubmitted] = useState(false)
  const [score, setScore] = useState({ correct: 0, total: 0 })

  // Load saved questions when component mounts
  useEffect(() => {
    const savedResponse = localStorage.getItem('queryResponse')
    if (savedResponse) {
      try {
        const response = JSON.parse(savedResponse)
        if (response.practice_questions && response.practice_questions.length > 0) {
          setQuestions(response.practice_questions)
          // Initialize empty answers
          const initialAnswers = {}
          response.practice_questions.forEach((_, idx) => {
            initialAnswers[idx] = ''
          })
          setUserAnswers(initialAnswers)
        }
      } catch (err) {
        console.error('Error loading questions:', err)
      }
    }
  }, [])

  const handleAnswerSelect = (questionIndex, answerLetter) => {
    if (submitted) return // Don't allow changes after submission
    setUserAnswers(prev => ({
      ...prev,
      [questionIndex]: answerLetter
    }))
  }

  const handleSubmit = () => {
    if (Object.values(userAnswers).some(a => a === '')) {
      alert('Будь ласка, дайте відповіді на всі питання')
      return
    }

    // Calculate score
    let correct = 0
    questions.forEach((q, idx) => {
      if (userAnswers[idx] === q.correct_answer) {
        correct++
      }
    })

    setScore({ correct, total: questions.length })
    setSubmitted(true)
  }

  const handleReset = () => {
    setSubmitted(false)
    const resetAnswers = {}
    questions.forEach((_, idx) => {
      resetAnswers[idx] = ''
    })
    setUserAnswers(resetAnswers)
    setScore({ correct: 0, total: 0 })
  }

  const getCardClass = (questionIndex, optionLetter) => {
    if (!submitted) {
      return userAnswers[questionIndex] === optionLetter ? 'selected' : ''
    }

    const correctAnswer = questions[questionIndex].correct_answer
    const userAnswer = userAnswers[questionIndex]

    if (optionLetter === correctAnswer) {
      return 'correct'
    }
    if (optionLetter === userAnswer && userAnswer !== correctAnswer) {
      return 'incorrect'
    }
    return ''
  }

  const getQuestionCardClass = (questionIndex) => {
    if (!submitted) return ''

    const correctAnswer = questions[questionIndex].correct_answer
    const userAnswer = userAnswers[questionIndex]

    return userAnswer === correctAnswer ? 'card-correct' : 'card-incorrect'
  }

  if (questions.length === 0) {
    return (
      <div className="answer-check-container">
        <div className="form-card">
          <h2>✅ Перевірка відповідей</h2>
          <div className="empty-state">
            <div className="empty-icon">📝</div>
            <h3>Немає питань для перевірки</h3>
            <p>
              Спочатку перейдіть на сторінку <strong>"Запит"</strong> та згенеруйте
              лекційний матеріал з практичними питаннями.
            </p>
            <a href="/" className="go-to-query-btn">
              🚀 Перейти до Запитів
            </a>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="answer-check-container">
      <div className="form-card">
        <h2>✅ Перевірка відповідей</h2>
        <p className="form-description">
          Оберіть правильні відповіді для кожного питання та натисніть "Перевірити"
        </p>

        {submitted && (
          <div className={`score-panel ${score.correct === score.total ? 'perfect' : score.correct >= score.total / 2 ? 'good' : 'needs-work'}`}>
            <div className="score-icon">
              {score.correct === score.total ? '🎉' : score.correct >= score.total / 2 ? '👍' : '📚'}
            </div>
            <div className="score-text">
              <h3>Результат: {score.correct} / {score.total}</h3>
              <p>
                {score.correct === score.total
                  ? 'Відмінно! Всі відповіді правильні!'
                  : score.correct >= score.total / 2
                    ? 'Добре! Але є ще над чим працювати.'
                    : 'Потрібно повторити матеріал. Не здавайся!'}
              </p>
            </div>
            <button onClick={handleReset} className="reset-btn">
              🔄 Спробувати ще раз
            </button>
          </div>
        )}

        <div className="questions-grid">
          {questions.map((question, qIdx) => (
            <div key={qIdx} className={`question-card ${getQuestionCardClass(qIdx)}`}>
              <div className="question-header">
                <span className="question-number">Питання {qIdx + 1}</span>
                {submitted && (
                  <span className={`question-result ${userAnswers[qIdx] === question.correct_answer ? 'correct' : 'incorrect'}`}>
                    {userAnswers[qIdx] === question.correct_answer ? '✓ Правильно' : '✗ Неправильно'}
                  </span>
                )}
              </div>

              <div className="question-text">
                <ReactMarkdown
                  remarkPlugins={[remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                >
                  {question.question}
                </ReactMarkdown>
              </div>

              <div className="options-grid">
                {question.options?.map((option, optIdx) => {
                  const letter = String.fromCharCode(65 + optIdx)
                  return (
                    <button
                      key={optIdx}
                      className={`option-btn ${getCardClass(qIdx, letter)}`}
                      onClick={() => handleAnswerSelect(qIdx, letter)}
                      disabled={submitted}
                    >
                      <span className="option-letter">{letter}</span>
                      <span className="option-text">
                        <ReactMarkdown
                          remarkPlugins={[remarkMath]}
                          rehypePlugins={[rehypeKatex]}
                        >
                          {option}
                        </ReactMarkdown>
                      </span>
                    </button>
                  )
                })}
              </div>

              {submitted && userAnswers[qIdx] !== question.correct_answer && question.explanation && (
                <div className="explanation-box">
                  <strong>💡 Пояснення:</strong>
                  <ReactMarkdown
                    remarkPlugins={[remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                  >
                    {question.explanation}
                  </ReactMarkdown>
                </div>
              )}
            </div>
          ))}
        </div>

        {!submitted && (
          <button
            onClick={handleSubmit}
            className="submit-btn"
            disabled={Object.values(userAnswers).some(a => a === '')}
          >
            🔍 Перевірити відповіді
          </button>
        )}
      </div>
    </div>
  )
}

export default AnswerCheck
