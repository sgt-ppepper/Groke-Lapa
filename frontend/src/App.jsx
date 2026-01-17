import React, { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, useNavigate } from 'react-router-dom'
import QueryForm from './components/QueryForm'
import AnswerCheck from './components/AnswerCheck'
import BenchmarkSolver from './components/BenchmarkSolver'
import './App.css'

function App() {
  return (
    <Router>
      <div className="app">
        <nav className="navbar">
          <div className="nav-container">
            <h1 className="nav-title">🎓 Mriia AI Tutor</h1>
            <div className="nav-links">
              <Link to="/" className="nav-link">Запит</Link>
              <Link to="/check-answers" className="nav-link">Перевірка відповідей</Link>
              <Link to="/benchmark" className="nav-link">Benchmark</Link>
            </div>
          </div>
        </nav>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<QueryForm />} />
            <Route path="/check-answers" element={<AnswerCheck />} />
            <Route path="/benchmark" element={<BenchmarkSolver />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
