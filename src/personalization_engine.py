import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class PersonalizationEngine:
    def __init__(self, df_scores: pd.DataFrame, df_absences: pd.DataFrame):
        print("🔧 Initializing Personalization Engine...")
        
        #subjects used for benchmark
        target_subjects = ['Алгебра', 'Українська мова', 'Історія України']
        
        #filtering data
        self.df_scores = df_scores[df_scores['discipline_name'].isin(target_subjects)].copy()
        self.df_absences = df_absences[df_absences['discipline_name'].isin(target_subjects)].copy()
        
        #data cleaning
        self.df_scores['score_numeric'] = pd.to_numeric(self.df_scores['score_numeric'], errors='coerce')
        self.df_scores['lesson_date'] = pd.to_datetime(self.df_scores['lesson_date'])
        self.df_absences['lesson_date'] = pd.to_datetime(self.df_absences['lesson_date'])
        
        self.df_scores = self.df_scores.dropna(subset=['score_numeric'])
        print(f"✅ Ready! Loaded {len(self.df_scores)} scores.")

    def _find_similar_topics_cosine(self, target_topic: str, subject: str, threshold=0.45) -> list:
        """
        Finds similar topics using cosine similarity.
        """
        #take all unique topics from scores database
        journal_topics = self.df_scores[
            self.df_scores['discipline_name'] == subject
        ]['topic_name'].unique().tolist()
        
        #making strings
        journal_topics = [str(t) for t in journal_topics if t is not None]
        
        if not journal_topics:
            return []

        #first element is target topic, and other elements are all topics of this subject
        all_documents = [target_topic] + journal_topics
        
        #creating vectors from topics
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_documents)
        
        #counting cosine similarity of first element with others
        cosine_sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        
        #filtering results
        matched_topics = []
        
        #sorting all topics
        sorted_indices = cosine_sims.argsort()[::-1]
        
        found_any = False
        for idx in sorted_indices:
            score = cosine_sims[idx]
            topic_name = journal_topics[idx]
            
            if score > threshold:
                matched_topics.append(topic_name)
                found_any = True
            
            # Для демо покажемо топ-3 навіть якщо вони слабкі, щоб ти бачив, що воно рахує

        if not matched_topics:
            print("   ⚠️ No strong matches found. Using full subject history.")
            
        return matched_topics

    def get_student_context(self, student_id: int, subject: str, topic_from_router: str) -> dict:
        # --- PHASE 1: SEARCH & FILTERING (SCOPE) ---
        
        # 1. Semantic search for relevant topics
        relevant_topics = self._find_similar_topics_cosine(topic_from_router, subject)
        
        # 2. Filter dataframes by topic
        if relevant_topics:
            topic_filter = self.df_scores['topic_name'].isin(relevant_topics)
            absence_filter = self.df_absences['topic_name'].isin(relevant_topics)
        else:
            topic_filter = pd.Series([False] * len(self.df_scores))
            absence_filter = pd.Series([False] * len(self.df_absences))

        active_scores = self.df_scores[
            (self.df_scores['student_id'] == student_id) & 
            (self.df_scores['discipline_name'] == subject) &
            topic_filter
        ]
        
        active_absences = self.df_absences[
            (self.df_absences['student_id'] == student_id) & 
            (self.df_absences['discipline_name'] == subject) &
            absence_filter
        ]

        # 3. Fallback Logic (if no specific data found)
        scope_type = "SPECIFIC_TOPIC"
        is_fallback = False
        
        if active_scores.empty and active_absences.empty:
            print(f"⚠️ No specific data for '{topic_from_router}'. Switching to General Stats.")
            is_fallback = True
            scope_type = "GENERAL_SUBJECT_FALLBACK"
            
            # Fetch entire subject history
            active_scores = self.df_scores[
                (self.df_scores['student_id'] == student_id) & 
                (self.df_scores['discipline_name'] == subject)
            ]
            # Ignore specific absences in general mode
            active_absences = pd.DataFrame() 

            if active_scores.empty:
                return {"error": "No data", "prompt_injection": "Новий учень. Пояснюй з нуля."}

        # --- PHASE 2: SCORE ANALYTICS ---
        
        avg_score = active_scores['score_numeric'].mean()
        min_score = active_scores['score_numeric'].min()
        max_score = active_scores['score_numeric'].max()
        grades_count = len(active_scores)
        
        # Topic breakdown (Average score per topic)
        breakdown = active_scores.groupby('topic_name')['score_numeric'].mean().round(1).sort_values()
        weak_topics_list = breakdown[breakdown < 6].index.tolist()
        strong_topics_list = breakdown[breakdown > 9].index.tolist()

        # --- PHASE 3: ATTENDANCE ANALYTICS ---
        
        missed_count = len(active_absences)
        missed_details_list = [] # List for JSON output
        missed_prompt_str = ""   # String for LLM prompt
        last_missed_date = None

        if not active_absences.empty:
            # Sort by date
            sorted_absences = active_absences.sort_values('lesson_date')
            last_missed_date = sorted_absences['lesson_date'].max()
            
            # Format: "YYYY-MM-DD: Topic Name"
            for _, row in sorted_absences.iterrows():
                date_str = str(row['lesson_date']).split(' ')[0] # Extract date only
                topic_name = row['topic_name']
                missed_details_list.append(f"{date_str}: {topic_name}")
            
            # Aggregate into prompt string
            missed_prompt_str = "; ".join(missed_details_list)

        # --- PHASE 4: PROMPT GENERATION ---
        
        prompt_lines = []
        
        # 1. Context Header
        prompt_lines.append(f"📌 КОНТЕКСТ: {scope_type} ({topic_from_router}).")
        
        # 2. Student Archetype (Global Status)
        # Determine persona regardless of absences
        student_archetype = ""
        base_strategy = ""
        
        if avg_score < 5:
            student_archetype = "🔴 СТАТУС: Учень СЛАБКИЙ (Low Performer). Йому важко дається матеріал."
            base_strategy = "Пояснюй максимально просто, крок за кроком (Step-by-step). Використовуй аналогії. Не давай складних термінів одразу."
        elif avg_score >= 10:
            student_archetype = "🟢 СТАТУС: Учень СИЛЬНИЙ (High Performer)."
            base_strategy = "Можна тримати високий темп. Пропускай очевидні речі."
        else:
            student_archetype = "🟡 СТАТУС: Учень СЕРЕДНЬОГО рівня."
            base_strategy = "Використовуй збалансований підхід."

        prompt_lines.append(f"{student_archetype} Рівень: {avg_score:.1f}/12.")

        # 3. Score Details
        if len(breakdown) > 6:
            worst_str = ", ".join([f"{t}({s})" for t, s in breakdown.head(3).items()])
            best_str = ", ".join([f"{t}({s})" for t, s in breakdown.tail(2).items()])
            prompt_lines.append(f"📝 ОЦІНКИ: Слабкі місця: [{worst_str}] ... Сильні: [{best_str}].")
        else:
            full_str = "; ".join([f"{t}({s})" for t, s in breakdown.items()])
            prompt_lines.append(f"📝 ОЦІНКИ: {full_str}.")

        # 4. Strategy Formulation (Combined)
        strategy_parts = []
        
        # Add base strategy (Performance-based)
        strategy_parts.append(base_strategy)

        # Add situational strategy (Absence/Gap-based)
        if missed_count > 0:
            prompt_lines.append(f"⛔ ПРОПУЩЕНІ УРОКИ ({missed_count}): [{missed_prompt_str}].")
            strategy_parts.append("КРІМ ТОГО: Учень не був на цих уроках. Почни з ТЕОРІЇ саме по пропущених темах, перш ніж переходити до практики.")
        elif weak_topics_list:
            strategy_parts.append(f"ФОКУС: Є конкретні прогалини ({', '.join(weak_topics_list[:2])}). Дай додаткові тести на це.")
        
        # Combine into final string
        final_strategy = " ".join(strategy_parts)
        prompt_lines.append(f"⚠️ ІНСТРУКЦІЯ ДЛЯ ВЧИТЕЛЯ: {final_strategy}")

        final_prompt = " ".join(prompt_lines)

        # --- PHASE 5: RETURN RICH JSON ---
        return {
            "meta": {
                "student_id": int(student_id),
                "subject": subject,
                "scope_type": scope_type,
                "is_fallback": is_fallback
            },
            "metrics": {
                "average_score": round(avg_score, 2),
                "min_score": float(min_score),
                "max_score": float(max_score),
                "grades_count": int(grades_count)
            },
            "enrichment": {
                "full_topic_breakdown": breakdown.to_dict(),
                "weak_topics": weak_topics_list,
                "strong_topics": strong_topics_list,
            },
            "attendance": {
                "missed_count": int(missed_count),
                "last_missed_date": str(last_missed_date.date()) if last_missed_date else None,
                "missed_lessons_details": missed_details_list
            },
            "prompt_injection": final_prompt
        }