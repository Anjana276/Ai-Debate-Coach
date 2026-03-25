from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from groq import Groq
import os
import uuid
from datetime import datetime

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///debate.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ====================== MODELS ======================
class DebateSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), unique=True, nullable=False)
    topic = db.Column(db.String(200), nullable=False)
    difficulty = db.Column(db.Integer, default=2)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Performance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), nullable=False)
    user_text = db.Column(db.Text)
    ai_text = db.Column(db.Text)
    fillers = db.Column(db.Integer, default=0)
    eye_contact = db.Column(db.Integer, default=85)
    strength_score = db.Column(db.Integer, default=70)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ====================== GROQ SETUP ======================
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# ====================== IMPROVED GROQ FUNCTION ======================
def get_groq_response(topic, user_text, difficulty, history):
    difficulty_name = ['Beginner', 'Intermediate', 'Advanced'][difficulty - 1]

    system_prompt = f"""You are a sharp, respectful, and highly focused debate opponent.
Topic: {topic}
Difficulty level: {difficulty_name}

Rules you MUST follow:
- Always respond DIRECTLY to the user's latest argument. Never ignore what they just said.
- Point out specific flaws, weaknesses, or missing evidence in their statement.
- Stay strictly on the given topic. Do not change the subject.
- Never start with "Interesting point", "That's a good point", "You're right", etc.
- Keep your counter-argument between 70-110 words.
- Be logical, evidence-based, and polite but firm.

After your counter-argument, add exactly this separator on a new line:
---
Then give a short, helpful coaching tip (1-2 sentences)."""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.65,
            max_tokens=750,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Groq Error:", str(e))
        return "Sorry, the AI service is temporarily unavailable.\n---\nPlease try again in a few seconds."


# ====================== ROUTES ======================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/start_debate', methods=['POST'])
def start_debate():
    data = request.get_json()
    topic = data.get('topic', 'Should governments regulate AI development?')
    difficulty = int(data.get('difficulty', 2))

    session_id = str(uuid.uuid4())

    new_session = DebateSession(session_id=session_id, topic=topic, difficulty=difficulty)
    db.session.add(new_session)
    db.session.commit()

    return jsonify({
        "session_id": session_id,
        "ai_greeting": "Hello! I’m your AI debate opponent. You go first – make your opening argument."
    })


@app.route('/api/user_turn', methods=['POST'])
def user_turn():
    data = request.get_json()
    session_id = data.get('session_id')
    user_text = data.get('user_text', '').strip()
    eye_contact = data.get('eye_contact', 85)

    if not session_id or not user_text:
        return jsonify({"error": "Missing session_id or user_text"}), 400

    session = DebateSession.query.filter_by(session_id=session_id).first()
    if not session:
        return jsonify({"error": "Session not found"}), 404

    # Get conversation history
    past = Performance.query.filter_by(session_id=session_id)\
        .order_by(Performance.timestamp.asc()).limit(8).all()

    history = []
    for p in past:
        if p.user_text:
            history.append({"role": "user", "content": p.user_text})
        if p.ai_text:
            history.append({"role": "assistant", "content": p.ai_text})

    # Get AI response
    full_response = get_groq_response(session.topic, user_text, session.difficulty, history)

    # Split into argument and coaching tip
    if '---' in full_response:
        parts = [p.strip() for p in full_response.split('---', 1)]
        ai_response = parts[0]
        coach_tip = parts[1] if len(parts) > 1 else "Good effort!"
    else:
        ai_response = full_response
        coach_tip = "Good effort! Focus on directly addressing the opponent's points."

    # Metrics
    filler_list = ['um', 'uh', 'like', 'you know', 'actually', 'basically', 'so', 'well']
    fillers = sum(1 for word in filler_list if word in user_text.lower())

    strength = max(45, min(98, 72 + (len(user_text) // 10) - fillers * 8))

    # Save performance
    perf = Performance(
        session_id=session_id,
        user_text=user_text,
        ai_text=ai_response,
        fillers=fillers,
        eye_contact=eye_contact,
        strength_score=strength
    )
    db.session.add(perf)
    db.session.commit()

    return jsonify({
        "ai_response": ai_response,
        "coach_tip": coach_tip,
        "metrics": {
            "fillers": fillers,
            "eye_contact": eye_contact,
            "strength_score": strength
        }
    })


@app.route('/test_groq')
def test_groq():
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say hello and one fun fact about AI."}],
            max_tokens=100
        )
        return jsonify({"status": "OK", "response": res.choices[0].message.content.strip()})
    except Exception as e:
        return jsonify({"status": "FAILED", "error": str(e)}), 500


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)