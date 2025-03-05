from flask import Flask, request, jsonify
import google.generativeai as genai
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

genai.configure(api_key='AIzaSyCTOV6crU3iVqyO5vMHeJKsoH_z1g-jUZw')

@app.route('/suggest-questions', methods=['POST'])
def suggest_questions():
    data = request.json
    chat_history = data.get('chat_history', [])
    current_question = data.get('current_question', '')
    
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""
    Given this chat history: {chat_history[-3:]}
    And the latest question: {current_question}
    Suggest 2 concise follow-up questions that a manufacturing analyst might ask next.
    Return only the questions separated by newlines.
    """
    
    response = model.generate_content(prompt)
    questions = [q.strip() for q in response.text.split('\n') if q.strip()]
    return jsonify({'suggestions': questions[:2]})

if __name__ == '__main__':
    app.run(port=5001)