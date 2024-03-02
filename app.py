from flask import Flask, render_template, request
import pickle
import json
import random

app = Flask(__name__)

# Load the trained model and vectorizer
with open('model/chatbot_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load the intents data
with open('dataset/intents1.json', 'r') as f:
    intents = json.load(f)

def chatbot_response(user_input):
    input_text = vectorizer.transform([user_input])
    predicted_intent = best_model.predict(input_text)[0]

    response = None
    for intent in intents['intents']:
        if intent['tag'] == predicted_intent:
            response = random.choice(intent['responses'])
            break

    # If no matching intent is found, provide the fallback response
    if response is None:
        response = "That's a great question! While I don't have the answer readily available, I can point you in the right direction. Would you like me to provide contact information for the college office who may be able to assist you?"

    return response

@app.route('/')
def demo_website():
    return render_template('index.html')

@app.route('/chatbot')
def home():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = chatbot_response(user_input)
    return response

if __name__ == '__main__':
    app.run(debug=True)