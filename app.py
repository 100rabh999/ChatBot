from flask import Flask, render_template, request, redirect, url_for
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

def save_intents_data():
    # Save updated intents data back to the file
    with open('dataset/intents1.json', 'w') as f:
        json.dump(intents, f, indent=4)

def train_chatbot_model():
    # Extract patterns and responses from intents data
    all_patterns = []
    all_responses = []

    for intent in intents['intents']:
        all_patterns.extend(intent['patterns'])
        all_responses.extend(intent['responses'])

    # Check lengths
    print("Number of patterns:", len(all_patterns))
    print("Number of responses:", len(all_responses))

    # Train the model with the updated data
    input_text = vectorizer.transform(all_patterns)
    print("Shape of input_text:", input_text.shape)

    try:
        # Attempt to fit the model
        best_model.fit(input_text, all_responses)
        print("Training successful.")
    except ValueError as e:
        print("Error:", e)
        print("Make sure patterns and responses have consistent lengths.")

    
def chatbot_response(user_input):
    input_text = vectorizer.transform([user_input])
    predicted_intent = best_model.predict(input_text)[0]

    response = None
    for intent in intents['intents']:
        if intent['tag'] == predicted_intent:
            response = random.choice(intent['responses'])
            break

    if response is None:
        response = "That's a great question! While I don't have the answer readily available, You can contact College office at Ground floor in college for your query."

    return response

@app.route('/')
def demo_website():
    return render_template('index.html')

@app.route('/chatbot')
def home():
    return render_template('chatbot.html')
@app.route('/help')
def help():
    return render_template('help.html')
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'admin' and password == 'adminpassword':
            return redirect(url_for('edit_intents'))
        else:
            error = 'Invalid admin credentials'

    return render_template('admin_login.html', error=error if 'error' in locals() else None)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    print(user_input)
    response = chatbot_response(user_input)
    return response
@app.route('/edit_intents', methods=['GET', 'POST', 'PUT'])
def edit_intents():
    if request.method == 'POST' or request.method == 'PUT':
        action = request.form.get('_method')

        
        if action == 'add_new':
            new_tag = request.form['new_tag']
            new_patterns = request.form['new_patterns'].split('\n')
            new_responses = request.form['new_responses'].split('\n')

            intents['intents'].append({
                'tag': new_tag,
                'patterns': new_patterns,
                'responses': new_responses
            })
            save_intents_data()
            train_chatbot_model()

            return redirect(url_for('edit_intents'))

        elif action == 'update_existing':
            num_intents = len(intents['intents'])
            for i in range(num_intents):
                tag = request.form.get(f'tag_{i+1}')
                if tag:
                    new_patterns = request.form.get(f'patterns_{i+1}').split('\n')
                    new_responses = request.form.get(f'responses_{i+1}').split('\n')
                    intents['intents'][i]['patterns'] = new_patterns
                    intents['intents'][i]['responses'] = new_responses
                    save_intents_data()
                    train_chatbot_model()

                    return redirect(url_for('edit_intents'))

    return render_template('edit_intents.html', intents=intents['intents'])

if __name__ == '__main__':
    app.run(debug=True)
