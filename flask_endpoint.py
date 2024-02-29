from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

app = Flask(__name__)

device = torch.device('cpu')

model = AutoModelForSequenceClassification.from_pretrained("deepset/gbert-base", num_labels=5)

# Load the model
model.load_state_dict(torch.load("pytorch_model.bin", map_location=device))
model.eval()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")

@app.route('/upvotes', methods=['POST'])
def eval_post():
    try:
        question_data = request.get_json()
        title = question_data.get("title")
        username = question_data.get("author")
        text = question_data.get("text")

        post_combined = "title: " + title + " \n " + "author: " + username +  " \n " +  "text: " + text
        if len(text)>10:
            # Tokenize the post
            inputs = tokenizer(post_combined, return_tensors="pt")

            # Make a prediction
            outputs = model(**inputs)
            logits = outputs.logits

            # Get the predicted class
            predicted_class = np.argmax(logits.detach().numpy())
        else:
            predicted_class = 0

        convert_dict = {   
            0: "0-1",
            1: "2-5",
            2: "6-10",
            3: "11-29",
            4: "30+"}
        
        prediction = convert_dict[predicted_class]

        return_json = {
            "output": f"This post is predicted to receive {prediction} net upvotes on the r/de_EDV subreddit."}

        return jsonify(return_json), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)