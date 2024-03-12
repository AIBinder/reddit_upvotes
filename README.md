To run:
- install virtual python env with packages in requirements.txt
- download pytorch_model.bin from https://huggingface.co/AI-Binder/reddit_upvote_prediction/
- export FLASK_APP=flask_endpoint.py; flask run
- streamlit run app.py

Then the app should be accessible under http://localhost:8501.

To train the model on German reddit data, see:
train_upvotes_gpu_r_edv.py
(based on deepset gbert model, for English subreddits use English language model accordingly)