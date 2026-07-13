import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings only once
df = joblib.load("embeddings.joblib")


def create_embedding(text):

    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": [text]
        }
    )

    data = r.json()

    return data["embeddings"][0]


def inference(prompt):

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    return r.json()["response"]


def ask_question(question):

    question_embedding = create_embedding(question)

    similarities = cosine_similarity(
        np.vstack(df["embedding"]),
        [question_embedding]
    ).flatten()

    top_results = 6

    indices = similarities.argsort()[::-1][:top_results]

    retrieved = df.loc[indices]

    prompt = f"""
I am teaching web development in my Sigma Web Development course.

Here are lecture chunks:

{retrieved[['title','number','start','end','text']].to_json(orient='records')}

------------------------------------------------

User Question:

{question}

Instructions:

Answer naturally.

Mention:

• Which lecture contains the answer

• Video number

• Timestamp

If the question is unrelated to the course, politely tell the user you can only answer course-related questions.
"""

    answer = inference(prompt)

    with open("prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    with open("response.txt", "w", encoding="utf-8") as f:
        f.write(answer)

    return answer, retrieved