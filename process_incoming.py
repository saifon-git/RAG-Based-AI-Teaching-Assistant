import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity


def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    ) 

    data = r.json()

    if "embeddings" not in data:
        print("Error:", data)
        return None

    return data["embeddings"]

def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            # "model": "deepseek-r1",
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    ) 
     
    response = r.json() 
    return response
     
    

# Load saved embeddings
df = joblib.load("embeddings.joblib")

incoming_query = input("Ask a Question: ")

question_embedding = create_embedding([incoming_query])

if question_embedding is None:
    raise Exception("Failed to generate question embedding.")

question_embedding = question_embedding[0]

similarities = cosine_similarity(
    np.vstack(df["embedding"]),
    [question_embedding]
).flatten()

top_results = 6
max_indx = similarities.argsort()[::-1][:top_results]

new_df = df.loc[max_indx]

# print(new_df[["title", "number", "text"]])
prompt = f'''I am teaching web development in my Sigma wed development course. Here are video subtitle chucks contaning video title, video number, start time in seconds, end time in seconds, the text at that time:
{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
---------------------------------------------
"{incoming_query}"
user asked this question realted to the video chucks, you have to answer in a human way (dont mension the above format, its only just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrealted question, tell him that you can only answer questions realted to the course
 '''
with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)


with open("response.txt", "w") as f:
    f.write(response)

# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])