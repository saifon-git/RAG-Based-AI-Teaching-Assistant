import whisper
import json
import os

model = whisper.load_model("base")

audio = os.listdir("audio")

for i in audio:
    if "_" in i: 
        number = i.split("_")[0]
        title = i.split("_")[1][:-4]   # Remove .mp3

        print(number, title)

        result = model.transcribe(
            audio=f"audio/{i}",
            language="hi",
            task="translate",
            word_timestamps=False
        )


        chunks = []
        for segment in result["segments"]:
            chunks.append({"number": number, "title": title, "start": segment["start"], "end": segment["end"], "text": segment["text"]})
        
        chunks_with_metadata = {"chunks": chunks, "text": result["text"]}

        with open(f"jsons/{i}.json", "w") as f:
            json.dump(chunks_with_metadata,f,indent=4)