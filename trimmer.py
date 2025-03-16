import os
from pydub import AudioSegment
from io import BytesIO

folder_path = "db1"
i = 1


def trim_mp3(file_path, duration_ms=20000):
    with open(file_path, "rb") as f:
        audio = AudioSegment.from_file(BytesIO(f.read()), format="mp3")
    trimmed_audio = audio[:duration_ms]
    trimmed_audio.export(file_path, format="mp3")


for filename in os.listdir(folder_path):
    if filename.endswith(".mp3"):
        file_path = os.path.join(folder_path, filename)
        print(f"Processing {filename}")
        trim_mp3(file_path)

print("Done!")
