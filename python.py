import requests

url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
response = requests.get(url)

# Guardar el archivo en el disco
with open("gesture_recognizer.task", "wb") as f:
    f.write(response.content)

print("Modelo descargado con éxito")
