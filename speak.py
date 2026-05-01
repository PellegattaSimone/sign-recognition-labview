import pyttsx3

def speak_word(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    return "Done"