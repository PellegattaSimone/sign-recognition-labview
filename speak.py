import pyttsx3

def speak_word(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 100)

    engine.say(text)
    engine.runAndWait()
    return "Done"