from fastapi import FastAPI, Request
import base64
from ASRManager import ASRManager

app = FastAPI()

asr_manager = ASRManager()


#with open('audio_audio_0.wav', "rb") as file:
#    audio_bytes = file.read()
#    audio_bytes=base64.b64encode(audio_bytes)
#temp = asr_manager.transcribe(audio_bytes)


@app.get("/health")
def health():
    return {"message": "health ok"}


@app.post("/stt")
async def stt(request: Request):
    """
    Performs ASR given the filepath of an audio file
    Returns transcription of the audio
    """

    # get base64 encoded string of audio, convert back into bytes
    input_json = await request.json()

    predictions = []
    for instance in input_json["instances"]:
        # each is a dict with one key "b64" and the value as a b64 encoded string
        
        #should not need the below code as decoding is done in ASRManager.transcribe. Uncomment if error. 
        #audio_bytes = base64.b64decode(instance["b64"])

        transcription = asr_manager.transcribe(instance["b64"])
        predictions.append(transcription)

    return {"predictions": predictions}