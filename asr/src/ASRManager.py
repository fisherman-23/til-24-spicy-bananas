from faster_whisper import WhisperModel
from faster_whisper import tokenizer
import time
import os
import base64
import numpy as np
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class ASRManager:
    def __init__(self):
        # initialize the model here
        #defines model type
        self.model = WhisperModel("small.en")

        #tokenizer that blocks numeric tokens (makes it not output number in numeric form)
        token = tokenizer.Tokenizer(tokenizer=self.model.hf_tokenizer, multilingual=False)
        self.number_tokens = [
            i 
            for i in range(token.eot)
            if all(c in "0123456789" for c in token.decode([i]).removeprefix(" "))
        ]

        pass

    def transcribe(self, audio_bytes: bytes) -> str:
        # perform ASR transcription
        
        #decode B64 Encoded Audio and convert into NDARRAY
        audio_bytes = base64.b64decode(audio_bytes)
        array = np.frombuffer(audio_bytes, np.int16).flatten().astype(np.float32) / 32768.0
        segments, info = self.model.transcribe(array, suppress_tokens=[-1] + self.number_tokens)
        result = ''
        for segment in segments:
            result += segment.text
        return result.strip()
