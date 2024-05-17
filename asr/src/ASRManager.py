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
        self.model = WhisperModel("medium.en")

        #defines model type
        #tokenizer that blocks numeric tokens (makes it not output number in numeric form)
        token = tokenizer.Tokenizer(tokenizer=self.model.hf_tokenizer, multilingual=False)
        self.number_tokens = [
            i 
            for i in range(token.eot)
            if all(c in "-" for c in token.decode([i]).removeprefix(" "))
        ]
        self.numeric_to_string = {'0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five','6':'six','7':'seven','8':'eight','9':'niner'}
        #tokenizer that blocks numeric tokens (makes it not output number in numeric form)

        pass

    def transcribe(self, audio_bytes: bytes) -> str:
        # perform ASR transcription
        
        #decode B64 Encoded Audio and convert into NDARRAY
        audio_bytes = base64.b64decode(audio_bytes)
        array = np.frombuffer(audio_bytes, np.int16).flatten().astype(np.float32) / 32768.0
        segments, info = self.model.transcribe(array,suppress_tokens=[-1] + self.number_tokens)
        result = ''
        for segment in segments:
            result += segment.text
        result = result.replace(',','').replace('.','')
        list_of_words = result.split(' ')
        length = len(list_of_words)
        for i in range(length):
            try:
                if list_of_words[i].isdigit() == True:
                    if len(list_of_words[i])>1:
                        temp = list_of_words[i]
                        temp = list(temp.replace(' ',''))
                        for x in range(len(temp)):
                            temp[x] = self.numeric_to_string[temp[x]]
                        list_of_words[i] = ' '.join(temp)
                    else:
                        if list_of_words[i] == '2':
                            if list_of_words[i-1].isdigit() == True or list_of_words[i+1].isdigit() == True:
                                list_of_words[i]=self.numeric_to_string[list_of_words[i]]
                            else:
                                list_of_words[i] = 'tool'

                elif list_of_words[i] == 'as' or list_of_words[i] == 'at':
                    list_of_words[i] = 'is'
            except Exception as e:
                continue 
                

        result = ' '.join(list_of_words)
        return result.strip()
