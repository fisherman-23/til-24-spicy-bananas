from typing import Dict
import spacy
import json
from spacy.tokens import DocBin

import pandas as pd
import os
from tqdm import tqdm


class NLPManager:
    def __init__(self):
        # initialize the model here
        
        self.nlp_ner = spacy.load("model-best")
        self.string_to_numeric = {'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7','eight':'8','nine':'9','niner':'9'}

    def qa(self, context: str) -> Dict[str, str]:
        try:
            doc = self.nlp_ner(context)
            result = {"heading": "", "tool": "", "target": ""}
            for ent in doc.ents:

                if ent.label_.lower() == 'heading':
                    temp = ent.text
                    temp = temp.split(' ')
                    temp = ''.join([self.string_to_numeric[x] for x in temp])
                    result[ent.label_.lower()] = temp
                else:
                    result[ent.label_.lower()] = ent.text

        except:
            return {"heading": "", "tool": "", "target": ""}
        # perform NLP question-answering
        
        return result
