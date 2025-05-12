"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
import spacy
import re


class Tokenizer:
    def __init__(self):
        try:
            self.spacy_en = spacy.load('en_core_web_sm')
            self.spacy_de = spacy.load('de_core_news_sm')
        except:
            print("Failed to load spaCy models, using simple tokenization...")
            self.spacy_en = None
            self.spacy_de = None

    def tokenize_en(self, text):
        if self.spacy_en:
            return [tok.text for tok in self.spacy_en.tokenizer(text)]
        else:
            return re.findall(r'\w+|[^\w\s]', text)

    def tokenize_de(self, text):
        if self.spacy_de:
            return [tok.text for tok in self.spacy_de.tokenizer(text)]
        else:
            return re.findall(r'\w+|[^\w\s]', text)
