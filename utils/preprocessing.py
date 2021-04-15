import re


class Preprocessor:

    instance = None

    def __init__(self, str_replacements=[]):
        self.str_replacements = str_replacements

    def preprocess(self, text):
        text = text.lower()
        for old, new in self.str_replacements:
            text = text.replace(old, new)
        text = text.strip()

        # remove RTF-originated characters
        text = re.sub(r"\{\*?\\[^{}]+}|[{}]|\\\n?[A-Za-z]+\n?(?:-?\d+)?[ ]?", " ", text)
        # TODO: not all RTF-originated characters are removed

        text = re.sub("\s{2,}", " ", text)  # remove double whitespaces
        text = re.sub("ï¿½", "", text)      # remove this corrupted character

        # change '\dx\d' with '\d x \d'
        new_text = ""
        s = re.search("\dx\d", text)
        while s is not None:   # TODO: generalize
            k = s.regs[0][0]+1
            new_text += text[:k] + " x "
            text = text[k+1:]
            s = re.search("\dx\d", text)
        new_text += text

        # TODO: handle numeric strings?
        return new_text.strip()

    def preprocess_batch(self, texts):
        prep = self.preprocess
        return [prep(text) for text in texts]

    @staticmethod
    def get_default():
        if Preprocessor.instance is None:
            Preprocessor.instance = Preprocessor([("cm.", "cm "), ("mm.", "mm "), ("\n", " "), ("\t", " ")])
        return Preprocessor.instance
