import re
import string


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

        # remove all invalid characters: accented letters, strange characters originated by a bad encoding...
        text = re.sub(r"[^0-9a-z\s" + string.punctuation + "]", "", text)

        # change '\dx\d' with '\d x \d'
        new_text = ""
        s = re.search("\dx\d", text)
        while s is not None:   # TODO: generalize
            k = s.regs[0][0]+1
            new_text += text[:k] + " x "
            text = text[k+1:]
            s = re.search("\dx\d", text)
        text = new_text + text

        for c in string.punctuation:  # spaces around punctuation
            text = re.sub(re.escape(c), " " + c + " ", text)

        text = re.sub("\s{2,}", " ", text)  # remove double whitespaces

        # grading: convert to short form
        text = re.sub("grad(.?|ing)\s(v|5|cinque)", " g5 ", text)
        text = re.sub("grad(.?|ing)\s(iv|4|quattro)", " g4 ", text)
        text = re.sub("grad(.?|ing)\s(iii|3|tre)", " g3", text)
        text = re.sub("grad(.?|ing)\s(ii|2|due)", " g2 ", text)
        text = re.sub("grad(.?|ing)\s(i|1|uno)", " g1 ", text)
        for i in range(1, 6):
            text = text.replace(" g {} ".format(i), " g{} ".format(i))

        # stadio_N
        text = re.sub("(n\d)", r" \1 ", text)

        # stadio_T
        text = re.sub("(t\d)", r" \1 ", text)

        # tipo_T
        # text = text.replace("yp", " yp ") # with this replace is worse # TODO: investigate

        # cerb: make the same token
        text = re.sub("c?\s?-?\s?erb\s?-?\s?b?\s?-?\s?2?", " cerb ", text)
        text = re.sub("\sher\s?-?\s?2?", " cerb ", text)

        # ki67: make the same token
        text = re.sub("ki\s?-?\s?67", " ki67 ", text)

        # mib1: make the same token
        text = re.sub("\smib?\s?-?\s?1", " mib1 ", text)

        text = re.sub("\s{2,}", " ", text)  # remove (again) double whitespaces

        # TODO: handle numeric strings?
        return text.strip()

    def preprocess_batch(self, texts):
        prep = self.preprocess
        return [prep(text) for text in texts]

    @staticmethod
    def get_default():
        if Preprocessor.instance is None:
            Preprocessor.instance = Preprocessor([("cm.", "cm "), ("mm.", "mm "), ("\n", " "), ("\t", " ")])
        return Preprocessor.instance
