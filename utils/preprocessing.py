import re

def preprocess(text):

    text = text.lower()\
        .replace("cm.", "cm ").replace("mm.", "mm ")\
        .replace("\n", " ").replace("\t", " ").strip()

    # change '\dx\d' with '\d x \d'
    while (s := re.search("\dx\d", text)) is not None:
        k = s.regs[0][0]+1
        text = text[:k] + " x " + text[k+1:]

    # TODO: come gestire stringhe numeriche?

    return text