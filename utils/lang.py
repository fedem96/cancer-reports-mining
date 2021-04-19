from nltk.corpus import wordnet as wn


def get_synonyms(word, lang="ita"):
    return set([lemma.name() for synset in wn.synsets(word, lang=lang) for lemma in synset.lemmas(lang=lang)])