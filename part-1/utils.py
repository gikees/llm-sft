import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)

_detokenizer = TreebankWordDetokenizer()


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Combined synonym replacement + typo introduction transformation.
    # Each qualifying word (alphabetic, length >= 3) independently has:
    #   - 30% chance of synonym replacement via WordNet
    #   - 20% chance of a single-character QWERTY-neighbor typo
    # Both checks are independent, so a word can receive both transformations.
    # Multi-word synset lemmas are excluded; original capitalisation is preserved.
    # This is a reasonable transformation because real users vary vocabulary and
    # occasionally make keyboard typos — both are plausible at inference time.

    qwerty_neighbors = {
        'a': ['s', 'q', 'w'], 'b': ['v', 'n', 'g'], 'c': ['x', 'v', 'd'],
        'd': ['s', 'e', 'f', 'c'], 'e': ['w', 'r', 'd'], 'f': ['d', 'r', 'g'],
        'g': ['f', 't', 'h', 'b'], 'h': ['g', 'y', 'j', 'n'], 'i': ['u', 'o', 'k'],
        'j': ['h', 'u', 'k', 'm'], 'k': ['j', 'i', 'l', 'o'], 'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k'], 'n': ['b', 'h', 'j', 'm'], 'o': ['i', 'p', 'l', 'k'],
        'p': ['o', 'l'], 'q': ['w', 'a'], 'r': ['e', 't', 'f', 'd'],
        's': ['a', 'w', 'd', 'x'], 't': ['r', 'y', 'g'], 'u': ['y', 'i', 'j', 'h'],
        'v': ['c', 'b', 'f'], 'w': ['q', 'e', 's', 'a'], 'x': ['z', 'c', 's'],
        'y': ['t', 'u', 'h', 'g'], 'z': ['x', 'a']
    }

    tokens = word_tokenize(example["text"])
    transformed_tokens = []

    for token in tokens:
        if not token.isalpha() or len(token) < 3:
            transformed_tokens.append(token)
            continue

        word_lower = token.lower()
        original_case = token[0].isupper()
        result = token

        if random.random() < 0.3:
            synsets = wordnet.synsets(word_lower)
            if synsets:
                synonyms = []
                for syn in synsets:
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace('_', ' ').lower()
                        if synonym != word_lower and len(synonym.split()) == 1:
                            synonyms.append(synonym)
                if synonyms:
                    chosen = random.choice(synonyms)
                    result = chosen.capitalize() if original_case else chosen

        if random.random() < 0.2:
            pos = random.randint(1, len(result) - 2)
            char = result[pos].lower()
            if char in qwerty_neighbors:
                typo_char = random.choice(qwerty_neighbors[char])
                result = result[:pos] + typo_char + result[pos + 1:]

        transformed_tokens.append(result)

    example["text"] = _detokenizer.detokenize(transformed_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
