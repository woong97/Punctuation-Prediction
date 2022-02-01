import logging
import random
import string
import yaml
import sys
import re

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(logging.INFO)


def load_yaml():
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open('config.yaml') as f:
        cfg = yaml.load(f, Loader=loader)
    return cfg


def remove_pad(trg, src, pad_idx):
    first_pad_idx = -(src == pad_idx).sum().item()
    if first_pad_idx < 0:
        trg = trg[:first_pad_idx]
        src = src[:first_pad_idx]
    return trg, src


def stuck_alphabet_test(line):
    test = True
    for alphabet in string.ascii_lowercase:
        stuck_alphabets = alphabet * 3
        if stuck_alphabets in line:
            test = False
            break
    return test


def change_punctuation_to_token(cfg, text):
    text = text.strip()
    text = text + " "
    for key in cfg['punctuation'].keys():
        if key + " " in text:
            text = text.replace(key + " ", " " + cfg['punctuation'][key] + " ")
    return text


def random_cut_text(cut_sentence_ratio, text):
    if text[-1] == ".":
        if random.random() < cut_sentence_ratio and len(text.split()) > 4:
            rand_cut_count = random.randint(1, 3)
            words = text.split()
            words = words[:-rand_cut_count]
            text = " ".join(words)
    return text

