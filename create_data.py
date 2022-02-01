from utils.utils import *
from tqdm import tqdm
import numpy as np
import argparse
import glob
import os


def merge_input_files(input_dir, output_dir, min_words, max_words, max_chars):
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"), recursive=True)
    merged_path = os.path.join(output_dir, "merged.txt")
    if os.path.exists(merged_path):
        raise FileExistsError(f"{merged_path} is already existed. Delete this file, if you want to recreate dataset")

    f = open(merged_path, "w")

    for txt_file in txt_files:
        with open(txt_file, "r") as f_:
            for line in tqdm(f_):
                if line == '\n':
                    continue
                line = line.strip()
                words = line.split()
                word_counts = [len(word) for word in words]
                if max(word_counts) > max_chars:
                    continue
                if not stuck_alphabet_test(line):
                    continue
                if len(words) < min_words or len(words) > max_words:
                    continue

                f.write(line.strip() + "\n")
    f.close()
    print(f"{merged_path} is saved")


def create_and_split(file_path, output_dir, cut_sentence_ratio, flush_sort_count, for_inference):

    def flush(out_type, close=False):
        selected_f = buffer[out_type]['file_descriptor']
        sorted_list = sorted(buffer[out_type]['list'], key=lambda x: len(x.split()))
        for sent in sorted_list:
            selected_f.write(sent.strip() + "\n")
        if close:
            selected_f.close()

    if for_inference:
        save_path = file_path[:-4] + "-labeled.txt"
        save_f = open(save_path, "w")
    else:
        buffer = {
            'train': {'list': [], 'file_descriptor': open(os.path.join(output_dir, "train.txt"), "w")},
            'valid': {'list': [], 'file_descriptor': open(os.path.join(output_dir, "valid.txt"), "w")},
            'test': {'list': [], 'file_descriptor': open(os.path.join(output_dir, "test.txt"), "w")}
        }

    stuck_punctuation = ["..", ". .", "??", "? ?", "?.", ".?", "? .", ". ?"]
    with open(file_path, "r") as f:
        for line in tqdm(f):

            test = True
            for invalid_punc in stuck_punctuation:
                if invalid_punc in line:
                    test = False
                    break
            if not test:
                continue

            if "mr. " in line or "mrs. " in line:
                continue

            if len(re.findall(r'\d+', line)) > 2:
                continue

            if not for_inference:
                selected_type = np.random.choice(['train', 'valid', 'test'], p=list(cfg['split_ratio'].values()))
                line = random_cut_text(cut_sentence_ratio, line.strip())
                line = change_punctuation_to_token(cfg, line)

                buffer[selected_type]['list'].append(line)

                if len(buffer[selected_type]['list']) == flush_sort_count:
                    flush(out_type=selected_type)
                    buffer[selected_type]['list'] = []
            else:
                line = change_punctuation_to_token(cfg, line)
                save_f.write(line.strip() + "\n")

    if not for_inference:
        for selected_type in buffer.keys():
            flush(out_type=selected_type, close=True)

        print(f"{os.path.join(output_dir, 'train.txt')} is saved")
        print(f"{os.path.join(output_dir, 'valid.txt')} is saved")
        print(f"{os.path.join(output_dir, 'test.txt')} is saved")
    else:
        save_f.close()
        print(f"{save_path} is saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prepare Data and Vocab for training')
    parser.add_argument('--task', default=None, required=True, type=str)

    parser.add_argument('--input_dir', default=None, required=False, type=str)
    parser.add_argument('--output_dir', default=None, required=False, type=str)

    parser.add_argument('--min_words', default=4, required=False, type=int)
    parser.add_argument('--max_words', default=50, required=False, type=int)
    parser.add_argument('--max_chars', default=20, required=False, type=int)

    parser.add_argument('--cut_sentence_ratio', default=0.5, required=False, type=float)
    parser.add_argument('--flush_sort_count', default=5120, required=False, type=int)

    parser.add_argument('--path_for_inference', default=None, required=False, type=str)

    args = parser.parse_args()

    cfg = load_yaml()

    if args.task == 'merge':
        assert args.input_dir is not None
        merge_input_files(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            min_words=args.min_words,
            max_words=args.max_words,
            max_chars=args.max_chars
        )
    elif args.task == 'create_and_split':
        if args.path_for_inference:
            text_path = args.path_for_inference
        else:
            text_path = os.path.join(args.input_dir, "merged.txt")
        create_and_split(
            file_path=text_path,
            output_dir=args.output_dir,
            cut_sentence_ratio=args.cut_sentence_ratio,
            flush_sort_count=args.flush_sort_count,
            for_inference=True if args.path_for_inference else False
        )
