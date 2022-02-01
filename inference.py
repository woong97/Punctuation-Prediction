from train import ModelLoader
from utils.utils import *

import torch.nn.functional as F
import torch

from tqdm import tqdm
import argparse
import json
import os


def load_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    return model


def prepare_inference(cfg, checkpoint_path, text_path, batch_size, label_existed):
    loader = ModelLoader(cfg, device=torch.device('cpu'))
    model = loader.model
    model = load_model(model, checkpoint_path=checkpoint_path)
    model.eval()
    dataloader = loader.get_dataloader(
        text_path=text_path,
        batch_size=batch_size,
        label_existed=label_existed
    )
    return model, dataloader, loader


def forward_step(model, input_ids):
    with torch.no_grad():
        logits = model(input_ids)
    lprob = F.log_softmax(logits, dim=-1).squeeze(0)
    predictions = torch.argmax(lprob, dim=-1)
    confidences = torch.exp(lprob)
    return predictions, confidences


def generate_sentence(tokenizer, num_labels, input_ids, prediction):
    punctuation_ids = [tokenizer.convert_tokens_to_ids(punctuation)
                       for punctuation in cfg['punctuation'].values()]

    input_ids = input_ids[:prediction.size(0)]
    result_ids = []
    for input_id, label_id in zip(input_ids, prediction):
        result_ids.append(input_id.item())
        if label_id.item() in range(1, num_labels):
            result_ids.append(punctuation_ids[label_id.item() - 1])

    return tokenizer.decode(result_ids)


def test_labeled_data(cfg, args):
    assert args.labeled_path and os.path.exists(args.labeled_path),\
        "To test accuracy, prepare text files with punctuation label " \
        "Run create_data.py to generate chosen punctuation marks"

    model, dataloader, loader = prepare_inference(
                                    cfg=cfg,
                                    checkpoint_path=args.checkpoint_path,
                                    text_path=args.labeled_path,
                                    batch_size=args.batch_size,
                                    label_existed=True
                                )

    f = open("output/incorrect_predicted.txt", "w")
    sentences = 0
    correct_sentences = 0
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids']
        predictions, confidences = forward_step(model, input_ids=input_ids)

        labels = batch['labels']
        for ith, (prediction, label) in enumerate(zip(predictions, labels)):
            prediction, label = remove_pad(
                                    trg=prediction,
                                    src=label,
                                    pad_idx=loader.label_pad_idx
                                )

            for idx, label_idx in enumerate(prediction):
                if label_idx in range(1, loader.num_labels):
                    confidence = confidences[ith, idx, label_idx]
                    if confidence.item() < args.confidence_cutoff:
                        prediction[idx] = 0
            if torch.equal(prediction, label):
                correct_sentences += 1
            else:
                label_sentence = batch['sentences'][ith]
                incorrect_predicted_sentence = generate_sentence(
                                                    tokenizer=loader.tokenizer,
                                                    num_labels=loader.num_labels,
                                                    input_ids=input_ids[ith],
                                                    prediction=prediction
                                                )
                f.write(f"trg: {label_sentence}\n")
                f.write(f"pred: {incorrect_predicted_sentence}\n\n")

        sentences += labels.size(0)
    f.close()
    print(f"==== Cutoff : {args.confidence_cutoff} Accuracy: ", 100 * (correct_sentences / sentences))


def predict_unlabeled_data(cfg, args):
    assert args.unlabeled_path and os.path.exists(args.unlabeled_path)

    model, dataloader, loader = prepare_inference(
                                    cfg=cfg,
                                    checkpoint_path=args.checkpoint_path,
                                    text_path=args.unlabeled_path,
                                    batch_size=args.batch_size,
                                    label_existed=False
                                )

    output_result = []
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids']
        predictions, confidences = forward_step(model, input_ids=input_ids)

        for ith, (prediction, input_ids_) in enumerate(zip(predictions, input_ids)):
            prediction, input_ids_ = remove_pad(trg=prediction, src=input_ids_, pad_idx=loader.input_pad_idx)
            confidence_record = []
            for idx, label_idx in enumerate(prediction):
                if label_idx in range(1, loader.num_labels):
                    confidence = confidences[ith, idx, label_idx]
                    if confidence.item() < args.confidence_cutoff:
                        prediction[idx] = 0
                        continue
                    confidence_record.append(confidence.item())

            src_sentence = batch['sentences'][ith]
            predicted_sentence = generate_sentence(
                                    tokenizer=loader.tokenizer,
                                    num_labels=loader.num_labels,
                                    input_ids=input_ids_,
                                    prediction=prediction
                                )
            update_item = {
                'src': src_sentence,
                'predicted': predicted_sentence,
                'punctuations_confidence': confidence_record
            }
            output_result.append(update_item)
    output_path = args.unlabeled_path[:-4] + "-predict.json"

    with open(output_path, "w") as f:
        f.write(json.dumps(output_result, indent=2))
    print(f"{output_path} is saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Inference model')
    parser.add_argument('--checkpoint_path', required=True, type=str)
    parser.add_argument('--type', required=True, type=str)
    parser.add_argument('--labeled_path', required=False, default=None, type=str)
    parser.add_argument('--unlabeled_path', required=False, default=None, type=str)
    parser.add_argument('--confidence_cutoff', required=False, default=0.1, type=float)
    parser.add_argument('--batch_size', required=False, default=4, type=int)
    args = parser.parse_args()

    cfg = load_yaml()

    if args.type == 'test_labeled_data':
        test_labeled_data(cfg, args)
    elif args.type == 'predict_unlabeled_data':
        predict_unlabeled_data(cfg, args)
    else:
        raise Exception('Invalid Argument - args.type should be one of [test_labeled_data, predict_unlabeled_data]')
