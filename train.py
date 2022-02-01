from model.punc_model import PunctuationModel
from model.dataset import BatchGenerator
from utils.utils import *

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
import torch

from transformers import BertTokenizer
from tqdm import tqdm
import argparse
import pickle
import os


class ModelLoader:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        self.tokenizer = self.load_tokenizer()

        self.punctuations = list(self.cfg['punctuation'].values())
        self.num_labels = len(self.punctuations) + 1
        self.input_pad_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.label_pad_idx = self.num_labels + 1

        self.model = self.get_model()

    def load_tokenizer(self, pretrained_model_name_or_path='bert-base-uncased'):
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        vocab_size = len(tokenizer)
        tokenizer.add_tokens(list(self.cfg['punctuation'].values()))

        for item in self.cfg['punctuation'].values():
            tokenizer.vocab[item] = vocab_size
            vocab_size += 1
        return tokenizer

    def get_model(self):
        model = PunctuationModel(
                    model_cfg=self.cfg['model'],
                    vocab_size=len(self.tokenizer),
                    num_labels=self.num_labels,
                    pad_idx=self.input_pad_idx
                )
        model = model.to(self.device)
        return model

    def get_dataloader(self, text_path, batch_size, label_existed=True):
        generator = BatchGenerator(
            cfg=self.cfg,
            tokenizer=self.tokenizer,
            input_pad_idx=self.input_pad_idx,
            label_pad_idx=self.label_pad_idx,
            text_path=text_path,
            batch_size=batch_size,
            label_existed=label_existed
        )
        logger.info(f"Dataloader from {text_path} is loaded")
        return generator.generate()


class Trainer(ModelLoader):
    def __init__(self, cfg, device, text_paths, output_dir):
        super().__init__(cfg, device)
        self.text_paths = text_paths
        self.output_dir = output_dir

        self.optimizer, self.scheduler = self.load_optmizer_and_scheduler()

        self.total_steps = 0
        self.best_loss = 1000
        if cfg['net']['resume_from_checkpoint']:
            self.total_steps, self.best_loss = self.load_checkpoint()

        self.valid_count = 0
        self.test_count = 0

        self.tensorboard_record = SummaryWriter()
        self.accuracies = []
        self.punctuation_accuracies = []

    def load_optmizer_and_scheduler(self):
        optimizer = optim.Adam(
                        self.model.parameters(),
                        lr=self.cfg['net']['lr'],
                        betas=(0.9, 0.999),
                        eps=1e-08
                    )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="min",
                        factor=self.cfg['net']['scheduler']['factor'],
                        patience=self.cfg['net']['scheduler']['patience']
                    )

        return optimizer, scheduler

    def load_checkpoint(self):
        _, latest_checkpoint = self.get_latest_or_oldest_checkpoint(reverse=True)
        checkpoint = torch.load(os.path.join(self.output_dir, latest_checkpoint + ".pt"), map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        total_steps = checkpoint['total_steps']
        best_loss = checkpoint['best_loss']
        return total_steps, best_loss

    def loss(self, logits, labels):
        lprobs = F.log_softmax(logits, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        loss = F.nll_loss(
                        lprobs,
                        labels.view(-1),
                        ignore_index=self.label_pad_idx,
                        reduction=self.cfg['net']['loss_reduction']
                    )
        return loss

    def train(self):
        train_loss = 0

        self.model.train()
        for epoch in range(self.cfg['net']['epochs']):
            train_dataloader = self.get_dataloader(
                                    text_path=self.text_paths['train'],
                                    batch_size=self.cfg['net']['batch_size']
                                )
            step = 0
            for i, batch in tqdm(enumerate(train_dataloader)):
                input_ids = batch['input_ids']
                labels = batch['labels']

                self.optimizer.zero_grad()
                logits = self.model(input_ids.to(self.device))
                loss = self.loss(logits=logits, labels=labels.to(self.device))
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['net']['grad_norm'])
                self.optimizer.step()
                train_loss += loss.item()
                self.total_steps += 1
                step += 1

                if self.total_steps % self.cfg['net']['logging_steps'] == 0:
                    logger.info(f"Epoch : {epoch} - Total Steps : {self.total_steps} - Loss : {loss}")

                if self.total_steps % self.cfg['net']['valid_every'] == 0:
                    self.valid(epoch=epoch)
                    self.model.train()
            train_loss /= step
            logger.info(f"Epoch : {epoch} - Train : Loss: {round(train_loss, 3)}")
            self.tensorboard_record.add_scalar("Loss/Train", train_loss, global_step=epoch)
            self.valid(epoch=epoch)

    def valid(self, epoch):
        self.model.eval()
        valid_loss = 0
        logger.info(f"Validation Start")
        with torch.no_grad():
            valid_dataloader = self.get_dataloader(
                                    text_path=self.text_paths['valid'],
                                    batch_size=self.cfg['net']['batch_size']
                                )
            step = 0
            for i, batch in enumerate(valid_dataloader):
                input_ids = batch['input_ids']
                labels = batch['labels']
                logits = self.model(input_ids.to(self.device))
                loss = self.loss(logits=logits, labels=labels.to(self.device))

                valid_loss += loss.item()
                step += 1

                if step % self.cfg['net']['logging_steps'] == 0:
                    logger.info(f"Epoch : {epoch} - Valid Steps : {step}")

        valid_loss /= step
        self.scheduler.step(valid_loss)
        logger.info(f"Epoch : {epoch} - Total Steps : {self.total_steps} - Valid Loss: {valid_loss}")
        self.tensorboard_record.add_scalar("Loss/Valid", valid_loss, global_step=self.valid_count)
        self.valid_count += 1

        self.save_checkpoint(epoch, valid_loss)

        if self.cfg['net']['evaluate_during_training']:
            self.test()

    def test(self):
        logger.info("Test(Evaluation) Start")

        sentences = 0
        correct_sentences = 0

        punctuations = 0
        correct_punctuations = 0

        punctuation_labels = [label_idx for label_idx in range(1, self.num_labels)]
        specific_punctuations = {label_idx: 0 for label_idx in punctuation_labels}
        correct_specific_punctuations = {label_idx: 0 for label_idx in punctuation_labels}
        self.model.eval()
        with torch.no_grad():
            test_dataloader = self.get_dataloader(
                                    text_path=self.text_paths['test'],
                                    batch_size=self.cfg['net']['batch_size']
                                )
            for batch in test_dataloader:
                input_ids = batch['input_ids']
                labels = batch['labels']
                logits = self.model(input_ids.to(self.device))

                lprob = F.log_softmax(logits, dim=-1).squeeze(0)
                predictions = torch.argmax(lprob, dim=-1)
                labels = labels.to(self.device)
                for prediction, label in zip(predictions, labels):
                    prediction, label = remove_pad(trg=prediction, src=label, pad_idx=self.label_pad_idx)
                    if torch.equal(prediction, label):
                        correct_sentences += 1

                    for idx, label_ in enumerate(label):
                        if label_.item() in punctuation_labels:
                            punctuations += 1
                            specific_punctuations[label_.item()] += 1
                            if torch.equal(prediction[idx], label_):
                                correct_punctuations += 1
                                correct_specific_punctuations[label_.item()] += 1
                sentences += labels.size(0)

        sentence_accuracy = 100 * (correct_sentences / sentences)
        punctuation_accuracy = 100 * (correct_punctuations / punctuations)

        self.tensorboard_record.add_scalar(
                "Sentence Accuracy",
                sentence_accuracy, global_step=self.test_count
        )
        self.tensorboard_record.add_scalar(
                "Punctuation Accuracy",
                punctuation_accuracy, global_step=self.test_count
        )

        for label_idx in specific_punctuations.keys():
            ith_label_accuracy = 100 * (correct_specific_punctuations[label_idx] / specific_punctuations[label_idx])
            punctuation = self.punctuations[label_idx - 1]
            self.tensorboard_record.add_scalar(
                f"{punctuation} Accuracy",
                ith_label_accuracy, global_step=self.test_count
            )

        self.test_count += 1
        self.accuracies.append(sentence_accuracy)
        self.punctuation_accuracies.append(punctuation_accuracy)

        with open(os.path.join(self.output_dir, "loss_history.pkl"), "wb") as f:
            pickle.dump({'accuracy': self.accuracies,
                         'punctuation_accuracies': self.punctuation_accuracies}, f)

    def save_checkpoint(self, epoch, valid_loss):
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'epoch': epoch,
            'total_steps': self.total_steps,
            'best_loss': self.best_loss,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint-{self.total_steps}.pt")
        torch.save(checkpoint, checkpoint_path)

        if self.best_loss > valid_loss:
            self.best_loss = valid_loss
            best_checkpoint_path = os.path.join(self.output_dir, f"checkpoint-best.pt")
            torch.save(checkpoint, best_checkpoint_path)

        self.remove_old_checkpoint()

    def get_latest_or_oldest_checkpoint(self, reverse):
        matches = [re.match(r'checkpoint-\d+', path) for path in os.listdir(self.output_dir)]
        checkpoints = [match.group(0) for match in matches if match is not None]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]), reverse=reverse)
        return len(checkpoints), checkpoints[0]

    def remove_old_checkpoint(self):
        checkpoints_count, oldest_checkpoint = self.get_latest_or_oldest_checkpoint(reverse=False)
        if checkpoints_count == self.cfg['net']['max_checkpoints']:
            os.remove(os.path.join(self.output_dir, oldest_checkpoint + ".pt"))


def main(args):
    cfg = load_yaml()
    device = torch.device("cuda:0" if (torch.cuda.is_available() and cfg['net']['ngpu'] > 0) else "cpu")
    logger.info(f"Device =  {device}")

    if cfg['net']['evaluate_during_training']:
        assert args.test_path is not None

    if not cfg['net']['resume_from_checkpoint']:
        if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
            raise FileExistsError("Checkpoints already existed. Reset Output directory")

    if cfg['net']['resume_from_checkpoint']:
        if not os.path.exists(args.output_dir) or len(os.listdir(args.output_dir)) == 0:
            raise FileNotFoundError("Checkpoints don't existed")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    trainer = Trainer(
        cfg=cfg,
        device=device,
        text_paths={
            'train': args.train_path,
            'valid': args.valid_path,
            'test': args.test_path
        },
        output_dir=args.output_dir
    )
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Input path of training and validation text files')
    parser.add_argument('--train_path', default=None, required=True, type=str)
    parser.add_argument('--valid_path', default=None, required=True, type=str)
    parser.add_argument('--test_path', default=None, required=False, type=str)
    parser.add_argument('--output_dir', default=None, required=True, type=str)
    args = parser.parse_args()

    main(args)
