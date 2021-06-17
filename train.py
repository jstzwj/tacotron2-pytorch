import argparse
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from tacotron2.hparams import create_hparams
from tacotron2.dataset import ThchsDataset
from tacotron2.model import Tacotron2
from tacotron2.loss_function import Tacotron2Loss

class LitTacotron2(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.model = Tacotron2(hparams)
        self.loss = Tacotron2Loss()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = text_padded.long()
        input_lengths = input_lengths.long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = mel_padded.float()
        gate_padded = gate_padded.float()
        output_lengths = output_lengths.long()

        x = (text_padded, input_lengths, mel_padded, max_len, output_lengths)
        y = (mel_padded, gate_padded)
        xout = self.forward(x)
        loss = self.loss(xout, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus, hparams):
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    dataset = ThchsDataset('data/data_thchs30/data')
    train_loader = DataLoader(dataset, batch_size=hparams.batch_size, collate_fn=TextMelCollate(hparams.n_frames_per_step))
    model = LitTacotron2(hparams)

    tb_logger = pl_loggers.TensorBoardLogger(log_directory)
    trainer = pl.Trainer(default_root_dir=output_directory, logger=tb_logger, gpus=n_gpus)
    trainer.fit(model, train_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default='./outdir',
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_dir', type=str, default='./logdir',
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')

    args = parser.parse_args()
    '''
    for k, v in args.__dict__.items():
        print(k + ': ', v)
    '''
    print(args)

    train(args.output_dir, args.log_dir, args.checkpoint_path, args.warm_start, args.n_gpus, create_hparams())
