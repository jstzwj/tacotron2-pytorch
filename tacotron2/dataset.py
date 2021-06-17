import torch
import glob
import os
import numpy as np
from scipy.io import wavfile
import librosa

class ThchsDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = path
        self.wav_files = list(glob.glob(os.path.join(self.path, '*.wav')))
        self.txt_files = [file + '.trn' for file in self.wav_files]

        self.audio_and_text = list(zip(self.txt_files, self.wav_files))

        # vocab
        self.vocab_path = "./vocab/bert-base-chinese-vocab.txt"
        self.vocab = []
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.vocab.append(line[:-1])
        self.word_to_id = {}
        for i, w in enumerate(self.vocab):
            self.word_to_id[w] = i

        self.max_wav_value=32768.0
        self.sampling_rate=16000 # 22050
        self.filter_length=1024
        self.hop_length=256
        self.win_length=1024
        self.n_mel_channels=80
        self.mel_fmin=0.0
        self.mel_fmax=8000.0

    def _get_mel(self, filename):
        sampling_rate, data = wavfile.read(filename)
        audio, sampling_rate = data.astype(np.float32), sampling_rate
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        # audio_norm = np.expand_dims(audio_norm, axis=0)
        melspec = librosa.feature.melspectrogram(
            y=audio_norm,
            sr=sampling_rate,
            n_fft=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mel_channels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax)
        melspec = torch.FloatTensor(melspec)
        # melspec = torch.squeeze(melspec, 0)

        return melspec

    def _get_text(self, text_path):
        text = ''
        with open(text_path, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.replace(' ', '')
                text = text.strip()
                break
        ids = []
        for c in text:
            if c in self.word_to_id:
                ids.append(self.word_to_id[c])
        text_norm = torch.IntTensor(ids)
        return text_norm

    def __getitem__(self, index):
        audio_path = self.audio_and_text[index][1]
        text_path = self.audio_and_text[index][0]

        text = self._get_text(text_path)
        mel = self._get_mel(audio_path)
        return text, mel

    def __len__(self):
        return len(self.audio_and_text)


