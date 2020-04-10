from torch.nn import LSTM, Linear, BatchNorm1d, Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class NoOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class ISTFT(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        center=False
    ):
        super(ISTFT, self).__init__()
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center
    def forward(self, Estimates, mixaudiolen):
        """
        Parameters:
        Estimates, float32 of the shape [nb_frames, nb_samples, nb_channels, nb_bins, 2]
        mixaudiolen: int, with the length of the original mixture for adjusting right padding x.shape[-1]
        Output: time-domain floats with shape [nb_samples, nb_channels, nb_timesteps]
        """
        nb_frames, nb_samples, nb_channels, nb_bins, _ = Estimates.size()
        device = Estimates.device
        Estimates = Estimates.permute(1, 2, 3, 0, 4)  # undo reshape
        Estimates = Estimates.reshape(nb_samples * nb_channels, self.n_fft // 2 + 1, -1, 2)  # merge batch and channels in multichannel stft
        y = torchaudio.functional.istft(Estimates, n_fft=self.n_fft,
                                        window=torch.hann_window(window_length=self.n_fft).to(device),
                                        pad_mode='reflect', center=True, onesided=True, length=mixaudiolen)
        y = y.contiguous().view(nb_samples, nb_channels, -1)
        return y

class STFT(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        center=False
    ):
        super(STFT, self).__init__()
        #make it compatible with old models
        self.window = nn.Parameter(
            torch.hann_window(n_fft),
            requires_grad=False
        )
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        """

        nb_samples, nb_channels, nb_timesteps = x.size()
        device = x.device
        # merge nb_samples and nb_channels for multichannel stft
        x = x.reshape(nb_samples*nb_channels, -1)

        # compute stft with parameters as close as possible scipy settings
        stft_f = torch.stft(
            x,
            n_fft=self.n_fft,
            window=torch.hann_window(window_length=self.n_fft).to(device),
            pad_mode='reflect',
            center=True,
            onesided=True
        )

        # reshape back to channel dimension
        stft_f = stft_f.contiguous().view(
            nb_samples, nb_channels, self.n_fft // 2 + 1, -1, 2
        )
        return stft_f


class OpenUnmixSingle(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        input_is_spectrogram=False,
        hidden_size=1024,
        nb_channels=2,
        sample_rate=44100,
        nb_layers=3,
        input_mean=None,
        input_scale=None,
        max_bin=None,
        unidirectional=False,
        power=1,
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Output: Power/Mag Spectrogram
                (nb_frames, nb_samples, nb_channels, nb_bins)
        """

        super(OpenUnmixSingle, self).__init__()
        self.nb_output_bins = n_fft // 2 + 1
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.hidden_size = hidden_size

        self.stft = STFT(n_fft=n_fft, n_hop=n_hop)
        self.istft = ISTFT(n_fft=n_fft, n_hop=n_hop)
        self.register_buffer('sample_rate', torch.tensor(sample_rate))

        self.fc1 = Linear(
            self.nb_bins*nb_channels, hidden_size,
            bias=False
        )

        self.bn1 = BatchNorm1d(hidden_size)

        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4,
        )

        self.fc2 = Linear(
            in_features=hidden_size*2,
            out_features=hidden_size,
            bias=False
        )

        self.bn2 = BatchNorm1d(hidden_size)

        self.fc3_1 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins*nb_channels,
            bias=False
        )
        self.fc3_2 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins*nb_channels,
            bias=False
        )
        self.fc3_3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins*nb_channels,
            bias=False
        )

        self.fc3_4 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins*nb_channels,
            bias=False
        )



        self.bn3_1 = BatchNorm1d(self.nb_output_bins*nb_channels)
        self.bn3_2 = BatchNorm1d(self.nb_output_bins*nb_channels)
        self.bn3_3 = BatchNorm1d(self.nb_output_bins*nb_channels)
        self.bn3_4 = BatchNorm1d(self.nb_output_bins*nb_channels)


        if input_mean is not None:
            input_mean = torch.from_numpy(
                -input_mean[:self.nb_bins]
            ).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(
                1.0/input_scale[:self.nb_bins]
            ).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(
            torch.ones(self.nb_output_bins).float()
        )
        self.output_mean = Parameter(
            torch.ones(self.nb_output_bins).float()
        )


    def forward(self, x):
        # check for waveform or spectrogram
        # transform to spectrogram if (nb_samples, nb_channels, nb_timesteps)
        # and reduce feature dimensions, therefore we reshape
        x = self.transform(x)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        # crop
        x = x[..., :self.nb_bins]

        # shift and scale input to mean=0 std=1 (across all bins)
        x += self.input_mean
        x *= self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels*self.nb_bins))
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)

        # lstm skip connection
        x = torch.cat([x, lstm_out[0]], -1)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = F.relu(x)

        # second dense stage + layer norm
        x_1 = self.fc3_1(x)
        x_2 = self.fc3_2(x)
        x_3 = self.fc3_3(x)
        x_4 = self.fc3_4(x)

        x_1 = self.bn3_1(x_1)
        x_2 = self.bn3_2(x_2)
        x_3 = self.bn3_3(x_3)
        x_4 = self.bn3_4(x_4)

        # reshape back to original dim
        x_1 = x_1.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)
        x_2 = x_2.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)
        x_3 = x_3.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)
        x_4 = x_4.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)


        # apply output scaling
        x_1 *= self.output_scale
        x_1 += self.output_mean
        x_2 *= self.output_scale
        x_2 += self.output_mean
        x_3 *= self.output_scale
        x_3 += self.output_mean
        x_4 *= self.output_scale
        x_4 += self.output_mean
        # since our output is non-negative, we can apply RELU

        x_1 = F.relu(x_1)
        x_2 = F.relu(x_2)
        x_3 = F.relu(x_3)
        x_4 = F.relu(x_4)

        return [x_1, x_2, x_3, x_4]