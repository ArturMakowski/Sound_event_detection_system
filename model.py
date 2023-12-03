import warnings
import yaml

import torch.nn as nn
import torch
import torchsummary
import torchinfo


# ! CNN
class CNN(nn.Module):
    def __init__(
        self,
        n_in_channel,
        activation="Relu",
        conv_dropout=0,
        kernel_size=[3, 3, 3],
        padding=[1, 1, 1],
        stride=[1, 1, 1],
        nb_filters=[64, 64, 64],
        pooling=[(1, 4), (1, 4), (1, 4)],
        normalization="batch",
        **kwargs,
    ):
        """
            Initialization of CNN network

        Args:
            n_in_channel: int, number of input channel
            activation: str, activation function
            conv_dropout: float, dropout
            kernel_size: kernel size
            padding: padding
            stride: list, stride
            nb_filters: number of filters
            pooling: list of tuples, time and frequency pooling
            normalization: choose between "batch" for BatchNormalization and "layer" for LayerNormalization.
        """
        super(CNN, self).__init__()

        self.nb_filters = nb_filters
        cnn = nn.Sequential()

        def conv(i, normalization="batch", dropout=None, activ="relu"):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]
            cnn.add_module(
                "conv{0}".format(i),
                nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]),
            )
            if normalization == "batch":
                cnn.add_module(
                    "batchnorm{0}".format(i),
                    nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99),
                )
            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, nOut))

            if activ.lower() == "leakyrelu":
                cnn.add_module("relu{0}".format(i), nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("relu{0}".format(i), nn.ReLU())

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        for i in range(len(nb_filters)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            cnn.add_module(
                "pooling{0}".format(i), nn.AvgPool2d(pooling[i])
            )  # bs x tframe x mels

        self.cnn = cnn

    def forward(self, x):
        """
        Forward step of the CNN module

        Args:
            x (Tensor): input batch of size (batch_size, n_channels, n_frames, n_freq)

        Returns:
            Tensor: batch embedded
        """
        # conv features
        x = self.cnn(x)
        return x


# ! RNN
class BidirectionalGRU(nn.Module):
    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1):
        """
            Initialization of BidirectionalGRU instance
        Args:
            n_in: int, number of input
            n_hidden: int, number of hidden layers
            dropout: flat, dropout
            num_layers: int, number of layers
        """

        super(BidirectionalGRU, self).__init__()
        self.rnn = nn.GRU(
            n_in,
            n_hidden,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
            num_layers=num_layers,
        )

    def forward(self, input_feat):
        recurrent, _ = self.rnn(input_feat)
        return recurrent


# ! CRNN
class CRNN(nn.Module):
    def __init__(
        self,
        n_in_channel=1,
        nclass=10,
        activation="Relu",
        dropout=0.5,
        rnn_type="BGRU",
        n_RNN_cell=128,
        n_layers_RNN=2,
        dropout_recurrent=0,
        **kwargs,
    ):
        """
            Initialization of CRNN model

        Args:
            n_in_channel: int, number of input channel
            n_class: int, number of classes
            activation: str, activation function
            dropout: float, dropout
            train_cnn: bool, training cnn layers
            rnn_type: str, rnn type
            n_RNN_cell: int, RNN nodes
            n_layer_RNN: int, number of RNN layers
            dropout_recurrent: float, recurrent layers dropout
            cnn_integration: bool, integration of cnn
            freeze_bn:
            **kwargs: keywords arguments for CNN.
        """
        super(CRNN, self).__init__()

        self.n_in_channel = n_in_channel

        n_in_cnn = n_in_channel

        self.cnn = CNN(
            n_in_channel=n_in_cnn,
            activation=activation,
            conv_dropout=int(dropout),
            **kwargs,
        )

        # n_in_channel,
        # activation="Relu",
        # conv_dropout=0,
        # kernel_size=[3, 3, 3],
        # padding=[1, 1, 1],
        # stride=[1, 1, 1],
        # nb_filters=[64, 64, 64],
        # pooling=[(1, 4), (1, 4), (1, 4)],
        # normalization="batch"

        if rnn_type == "BGRU":
            nb_in = self.cnn.nb_filters[-1]
            self.rnn = BidirectionalGRU(
                n_in=nb_in,
                n_hidden=n_RNN_cell,
                dropout=dropout_recurrent,
                num_layers=n_layers_RNN,
            )
        else:
            NotImplementedError("Only BGRU supported for CRNN for now")

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell * 2, nclass)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pad_mask=None, embeddings=None):
        # print(f'x.shape input = {x.shape}')
        x = x.transpose(1, 2).unsqueeze(1)
        # print(f'x.transpose(1, 2).unsqueeze(1).shape = {x.shape}')
        # input size : (batch_size, n_channels, n_frames, n_freq)

        # conv features
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        # print(f'x.shape post cnn = {x.shape}')

        if freq != 1:
            warnings.warn(
                f"Output shape is: {(bs, frames, chan * freq)}, from {freq} staying freq"
            )
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]

        # print(f'x.shape pre rnn = {x.shape}')
        x = self.rnn(x)
        x = self.dropout(x)
        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        return strong.transpose(1, 2)


if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        configs = yaml.safe_load(f)

    model = CRNN(**configs["net"])
    torchinfo.summary(
        model,
        (64, 128, 618),
        verbose=1,
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
    )
