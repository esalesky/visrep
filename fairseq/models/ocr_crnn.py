from fairseq.modules.features_getter import ConvFeaturesGetter

import torch.nn as nn
import math
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.lstm import LSTM


from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn

# from ocrseq.models.textutils import uxxxx_to_utf8

# from torch.nn.utils.rnn import pack_padded_sequence as rnn_pack
# from torch.nn.utils.rnn import pad_packed_sequence as rnn_unpack

import logging

LOG = logging.getLogger(__name__)


@register_model("ocr_crnn")
class OcrCnnModel(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)

        LOG.info("OcrCnnModel:init")

        self.args = args

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--backbone",
            default="",
            help="CNN backbone architecture. (default: vista)",
        )
        parser.add_argument(
            "--use-bridge", action="store_true", help="use bridge layer"
        )
        parser.add_argument(
            "--use-pos-embeddings",
            action="store_true",
            help="use positional embeddings",
        )
        parser.add_argument(
            "--decoder-lstm-layers",
            type=int,
            default=3,
            help="Number of LSTM layers in model",
        )
        parser.add_argument(
            "--decoder-lstm-units",
            type=int,
            default=640,
            help="Number of LSTM hidden units in each LSTM layer (single number, or comma seperated list)",
        )
        parser.add_argument(
            "--decoder-lstm-dropout",
            type=float,
            default=0.50,
            help="Number of LSTM layers in model",
        )
        parser.add_argument(
            "--encoder-dim", type=int, default=256, help="Encoder dimensions"
        )
        parser.add_argument("--ocr-height", type=int,
                            default=32, help="Image height")
        parser.add_argument(
            "--max-allowed-width",
            type=int,
            default=1800,
            help="Max allowed image width",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        encoder = cls.build_encoder(args, src_dict)
        decoder = cls.build_decoder(args, tgt_dict)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict):
        return OcrCrnnEncoder(args, src_dict)

    @classmethod
    def build_decoder(cls, args, tgt_dict):
        return OcrCrnnDecoder(args, tgt_dict)

    def forward(self, src_images, src_images_width, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.
        """
        encoder_out = self.encoder(src_images=src_images)
        decoder_out = self.decoder(encoder_output=encoder_out)

        return decoder_out

    def decode(self, model_output, batch_actual_timesteps, is_uxxxx=False):
        min_prob_thresh = 3 * 1 / len(self.alphabet)

        T = model_output.size()[0]
        B = model_output.size()[1]

        prev_char = ["" for _ in range(B)]
        result = ["" for _ in range(B)]

        for t in range(T):

            gpu_argmax = False
            model_output_at_t_cpu = model_output.data[t].cpu().numpy()
            argmaxs = model_output_at_t_cpu.max(1).flatten()
            argmax_idxs = model_output_at_t_cpu.argmax(1).flatten()

            for b in range(B):
                # Only look at valid model output for this batch entry
                if t >= batch_actual_timesteps[b]:
                    continue

                if argmax_idxs[b] == 0:  # CTC Blank
                    prev_char[b] = ""
                    continue

                # Heuristic
                # If model is predicting very low probability for all letters in alphabet, treat that the
                # samed as a CTC blank
                if argmaxs[b] < min_prob_thresh:
                    prev_char[b] = ""
                    continue

                # char = self.alphabet.idx_to_char[argmax_idxs[b]]
                char = self.alphabet[argmax_idxs[b]]

                if prev_char[b] == char:
                    continue

                result[b] += char
                prev_char[b] = char

                # Add a space to all but last iteration
                # only need if chars encoded as uxxxx
                if is_uxxxx:
                    if t != T - 1:
                        result[b] += " "

        # Strip off final token-stream space if needed
        for b in range(B):
            if len(result[b]) > 0 and result[b][-1] == " ":
                result[b] = result[b][:-1]

        return result  # , uxxx_result


# output dimensionality for supported architectures
OUTPUT_DIM = {
    "resnet18": (512, 4),
    "resnet34": (512, 4),
    "resnet50": (2048, 4),
    "resnet101": (2048, 4),
    "resnet152": (2048, 4),
    "densenet121": (512, 4),
    "densenet161": (2208, 4),
    "densenet169": (1664, 4),
    "densenet201": (1920, 4),
    "vista": (256, 8),
}


class PositionalEmbedding(nn.Module):
    "Implement the PE function."

    def __init__(self, args, embedding_dim, num_embeddings=128):
        super().__init__()

        self.args = args

        print("...Using PositionalEmbedding")

        # Compute the positional encodings once in log space.
        # embed_num x embed_dim
        pe = torch.zeros(num_embeddings, embedding_dim)
        position = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1)
        emb = math.log(10000.0) / embedding_dim
        emb = torch.exp(torch.arange(0, embedding_dim,
                                     2, dtype=torch.float) * -emb)
        pe[:, 0::2] = torch.sin(position * emb)
        pe[:, 1::2] = torch.cos(position * emb)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if self.args.image_verbose:
            print("PositionalEmbedding: forward,  input", x.shape)
        out = self.pe[: x.size(0)]  # seq_len x embed_dim
        out = out.unsqueeze(1)
        if self.args.image_verbose:
            print("PositionalEmbedding: forward, out", out.shape)
        out = out.expand_as(x)
        if self.args.image_verbose:
            print("PositionalEmbedding: forward, out x", out.shape)
        return out


class OcrCrnnEncoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        self.args = args

        (cnn_out_c, cnn_out_h) = OUTPUT_DIM[args.backbone]
        self.cnn = ConvFeaturesGetter(args.backbone, False)

        cnn_feat_size = cnn_out_c * cnn_out_h

        LOG.info("CNN out height %d", cnn_out_h)
        LOG.info("CNN out channels %d", cnn_out_c)
        LOG.info(
            "CNN feature size (channels %d x height %d) = %d",
            cnn_out_c,
            cnn_out_h,
            cnn_feat_size,
        )

        self.bridge_layer = nn.Sequential(
            nn.Linear(cnn_feat_size, self.args.encoder_dim), nn.ReLU(
                inplace=True)
        )

        self.embed_positions = (
            PositionalEmbedding(
                args=self.args,
                embedding_dim=self.args.encoder_dim,
                num_embeddings=self.max_positions(),
            )
            if not args.use_pos_embeddings
            else None
        )

        # Finally, let's initialize parameters
        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.08, 0.08)

    def forward(self, src_images):  # , src_widths):

        LOG.debug("ENCODER: forward input %s", src_images.shape)

        x = self.cnn(src_images)
        LOG.debug("ENCODER: forward cnn features out (b, c, h, w) %s", x.shape)

        b, c, h, w = x.size()
        x = x.permute(3, 0, 1, 2).contiguous()
        LOG.debug("ENCODER: permute (w, b, c, h) %s", x.shape)

        x = x.view(-1, c * h)
        LOG.debug("ENCODER: view (w*b, c*h) %s", x.shape)

        x = self.bridge_layer(x)
        LOG.debug("ENCODER: forward bridge out %s", x.shape)

        x = x.view(w, b, -1)
        LOG.debug("ENCODER: forward bridge view %s", x.shape)

        if self.embed_positions is not None:
            embed = self.embed_positions(x)
            LOG.debug("ENCODER: forward position out %s", embed.shape)
            x = x + embed

        return {
            "encoder_out": x,
            "encoder_cnn_shape": list(x.shape),
            "input_shape": list(src_images.shape),
        }


class OcrCrnnDecoder(FairseqDecoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        self.args = args
        self.dictionary = dictionary

        self.lstm = nn.LSTM(
            self.args.encoder_dim,
            self.args.decoder_lstm_units,
            num_layers=self.args.decoder_lstm_layers,
            dropout=self.args.decoder_lstm_dropout,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * self.args.decoder_lstm_units, len(self.dictionary))
        )

        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.08, 0.08)

    def forward(self, encoder_output):
        embeddings = encoder_output["encoder_out"]
        LOG.debug("DECODER: embeddings %s", embeddings.shape)

        lstm_output, _ = self.lstm(embeddings)
        LOG.debug("DECODER: lstm output %s", lstm_output.shape)

        w = lstm_output.size(0)
        seq_len, b, embed_dim = embeddings.size()
        lstm_output_view = lstm_output.view(-1, lstm_output.size(2))
        LOG.debug("DECODER: lstm view %s", lstm_output_view.shape)

        logits = self.classifier(lstm_output_view)
        LOG.debug("DECODER: logits %s", logits.shape)

        logits = logits.view(w, b, -1)
        LOG.debug("DECODER: logits view %s", logits.shape)

        out_meta = {
            "input_shape": encoder_output["input_shape"],
            "encoder_cnn_shape": encoder_output["encoder_cnn_shape"],
            "embeddings": lstm_output_view,
            "logits": logits,
        }

        return out_meta


@register_model_architecture("ocr_crnn", "ocr_crnn_lstm")
def base_architecture(args):
    LOG.info("using architecture ocr_crnn_lstm")
    args.dropout = getattr(args, "dropout", 0.1)
    args.backbone = getattr(args, "backbone", "vistaocr")
    args.decoder_lstm_layers = getattr(args, "decoder_lstm_layers", 3)
    args.decoder_lstm_units = getattr(args, "decoder_lstm_units", 640)
    args.decoder_lstm_dropout = getattr(args, "decoder_lstm_dropout", 0.5)
    # args.use_bridge = getattr(args, "use_bridge", True)
    args.use_pos_embeddings = getattr(args, "use_pos_embeddings", True)
