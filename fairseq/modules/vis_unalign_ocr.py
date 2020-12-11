import torch
import torch.nn as nn

import logging

LOG = logging.getLogger(__name__)


class UnAlignOCR(nn.Module):
    """Vista OCR """

    def __init__(self, args):
        super(UnAlignOCR, self).__init__()

        out_ratio = (0.5, args.ocr_fract_width_perc)
        LOG.info("FractionalMaxPool2d output_ratio %s", out_ratio)

        self.cnn = nn.Sequential(
            *self.ConvBNReLU(1, 64),
            nn.FractionalMaxPool2d(2, output_ratio=out_ratio),
            *self.ConvBNReLU(64, 128),
            nn.FractionalMaxPool2d(2, output_ratio=out_ratio),
            *self.ConvBNReLU(128, 256)
        )

    def forward(self, x):

        # b, t, c, h, w = x.shape  # batch x token x channel x height x width
        # x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))

        x = self.cnn(x)

        return x

    def ConvBNReLU(self, nInputMaps, nOutputMaps, stride=1):
        return [
            nn.Conv2d(nInputMaps, nOutputMaps, stride=stride, kernel_size=3, padding=1),
            nn.BatchNorm2d(nOutputMaps),
            nn.ReLU(inplace=True),
        ]


class UnAlignOcrModel(torch.nn.Module):
    def __init__(self, args, alphabet, eval_only=False):
        super().__init__()
        self.args = args

        self.eval_only = eval_only

        self.alphabet = alphabet
        self.encoder = UnAlignOcrEncoder(args)
        self.decoder = UnAlignOcrDecoder(args, alphabet)

        LOG.info("UnAlignOcrModel eval_only %s", self.eval_only)
        LOG.info(repr(self))

    def forward(self, src_tokens):  # , src_widths):
        encoder_out = self.encoder(src_tokens)  # , src_widths)
        decoder_out = self.decoder(encoder_out)

        return decoder_out  # decoder_out

    def train(self, mode=True):

        if self.eval_only:
            mode = False

        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    # def decode(self, model_output, batch_actual_timesteps, is_uxxxx=False):
    #     min_prob_thresh = 3 * 1 / len(self.alphabet)

    #     T = model_output.size()[0]
    #     B = model_output.size()[1]

    #     prev_char = ['' for _ in range(B)]
    #     result = ['' for _ in range(B)]

    #     for t in range(T):

    #         gpu_argmax = False
    #         model_output_at_t_cpu = model_output.data[t].cpu().numpy()
    #         argmaxs = model_output_at_t_cpu.max(1).flatten()
    #         argmax_idxs = model_output_at_t_cpu.argmax(1).flatten()

    #         for b in range(B):
    #             # Only look at valid model output for this batch entry
    #             if t >= batch_actual_timesteps[b]:
    #                 continue

    #             if argmax_idxs[b] == 0:  # CTC Blank
    #                 prev_char[b] = ''
    #                 continue

    #             # Heuristic
    #             # If model is predicting very low probability for all letters in alphabet, treat that the
    #             # samed as a CTC blank
    #             if argmaxs[b] < min_prob_thresh:
    #                 prev_char[b] = ''
    #                 continue

    #             # char = self.alphabet.idx_to_char[argmax_idxs[b]]
    #             char = self.alphabet[argmax_idxs[b]]

    #             if prev_char[b] == char:
    #                 continue

    #             result[b] += char
    #             prev_char[b] = char

    #             # Add a space to all but last iteration
    #             # only need if chars encoded as uxxxx
    #             if is_uxxxx:
    #                 if t != T - 1:
    #                     result[b] += ' '

    #     # Strip off final token-stream space if needed
    #     for b in range(B):
    #         if len(result[b]) > 0 and result[b][-1] == ' ':
    #             result[b] = result[b][:-1]

    #     return result  # , uxxx_result


class UnAlignOcrEncoder(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Vista output size for 32 height input
        cnn_out_c = 256
        cnn_out_h = 8

        self.cnn = unalign_ocr(args)

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
            nn.Linear(cnn_feat_size, self.args.image_embed_dim), nn.ReLU(inplace=True)
        )

        # Finally, let's initialize parameters
        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.08, 0.08)

    def forward(self, src_tokens):  # , src_widths):

        LOG.debug("ENCODER: forward input %s", src_tokens.shape)

        x = self.cnn(src_tokens)
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

        return {
            "encoder_out": x,
            "encoder_cnn_shape": list(x.shape),
            "input_shape": list(src_tokens.shape),
        }

    def ConvBNReLU(self, nInputMaps, nOutputMaps):
        return [
            nn.Conv2d(nInputMaps, nOutputMaps, kernel_size=3, padding=1),
            nn.BatchNorm2d(nOutputMaps),
            nn.ReLU(inplace=True),
        ]


class UnAlignOcrDecoder(torch.nn.Module):
    def __init__(self, args, alphabet):
        super().__init__()

        self.args = args
        self.alphabet = alphabet

        self.lstm = nn.LSTM(
            self.args.image_embed_dim,
            self.args.decoder_lstm_units,
            num_layers=self.args.decoder_lstm_layers,
            dropout=self.args.decoder_lstm_dropout,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * self.args.decoder_lstm_units, len(self.alphabet))
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


def unalign_ocr(args):
    model = UnAlignOCR(args)
    return model
