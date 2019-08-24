
2019-08-23

Example Translation FConv
================================
python /expscratch/detter/src/fairseq-ocr/train.py \
/expscratch/detter/mt/iwslt14.tokenized.de-en/bin \
--source-lang=de \
--target-lang=en \
--user-dir=/expscratch/detter/src/fairseq-ocr \
--task=translation \
--arch=visual_fconv_iwslt_de_en \
--lr=0.25 \
--clip-norm=0.1 \
--dropout=0.1 \
--max-tokens=1024 \
--criterion=label_smoothed_cross_entropy \
--label-smoothing=0.1 \
--save-dir=/expscratch/detter/mt/latest/ckpt_fconv_visual_bin \
--num-workers=16 \
--image-type=word \
--image-font-path=/expscratch/detter/fonts/mt.txt \
--image-font-size=16 \
--image-embed-dim=256 \
--image-channels=3 \
--image-height=30 \
--image-width=120 \
--image-stride=1 \
--image-pad=1 \
--image-kernel=3 \
--image-maxpool-height=0.5 \
--image-maxpool-width=0.7 \
--image-verbose \
--image-samples-path=/expscratch/detter/mt/latest/samples/fconv/word


Normal FConv input   - batch, tokens                            torch.Size([16, 29])  
                     - batch                                    tensor([29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29], 
Visual FConv forward - batch, tokens, channels, height, width   torch.Size([16, 29, 3, 30, 120])
Reshape              - batch*tokens, channels, height, width    torch.Size([464, 3, 30, 120])
ImageWordEncoder out - batch*tokens, features                   torch.Size([464, 256])
Reshape              - batch, tokens, features                  torch.Size([16, 29, 256])
FConv encoder output - batch, tokens, features                  torch.Size([16, 29, 256])
                     - batch, tokens, features                  torch.Size([16, 29, 256])
                     - batch, tokens                            encoder_padding_mask None
                     
                     
Example Translation Transformer 
================================
python /expscratch/detter/src/fairseq-ocr/train.py \
/expscratch/detter/mt/iwslt14.tokenized.de-en/bin \
--source-lang=de \
--target-lang=en \
--user-dir=/expscratch/detter/src/fairseq-ocr \
--task=translation \
--arch=visual_transformer_iwslt_de_en \
--share-decoder-input-output-embed \
--optimizer=adam \
--adam-betas='(0.9, 0.98)' \
--clip-norm=0.0 \
--lr=5e-4 \
--lr-scheduler=inverse_sqrt \
--warmup-updates=4000 \
--dropout=0.3 \
--weight-decay=0.0001 \
--criterion=label_smoothed_cross_entropy \
--label-smoothing=0.1 \
--max-tokens=1024 \
--save-dir=/expscratch/detter/mt/latest/ckpt_trans_visual \
--num-workers=16 \
--image-type=word \
--image-font-path=/expscratch/detter/fonts/mt.txt \
--image-font-size=16 \
--image-embed-dim=512 \
--image-channels=3 \
--image-height=30 \
--image-width=100 \
--image-stride=1 \
--image-pad=1 \
--image-kernel=3 \
--image-maxpool-height=0.5 \
--image-maxpool-width=0.7 \
--image-verbose \
--image-samples-path=/expscratch/detter/mt/latest/samples/fconv/word

Normal Trans Encode input  - batch, tokens              src_tokens torch.Size([8, 84])  
                             batch                      src_lengths torch.Size([8])
Normal Trans Encode output - tokens, batch, features    encoder_out torch.Size([84, 8, 512]) 
                             batch, tokens              encoder_padding_mask None
                 

                 