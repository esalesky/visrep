
2019-09-04

The robust Fairseq codebase uses image representations of source text for machine translation.
Recent updates include train-time generation of source words into images, OCR style visual word encoder,
and a visual fully convolutional model.

Main code components:
Task      - tasks/translation.py [load_dataset] - calls IndexedImageDataset.py to load source as image
Data load - data/indexed_image_dataset.py - loads source text and calls image_generator.py to make images
Model     - models/visual_foncv.py and visual_transformer.py - Builds encoder/decoder model. Calls image_encoder.py to encoder word images.


To run:  (See grid_scripts)


Input/Output Shapes

Fully convolutional
Normal FConv input   - batch, tokens                            torch.Size([16, 29])  
                     - batch                                    tensor([29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29], 
Visual FConv forward - batch, tokens, channels, height, width   torch.Size([16, 29, 3, 30, 120])
Reshape              - batch*tokens, channels, height, width    torch.Size([464, 3, 30, 120])
ImageWordEncoder out - batch*tokens, features                   torch.Size([464, 256])
Reshape              - batch, tokens, features                  torch.Size([16, 29, 256])
FConv encoder output - batch, tokens, features                  torch.Size([16, 29, 256])
                     - batch, tokens, features                  torch.Size([16, 29, 256])
                     - batch, tokens                            encoder_padding_mask None

Transformer
Normal Trans Encode input  - batch, tokens              src_tokens torch.Size([8, 84])  
                             batch                      src_lengths torch.Size([8])
Normal Trans Encode output - tokens, batch, features    encoder_out torch.Size([84, 8, 512]) 
                             batch, tokens              encoder_padding_mask None



Comments/Issues:

It looks like the robust branch updated preprocessing to add [num_feats] as a parameter 
which is written in to the .bin files (see preprocess.py and tokenizer.py).
Adding a parameter to indexed_dataset.py and indexed_image_dataset.py called 
[flatten].  Parameter is passed in translation.py. 


