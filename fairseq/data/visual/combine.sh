#!/bin/bash

# Creates images for the paper
#
# 1. Generate images
#
#     ./image_generator.py --font-file fonts/NotoSans-Regular.ttf --text "В стенках (1) выполнены два отверстия (7), (8)." --prefix russian --window 35 --stride 10
#     ./image_generator.py --font-file fonts/NotoSans-Regular.ttf --text "Abre Sie müssen zuerts wzei Dnige über mcih wisse.n" --prefix german --window 35 --stride 10
#     ./image_generator.py --font-file fonts/NotoSans-Regular.ttf --text "Quand ils controllent leur cerveau, il penevut clntreloor leur douleur." --window 25 --stride 10 --prefix french
#     ./image_generator.py --font-file fonts/NotoSans-Regular.ttf --text ".ةعبسلا يناوخأ رغصأ انأ و ,ةيدنك انأ" --window 25 --stride 10 --prefix arabic
#
# 2. Combine them
#
#     ./combine.sh german
#
# 3. View
#
#     open german-{combined,cropped}.png

prefix=$1

for file in $prefix.*.png; do
  # echo $file
  convert -bordercolor black -border 2 $file $file; 
done

montage -mode concatenate -geometry +5 -tile x1 $prefix.??.png $prefix-combined.png

# Cropping images to paper page-width: 
#convert $prefix-combined.png -crop 640x27+1+1 $prefix-cropped.png
convert $prefix-combined.png -crop 2000x54+0+0 $prefix-cropped.png
