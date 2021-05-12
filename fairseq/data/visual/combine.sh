#!/bin/bash

# Creates images for the paper
#
# 1. Generate the images
#
#     ./image_generator.py --font-file fonts/NotoSans-Regular.ttf --text "В стенках (1) выполнены два отверстия (7), (8)." --prefix russian --window 25 --stride 10[m
#     ./image_generator.py --font-file fonts/NotoSans-Regular.ttf --text "Abre Sie müssen zuerts wzei Dnige über mcih wisse.n" --prefix german --window 30 --stride 10[m
#     ./image_generator.py --font-file fonts/NotoSans-Regular.ttf --text "Quand ils controllent leur cerveau, il penevut clntreloor leur douleur." --window 25 --stride 10 --prefix french[m
#
# 2. Combine them
#
#     ./combine.sh german
#
# 3. View
#
#     open german-{combined,cropped}.png

prefix=$1

for file in $prefix.??.png; do
  convert -bordercolor black -border 1 $file $file; 
done

montage -mode concatenate -tile x1 $prefix.??.png $prefix-combined.png

convert $prefix-combined.png -crop 640x23+0+0 $prefix-cropped.png
