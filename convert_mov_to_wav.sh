#!/bin/sh

cd ../
files=`find row/`

for file in $files
do
    file_name=`echo $file | cut -d "/" -f 2`
    if [ -z "$file_name" ]; then
    continue
    fi
    
    name=`echo $file_name | cut -d "." -f 1`

    ffmpeg -i row/"$file_name" -map 0:1 wav/"$name".wav
    echo $file_name
done
