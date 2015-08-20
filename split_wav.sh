#!/bin/sh

files=`find wav/`

for file in $files; do
    file_name=`echo $file | cut -d "/" -f 2`

    if [ -z "$file_name" ]; then
    continue
    fi

    name=`echo $file_name | cut -d "." -f 1`
    mkdir voice/"$name"

    sox "$file" voice/"$name"/.wav trim 0 1 : newfile : restart
done
