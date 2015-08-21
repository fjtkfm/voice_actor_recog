#!/bin/sh

if [ $# -ne 2 ]; then
    echo "Enter 2 params: input_path, output_path"
    exit 1
fi

from=$1
to=$2

files=`find "$from"`
for file in $files; do
    file_name=`echo $file | cut -d "/" -f 2`

    if [ -z "$file_name" ]; then
        continue
    fi

    if [ "$file_name" = "$from" ]; then
        continue
    fi

    name=`echo $file_name | cut -d "." -f 1`
    mkdir "$to"/"$name"

    sox "$file" "$to"/"$name"/.wav trim 0 1 : newfile : restart
done
