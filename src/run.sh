#!/bin/bash

# executables
python=python3

# constants
embedding_shape=(7 7)

train_epochs=4500
train_batch_size=50
train_save_epochs=(0 300 600 1000)

lin_folder_name="geo-lin"
bil_folder_name="geo-bil"

geo_lin_save_path="$lin_folder_name/geo-lin-${embedding_shape[0]}x${embedding_shape[1]}"
geo_bil_save_path="$bil_folder_name/geo-bil-${embedding_shape[0]}x${embedding_shape[1]}"

if (( $# == 0 ))
then
    echo "No arguments supplied."
    exit 1
fi

# train
if (( $# == 1 ))
then
    if [ "$1" = "geo-lin" ]
    then
        $python main.py --energy linear --embedding-shape ${embedding_shape[@]} train-geo -save-path "$geo_lin_save_path" -epochs $train_epochs -batch-size $train_batch_size -save-epochs ${train_save_epochs[@]}
    elif [ "$1" = "geo-bil" ]
    then
        $python main.py --energy bilinear --embedding-shape ${embedding_shape[@]} train-geo -save-path "$geo_bil_save_path" -epochs $train_epochs -batch-size $train_batch_size -save-epochs ${train_save_epochs[@]}
    elif [ "$1" = "tens-lin" ]
    then 
        $python main.py --energy linear --embedding-shape 10 10 train-tens -save-path "tens-lin/nations-lin-10x10" -epochs 5000 -batch-size 200
    elif [ "$1" = "tens-bil" ]
    then 
        $python main.py --energy bilinear --embedding-shape 10 10 train-tens -save-path "tens-bil/nations-bil-10x10" -epochs 5000 -batch-size 200
    else 
        echo "Unknown command."
        exit 1
    fi
elif (( $# == 2 ))
then
    if [ "$1" = "plot" ]
    then
        saves_directory="./models/saves/"
        
        if [ "$2" = "linear" ]
        then 
            saves_directory="$saves_directory$lin_folder_name"
        elif [ "$2" = "bilinear" ]
        then
            saves_directory="$saves_directory$bil_folder_name"
        else 
            echo "plot: bad 2nd argument."
            exit 1
        fi
        
        saves_directoty_size=${#saves_directory}+1
        
        for entry in "$saves_directory/"*
        do
            if [[ -f "$entry" ]]
            then
                pic_name="${entry:$saves_directoty_size:-4}.png"
                $python main.py --energy "$2" --embedding-shape ${embedding_shape[@]} plot -model-path $entry -name $pic_name
                echo "> pic:$pic_name saved."
            fi
        done
    fi
else
    echo "Unknown command."
    exit 1
fi