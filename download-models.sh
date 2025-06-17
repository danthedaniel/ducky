#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

# Function to download a file from URL to target path
# $1: repo
# $2: commit
# $3...: files
huggingface_download() {
    local repo="$1"
    local commit="$2"
    shift 2

    mkdir -p "models/$repo"

    for file in "$@"; do
        local url="https://huggingface.co/$repo/resolve/$commit/$file?download=true"
        local target_path="models/$repo/$file"
        if [ -f "$target_path" ]; then
            echo "Skipping $target_path"
            continue
        fi

        echo "Downloading $repo/$file..."

        curl -L --progress-bar -o "$target_path" "$url"

        if ! [ $? -eq 0 ]; then
            echo "Failed to download: $url" >&2
            return 1
        fi
    done
}

huggingface_download \
    'ggerganov/whisper.cpp' \
    '5359861c739e955e79d9a303bcbc70fb988958b1' \
    'ggml-base-q8_0.bin'

huggingface_download \
    'bartowski/Llama-3.2-1B-Instruct-GGUF' \
    '067b946cf014b7c697f3654f621d577a3e3afd1c' \
    'Llama-3.2-1B-Instruct-Q4_0.gguf'
