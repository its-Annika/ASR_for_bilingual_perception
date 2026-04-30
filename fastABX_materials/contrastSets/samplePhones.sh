#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <vowel1> <vowel2> <inputfile> <n>"
    exit 1
fi

V1="$1"
V2="$2"
INFILE="$3"
N="$4"

HEADER=$(head -1 "$INFILE")

tail -n +2 "$INFILE" | grep -P "^\S+\s+\S+\s+\S+\s+${V1}\s+" > v1_lines.tmp
tail -n +2 "$INFILE" | grep -P "^\S+\s+\S+\s+\S+\s+${V2}\s+" > v2_lines.tmp

echo "Found $(wc -l < v1_lines.tmp) lines for '$V1'"
echo "Found $(wc -l < v2_lines.tmp) lines for '$V2'"

{ shuf -n "$N" v1_lines.tmp; shuf -n "$N" v2_lines.tmp; } \
    | shuf \
    | cat <(echo "$HEADER") - > sampled.txt

rm -f v1_lines.tmp v2_lines.tmp

echo "Done! Sampled $N of each. Output written to sampled.txt"