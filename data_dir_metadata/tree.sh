find /mnt/d/DESED_dataset/ -type d -print0 | while read -d '' -r dir; do
    files=("$dir"/*)
    printf "%s (%s)\n" "${dir/#.\//}" "${#files[@]}"
done | sort | tree --fromfile --noreport --charset utf-8

