#!/bin/bash

# Size threshold: files larger than this will be ignored (size in kilobytes)
size_threshold=10000

# Find files larger than the threshold and append them to .gitignore
find . -type f -size +${size_threshold}k -exec echo {} \; >> .gitignore

# Optional: sort and remove duplicates from .gitignore
sort -u .gitignore -o .gitignore