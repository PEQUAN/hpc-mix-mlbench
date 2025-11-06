shopt -s extglob
rm -rf !(*.sh|*.tar.gz|include|README)
rm -rf .,/cache/*.txt