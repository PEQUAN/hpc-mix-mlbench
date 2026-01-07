docker run --rm -it \
  -v $(pwd):/workspace \
  -w /workspace/examples \
  promise-dev \
  pytest