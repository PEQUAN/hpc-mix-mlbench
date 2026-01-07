docker run \
  --platform=linux/amd64 \
  --rm -it \
  -v $(pwd):/workspace \
  -w /workspace/examples/SP7 \
  promise-dev \
  promise --precs=sd --nbDigits=2