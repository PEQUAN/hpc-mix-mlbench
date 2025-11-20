#!/bin/bash

# Define the content of the rename_debug.sh script
read -r -d '' SCRIPT_CONTENT << 'EOF'
#!/bin/bash
rm -rf debug2 debug4 debug6 debug8
promise --precs=chsd --nbDigits=1
mv debug debug1_1
promise --precs=chsd --nbDigits=2
mv debug debug1_2
promise --precs=chsd --nbDigits=3
mv debug debug1_3
promise --precs=chsd --nbDigits=4
mv debug debug1_4
promise --precs=chsd --nbDigits=5
mv debug debug1_5
promise --precs=chsd --nbDigits=6
mv debug debug1_6
promise --precs=chsd --nbDigits=7
mv debug debug1_7
promise --precs=chsd --nbDigits=8
mv debug debug1_8
promise --precs=chsd --nbDigits=9
mv debug debug1_9
promise --precs=chsd --nbDigits=10
mv debug debug1_10
promise --precs=whsd --nbDigits=1
mv debug debug2_1
promise --precs=whsd --nbDigits=2
mv debug debug2_2
promise --precs=whsd --nbDigits=3
mv debug debug2_3
promise --precs=whsd --nbDigits=4
mv debug debug2_4
promise --precs=whsd --nbDigits=5
mv debug debug2_5
promise --precs=whsd --nbDigits=6
mv debug debug2_6
promise --precs=whsd --nbDigits=7
mv debug debug2_7
promise --precs=whsd --nbDigits=8
mv debug debug2_8
promise --precs=whsd --nbDigits=9
mv debug debug2_9
promise --precs=whsd --nbDigits=10
mv debug debug2_10
promise --precs=cbsd --nbDigits=1
mv debug debug3_1
promise --precs=cbsd --nbDigits=2
mv debug debug3_2
promise --precs=cbsd --nbDigits=3
mv debug debug3_3
promise --precs=cbsd --nbDigits=4
mv debug debug3_4
promise --precs=cbsd --nbDigits=5
mv debug debug3_5
promise --precs=cbsd --nbDigits=6
mv debug debug3_6
promise --precs=cbsd --nbDigits=7
mv debug debug3_7
promise --precs=cbsd --nbDigits=8
mv debug debug3_8
promise --precs=cbsd --nbDigits=9
mv debug debug3_9
promise --precs=cbsd --nbDigits=10
mv debug debug3_10
promise --precs=wbsd --nbDigits=1
mv debug debug4_1
promise --precs=wbsd --nbDigits=2
mv debug debug4_2
promise --precs=wbsd --nbDigits=3
mv debug debug4_3
promise --precs=wbsd --nbDigits=4
mv debug debug4_4
promise --precs=wbsd --nbDigits=5
mv debug debug4_5
promise --precs=wbsd --nbDigits=6
mv debug debug4_6
promise --precs=wbsd --nbDigits=7
mv debug debug4_7
promise --precs=wbsd --nbDigits=8
mv debug debug4_8
promise --precs=wbsd --nbDigits=9
mv debug debug4_9
promise --precs=wbsd --nbDigits=10
mv debug debug4_10
EOF

# Iterate through each directory in the current location
for dir in */; do
  # Check if it's a directory
  if [ -d "$dir" ]; then
    # Create the rename_debug.sh file in the directory
    echo "$SCRIPT_CONTENT" > "${dir}rename_debug.sh"
    # Make the script executable
    chmod +x "${dir}rename_debug.sh"
    echo "Created rename_debug.sh in $dir"
  fi
done