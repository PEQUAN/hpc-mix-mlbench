rm -rf debug2, debug4, debug6, debug8

promise --precs=chsd --nbDigits=2 --noParsing
mv debug debug1_2
promise --precs=chsd --nbDigits=4 --noParsing
mv debug debug1_4
promise --precs=chsd --nbDigits=6 --noParsing
mv debug debug1_6
promise --precs=chsd --nbDigits=8 --noParsing
mv debug debug1_8

promise --precs=whsd --nbDigits=2 --noParsing
mv debug debug2_2
promise --precs=whsd --nbDigits=4 --noParsing
mv debug debug2_4
promise --precs=whsd --nbDigits=6 --noParsing
mv debug debug2_6
promise --precs=whsd --nbDigits=8 --noParsing
mv debug debug2_8

promise --precs=cbsd --nbDigits=2 --noParsing
mv debug debug3_2
promise --precs=cbsd --nbDigits=4 --noParsing
mv debug debug3_4
promise --precs=cbsd --nbDigits=6 --noParsing
mv debug debug3_6
promise --precs=cbsd --nbDigits=8 --noParsing
mv debug debug3_8


promise --precs=wbsd --nbDigits=2 --noParsing
mv debug debug4_2
promise --precs=wbsd --nbDigits=4 --noParsing
mv debug debug4_4
promise --precs=wbsd --nbDigits=6 --noParsing
mv debug debug4_6
promise --precs=wbsd --nbDigits=8 --noParsing
mv debug debug4_8