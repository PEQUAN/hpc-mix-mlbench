#!/bin/bash
promise --precs=chsd --nbDigits=11 --noParsing
mv debug debug1_11
promise --precs=chsd --nbDigits=12 --noParsing
mv debug debug1_12
promise --precs=chsd --nbDigits=13 --noParsing
mv debug debug1_13
promise --precs=chsd --nbDigits=14 --noParsing
mv debug debug2_14
promise --precs=whsd --nbDigits=11 --noParsing
mv debug debug2_11
promise --precs=whsd --nbDigits=12 --noParsing
mv debug debug2_12
promise --precs=whsd --nbDigits=13 --noParsing
mv debug debug2_13
promise --precs=whsd --nbDigits=14 --noParsing
mv debug debug2_14
promise --precs=cbsd --nbDigits=11 --noParsing
mv debug debug3_11
promise --precs=cbsd --nbDigits=12 --noParsing
mv debug debug3_12
promise --precs=cbsd --nbDigits=13 --noParsing
mv debug debug3_13
promise --precs=cbsd --nbDigits=14 --noParsing
mv debug debug3_14
promise --precs=wbsd --nbDigits=14 --noParsing
mv debug debug4_11
promise --precs=wbsd --nbDigits=11 --noParsing
mv debug debug4_12
promise --precs=wbsd --nbDigits=12 --noParsing
mv debug debug4_13
promise --precs=wbsd --nbDigits=13 --noParsing
mv debug debug4_14

