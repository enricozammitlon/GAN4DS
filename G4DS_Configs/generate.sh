#!/bin/bash

#compile_rooter
mkdir output
for(( loop =5;loop<250;loop++))
do
#rand=$(( RANDOM % 250 ))

#echo ${rand}

cp nr_generator.mac run_${loop}.mac
echo  "/run/filename outRun_${loop}" >> run_${loop}.mac
sed -i "s/12345/${loop}/g" run_${loop}.mac

./g4ds run_${loop}.mac
g4rooter_full outRun_${loop}.fil

mv run_${loop}.mac output/ 
mv outRun_${loop}.fil output/
mv outRun_${loop}.log output/
mv outRun_${loop}.root output/

done

#cd multi_attempt_1/
#hadd total_file.root *.root
