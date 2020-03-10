stat = (tails -1 gan4ds.out)
if [[ $stat == *"--JOB DONE--"* ]]; then
  echo "Job is done!"
else
    echo ERROR: Job is not done yet. 1>&2
    exit 1 # terminate and indicate error
fi
