stat = "$(tail -1 gan4ds.out)"
compare = "--JOB DONE--"
if [ $stat == $compare ]; then
  echo "Job is done!"
else
    echo ERROR: Job is not done yet. 1>&2
    exit 1 # terminate and indicate error
fi
