FILE=gan4ds.out
if [[ -f "$FILE" ]]; then
  stat="$(tail -1 gan4ds.out)"
  compare="--JOB DONE--"
  if [ $stat == $compare ]; then
    echo "Job is done!"
    echo -e "Email sent!" > gan4ds.out
  else
      echo ERROR: Job is not ready yet. 1>&2
      exit 1 # terminate and indicate error
  fi
  else
      echo ERROR: Job is not started yet. 1>&2
      exit 1 # terminate and indicate error
fi
