LIST=gan4ds.out
DONE="JOB DONE"
EMAIL="Email sent!"
case `grep -Fx "${DONE}" "$LIST" >/dev/null; echo $?` in
  0)
    if grep -Fxq "${EMAIL}" "$LIST"
    then
      echo "Email already sent!"
      exit 1
    else
      echo "Job is done!"
      printf '\nEmail sent!' >> "$LIST"
    fi
    ;;
  1)
    echo "Job is not done!"
    exit 1
    ;;
  *)
    echo "Unhandled error occurred!"
    exit 1
    ;;
esac
