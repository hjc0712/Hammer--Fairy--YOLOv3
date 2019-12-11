count=0
for entry in "$1"/*
do
  if (($count < $2))
  then
    echo "data/custom/$entry" >> "train.txt"
  else
    echo "data/custom/$entry" >> "valid.txt"
  fi
  ((count = count + 1))
done