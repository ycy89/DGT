input_str=$*
for i in {1..3}
do
 echo "Running experiment for the ${i}-th time"
 /data/miniconda3/bin/python ${input_str} --n_run ${i}
done
