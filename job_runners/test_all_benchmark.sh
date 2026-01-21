datasets="autofj wt kbwt ss"

for dataset in $datasets
do
   echo "Running dataset ${dataset}"
   sbatch tester.SBATCH $dataset
done

