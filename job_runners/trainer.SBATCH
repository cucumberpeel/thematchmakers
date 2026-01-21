#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 64GB
#SBATCH --time 10:00:00
#SBATCH --output logs/train_%J.out

./sing << EOF

cd matchmaker/
python main.py ${1}

EOF
