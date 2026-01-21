#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 64GB
#SBATCH --time 00:15:00
#SBATCH --output logs/train_%J.out

./sing << EOF

cd matchmaker/
python main.py ${1}

EOF
