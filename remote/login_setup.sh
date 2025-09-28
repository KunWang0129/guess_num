# Login to della
ssh 'bw5889@della-gpu.princeton.edu'

# Initialize conda environment
module purge
module load anaconda3/2025.6
conda activate guess_num

# cd to project directory
cd /scratch/gpfs/ABDIENG/bw5889/code/learning/guess_num