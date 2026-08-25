#!/bin/bash
#SBATCH --job-name=cleanup_shm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --partition=public
#SBATCH --nodelist=protagoras
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

echo "--- before cleanup ---"
df -h /dev/shm
find /dev/shm -maxdepth 1 -user ryonatan | wc -l

echo "--- killing orphaned torch_shm_manager processes owned by ryonatan ---"
pkill -u ryonatan -f torch_shm_manager -9
sleep 2
ps -eo user,cmd | grep torch_shm_manager | grep -v grep | wc -l

echo "--- killing any leftover dataset_builder processes owned by ryonatan ---"
pkill -u ryonatan -f training.dataset_builder -9
sleep 2
ps -eo user,cmd | grep training.dataset_builder | grep -v grep | wc -l

echo "--- removing ryonatan-owned files in /dev/shm ---"
find /dev/shm -maxdepth 1 -user ryonatan -delete
echo "removed."

echo "--- after cleanup ---"
df -h /dev/shm
find /dev/shm -maxdepth 1 -user ryonatan | wc -l
free -h
