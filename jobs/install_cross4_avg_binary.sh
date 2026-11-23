#!/bin/bash
# One-shot: install the CROSS4_LEG_RULE=avg-default cross4 CAD binary once the
# min-rule topopt job (which runs the binary per design with the rule unset)
# has finished. rename(2) is atomic, so a reader sees the old or the new file.
#SBATCH --job-name=install_cross4_avg
#SBATCH --partition=public --nodelist=protagoras --cpus-per-task=1 --mem=1G --time=00:05:00
#SBATCH --output=install_cross4_avg_%j.log
cd "$SLURM_SUBMIT_DIR"/data/cross4_11x11/CAD_model || exit 1
mv -f model.avgdefault_staged model && mv -f model.o.avgdefault_staged model.o && touch model && ls -la model model.o && md5sum model
