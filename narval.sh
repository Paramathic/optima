# module load python/3.10 arrow cuda/12.2
# source $SCRATCH/pytorch/bin/activate


module load apptainer arrow

rm -rf $SLURM_TMPDIR/torch-one-shot.sif;
mkdir ${SLURM_TMPDIR}/torch-one-shot.sif;
tar -xf /home/mozaffar/projects/def-mmehride/mozaffar/torch-one-shot.tar -C $SLURM_TMPDIR;
mkdir ${SLURM_TMPDIR}/torch-one-shot.sif/etc/pki;
mkdir ${SLURM_TMPDIR}/torch-one-shot.sif/etc/pki/tls;
mkdir ${SLURM_TMPDIR}/torch-one-shot.sif/etc/pki/tls/certs;
cp /etc/ssl/certs/ca-bundle.crt ${SLURM_TMPDIR}/torch-one-shot.sif/etc/pki/tls/certs/ca-bundle.crt;
singularity shell --bind $PWD:/home/mozaffar --bind $SLURM_TMPDIR:/tmp --nv ${SLURM_TMPDIR}/torch-one-shot.sif