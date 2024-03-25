
BATCH_SIZE=8192

for DIM1 in 768 1024 1280
do
  for DIM2 in $((DIM1 * 3)) $((DIM1 * 4))
  do
    python memory_analysis.py \
      --dim1 $DIM1 \
      --dim2 $DIM2 \
      --batch_size $BATCH_SIZE \
      --dense

    python memory_analysis.py \
      --dim1 $DIM1 \
      --dim2 $DIM2 \
      --batch_size $BATCH_SIZE

    echo "************************************************************"
  done
done