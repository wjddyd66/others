python data_initialization.py

Cell=("GM12878" "H1-hESC" "HeLa-S3" "HepG2" "K562")
TF=("JunD")

supervise=(0 10)

# Train
for c in "${Cell[@]}"; do
	for t in "${TF[@]}"; do
	  for s in "${supervise[@]}"; do
		  python train_DANN.py $c $t $s
		  python train_baseline.py $c $t $s
	  done
  done
done
