#!/bin/bash
q_LIST=("1" "2" "5")
for sig in {1..10}; do 
	#for nt_id in {1..4}; do 
		for set_id in {1..8}; do
			for q in "${q_LIST[@]}"; do 
				for seed in {1..100}; do
Python ../utils/simu.py $sig 4 $set_id $q 20 7 $seed 
done
done
done
done
#done