#!bin/sh

source /export/disk2/yushijie/anaconda3/etc/profile.d/conda.sh
round=1
while [[ $round -le 4 ]]
do
	#statements
	mkdir round_$round
	cd round_$round
	mkdir close
	mkdir open
	cp ../round_$(($round-1))/close/bias_select.rst7 ./close/bias.rst7
	cp ../round_$(($round-1))/open/bias_select.rst7 ./open/bias.rst7
	cp ../*.py ./

	conda activate base 
	export CUDA_VISIBLE_DEVICES=5
	nohup python run1.py & 
	sleep 20s
	export CUDA_VISIBLE_DEVICES=6
	python run.py

	wait
	sleep 10s

	conda activate py35
	python tica.py

	conda activate base 
	python SGD.py
	
	let round++
	cd ../
done
