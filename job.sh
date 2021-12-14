#!bin/sh

round=1
while [[ $round -le 5 ]]
do
        mkdir state_A/round_$round
        cp state_A/round_$(($round-1))/round_$(($round-1))_restart.rst7 state_A/round_$(($round))/
        mkdir state_B/round_$round
        cp state_B/round_$(($round-1))/round_$(($round-1))_restart.rst7 state_B/round_$(($round))/

        # 10ns equilibrate PPS
        cd state_A/round_$round
        export CUDA_VISIBLE_DEVICES=0
        nohup python openmm_eq.py &

        # 10ns equilibrate R
        cd ../../state_B/round_$round
        export CUDA_VISIBLE_DEVICES=1
        python openmm_eq.py
 

        sleep 5s

        # state_A
        cd ~/state_A/round_$round
        cp ../../LOAD.py LOAD.py
        nohup python LOAD.py &


        # state_B
        cd ~/state_B/round_$round
        cp ../../LOAD.py LOAD.py
        python LOAD.py

        wait

        cd ~/state_A/round_$round
        sed "s/ID/$round/g" ../../PCC.py > PCC.py
        nohup python PCC.py &


        cd ~/state_B/round_$round
        sed "s/ID/$round/g" ../../PCC.py > PCC.py
        python PCC.py

        wait



        # one step biased sampling
        cd ~/state_B/round_$round
        export CUDA_VISIBLE_DEVICES=2,3
        nohup PATH_TO_NAMD/namd2 +p 8 +devices 0,1 round_$(($round)).conf > round_$(($round)).log &


        # forced sampling
        export CUDA_VISIBLE_DEVICES=2,3
        PATH_TO_NAMD/namd2 +p 8 +devices 2,3 round_$(($round)).conf > round_$(($round)).log

        sleep 15s

        # write forced restart file
        cd ~/MVI/pps/round_$round
        sed "s/ID/$round/g" ../../forced_struct.py > forced_struct_$(($round)).py
        python forced_struct_$(($round)).py

        cd ~/MVI/r/round_$round
        sed "s/ID/$round/g" ../../forced_struct.py > forced_struct_$(($round)).py
        python forced_struct_$(($round)).py

        wait

        let round++
        cd ~
done

