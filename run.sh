for i in {0..7}; do nohup python -u train.py $i >log/soft_201_temp0dot1_$i.log 2>&1 & done 
