# activate and verify LLE conda environment
source activate LLE
conda env list

learning_rates=(3e-4 4e-5 5e-5)
gammas=(0.01 0.25 0.50 0.95)
hidden_sizes=(256 512 1024)

for learning_rate in "${learning_rates[@]}"; do
    for gamma in "${gammas[@]}"; do
        for hidden_size in "${hidden_sizes[@]}"; do

                python multi_att_copy.py \
                    --data_path "/home/benjaminbarrera-altuna/Desktop/LLE/dataset/data_50.json" \
                    --lr $learning_rate \
                    --gamma $gamma \
                    --hidden_size $hidden_size 

        done
    done
done
