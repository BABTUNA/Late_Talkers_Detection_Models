# activate and verify LLE conda environment
source activate LLE
conda env list



#!/bin/bash

learning_rates=(3e-5 5e-5 6e-5)
gammas=(0.99 0.50 0.001)
hidden_sizes=(512 1024)
heads=(8 16)  # Define possible values for 'heads'



# cd /home/benjaminbarrera-altuna/Desktop/LLE/multimodal/bert_mult/all

# for learning_rate in "${learning_rates[@]}"; do
#     for gamma in "${gammas[@]}"; do
#         for hidden_size in "${hidden_sizes[@]}"; do
#             for head in "${heads[@]}"; do

#                 python multi_att_copy.py \
#                     --data_path "/home/benjaminbarrera-altuna/Desktop/LLE/dataset/pos_mod/data_50_bike_pos.json" \
#                     --lr $learning_rate \
#                     --gamma $gamma \
#                     --hidden_size $hidden_size \
#                     --heads $head \
#                     --joint_nh $head \
#                     --audio_nh $head \
#                     --text_nh $head \
#                     --gpu 0 
#             done
#         done
#     done
# done


# cd /home/benjaminbarrera-altuna/Desktop/LLE/multimodal/bert_mult/audio_text

# for learning_rate in "${learning_rates[@]}"; do
#     for gamma in "${gammas[@]}"; do
#         for hidden_size in "${hidden_sizes[@]}"; do
#             for head in "${heads[@]}"; do

#                 python multi_att_copy.py \
#                     --data_path "/home/benjaminbarrera-altuna/Desktop/LLE/dataset/pos_mod/data_50_bike_pos.json" \
#                     --lr $learning_rate \
#                     --gamma $gamma \
#                     --hidden_size $hidden_size \
#                     --heads $head \
#                     --joint_nh $head \
#                     --audio_nh $head \
#                     --text_nh $head \
#                     --gpu 0 
#             done
#         done
#     done
# done

# cd /home/benjaminbarrera-altuna/Desktop/LLE/multimodal/bert_mult/audio_text

# for learning_rate in "${learning_rates[@]}"; do
#     for gamma in "${gammas[@]}"; do
#         for hidden_size in "${hidden_sizes[@]}"; do
#             for head in "${heads[@]}"; do

#                 python multi_att_copy.py \
#                     --data_path "/home/benjaminbarrera-altuna/Desktop/LLE/dataset/pos_mod/data_50_doc_pos.json" \
#                     --lr $learning_rate \
#                     --gamma $gamma \
#                     --hidden_size $hidden_size \
#                     --heads $head \
#                     --joint_nh $head \
#                     --audio_nh $head \
#                     --text_nh $head \
#                     --gpu 0 
#             done
#         done
#     done
# done




# ----------------------


# test for cross dataset with self model
cd /home/benjaminbarrera-altuna/Desktop/LLE/multimodal/bert_mult/audio_text

for learning_rate in "${learning_rates[@]}"; do
    for gamma in "${gammas[@]}"; do
        for hidden_size in "${hidden_sizes[@]}"; do
            for head in "${heads[@]}"; do

                python multi_att_copy.py \
                    --data_path "/home/benjaminbarrera-altuna/Desktop/LLE/dataset/pos_mod/data_50_retell_pos_mod.json" \
                    --data_path_name "retell" \
                    --data_path_test "/home/benjaminbarrera-altuna/Desktop/LLE/dataset/pos_mod/data_50_bike_pos.json" \
                    --data_path_name_test "bike" \
                    --lr $learning_rate \
                    --gamma $gamma \
                    --hidden_size $hidden_size \
                    --heads $head \
                    --joint_nh $head \
                    --audio_nh $head \
                    --text_nh $head \
                    --gpu 0 
            done
        done
    done
done


for learning_rate in "${learning_rates[@]}"; do
    for gamma in "${gammas[@]}"; do
        for hidden_size in "${hidden_sizes[@]}"; do
            for head in "${heads[@]}"; do

                python multi_att_copy.py \
                    --data_path "/home/benjaminbarrera-altuna/Desktop/LLE/dataset/pos_mod/data_50_retell_pos_mod.json" \
                    --data_path_name "retell" \
                    --data_path_test "/home/benjaminbarrera-altuna/Desktop/LLE/dataset/pos_mod/data_50_doc_pos.json" \
                    --data_path_name_test "doc" \
                    --lr $learning_rate \
                    --gamma $gamma \
                    --hidden_size $hidden_size \
                    --heads $head \
                    --joint_nh $head \
                    --audio_nh $head \
                    --text_nh $head \
                    --gpu 0 
            done
        done
    done
done


for learning_rate in "${learning_rates[@]}"; do
    for gamma in "${gammas[@]}"; do
        for hidden_size in "${hidden_sizes[@]}"; do
            for head in "${heads[@]}"; do

                python multi_att_copy.py \
                    --data_path "/home/benjaminbarrera-altuna/Desktop/LLE/dataset/pos_mod/data_50_bike_pos.json" \
                    --data_path_name "bike" \
                    --data_path_test "/home/benjaminbarrera-altuna/Desktop/LLE/dataset/pos_mod/data_50_retell_pos_mod.json" \
                    --data_path_name_test "retell" \
                    --lr $learning_rate \
                    --gamma $gamma \
                    --hidden_size $hidden_size \
                    --heads $head \
                    --joint_nh $head \
                    --audio_nh $head \
                    --text_nh $head \
                    --gpu 0 
            done
        done
    done
done


for learning_rate in "${learning_rates[@]}"; do
    for gamma in "${gammas[@]}"; do
        for hidden_size in "${hidden_sizes[@]}"; do
            for head in "${heads[@]}"; do

                python multi_att_copy.py \
                    --data_path "/home/benjaminbarrera-altuna/Desktop/LLE/dataset/pos_mod/data_50_bike_pos.json" \
                    --data_path_name "bike" \
                    --data_path_test "/home/benjaminbarrera-altuna/Desktop/LLE/dataset/pos_mod/data_50_doc_pos.json" \
                    --data_path_name_test "doc" \
                    --lr $learning_rate \
                    --gamma $gamma \
                    --hidden_size $hidden_size \
                    --heads $head \
                    --joint_nh $head \
                    --audio_nh $head \
                    --text_nh $head \
                    --gpu 0 
            done
        done
    done
done


for learning_rate in "${learning_rates[@]}"; do
    for gamma in "${gammas[@]}"; do
        for hidden_size in "${hidden_sizes[@]}"; do
            for head in "${heads[@]}"; do

                
                python multi_att_copy.py \
                    --data_path "/home/benjaminbarrera-altuna/Desktop/LLE/dataset/pos_mod/data_50_doc_pos.json" \
                    --data_path_name "doc" \
                    --data_path_test "/home/benjaminbarrera-altuna/Desktop/LLE/dataset/pos_mod/data_50_bike_pos.json" \
                    --data_path_name_test "bike" \
                    --lr $learning_rate \
                    --gamma $gamma \
                    --hidden_size $hidden_size \
                    --heads $head \
                    --joint_nh $head \
                    --audio_nh $head \
                    --text_nh $head \
                    --gpu 0 
            done
        done
    done
done


for learning_rate in "${learning_rates[@]}"; do
    for gamma in "${gammas[@]}"; do
        for hidden_size in "${hidden_sizes[@]}"; do
            for head in "${heads[@]}"; do

                
                python multi_att_copy.py \
                    --data_path "/home/benjaminbarrera-altuna/Desktop/LLE/dataset/pos_mod/data_50_doc_pos.json" \
                    --data_path_name "doc" \
                    --data_path_test "/home/benjaminbarrera-altuna/Desktop/LLE/dataset/pos_mod/data_50_retell_pos_mod.json" \
                    --data_path_name_test "retell" \
                    --lr $learning_rate \
                    --gamma $gamma \
                    --hidden_size $hidden_size \
                    --heads $head \
                    --joint_nh $head \
                    --audio_nh $head \
                    --text_nh $head \
                    --gpu 0 
            done
        done
    done
done

