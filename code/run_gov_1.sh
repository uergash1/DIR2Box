dataset=gov2
gpu=3

loss_type=hinge
use_gnn=1
llm=fine-tuned-MultipleNegativesRankingLoss

version=3
train_pair_count=50000

for learning_rate in 0.00005 0.00001
do
    for weight_decay in 0.001
    do
        for dim in 512
        do
            for num_doc in 100
            do
                for gamma in 0.0 0.1 0.5 1.0 5.0
                do
                    for delta in 0.5 1.0 5.0 10.0
                    do
                        for threshold in 0.1 0.5 0.9
                        do
                            for bias in 1
                            do
                                python main.py --learning_rate $learning_rate --dim $dim --gamma $gamma --delta $delta --gpu $gpu --dataset $dataset --threshold $threshold --bias $bias --use_gnn $use_gnn --loss_type $loss_type --version $version --num_doc $num_doc --llm $llm --save_model 1 --train_pair_count $train_pair_count --epochs 5 --weight_decay $weight_decay
                            done
                        done
                    done
                done
            done
        done
    done
done
