cuda=0


for seed in 0 ; do
    #
    # ERM
    #
    exp_name=civil_half_bert_erm_${seed}-dpVal-8warm
    CUDA_VISIBLE_DEVICES=${cuda} PYTHONHASHSEED=0 python train.py --config_file configs/config_train.json --validation_metric ED_PO1_AcrossZ --random_seed ${seed} --exp_name ${exp_name}
    CUDA_VISIBLE_DEVICES=${cuda} PYTHONHASHSEED=0 python eval.py --config_file configs/config_eval.json --exp_name ${exp_name}


    #
    # Text Trigger
    #
    for num_word in 5 ; do
        for lr in 0.1 ; do
            for training_ratio in 1.0 ; do
                # Soft
                for adv_weight in 0.0 0.5 5.0 20.0 100.0 ; do
                    exp_name=civil_half_${training_ratio}_bert_dp_adv_${adv_weight}weight_${seed}-${num_word}-dpVal-PostProcessing-${lr}lr-sample4eval-simplex-onehot-new
                    CUDA_VISIBLE_DEVICES=${cuda} PYTHONHASHSEED=0 python train.py --config_file configs/config_train.json --lr ${lr} --validation_metric ED_PO1_AcrossZ --random_seed ${seed} --exp_name ${exp_name}  --num_warmup_epoch 0 --adversary_loss_weight ${adv_weight} --num_trigger_word ${num_word} --only_optimize_trigger --dir_pretrain_model civil_half_bert_erm_${seed}-dpVal-8warm --sample4eval --sampling_method simplex --trigger_word_selector_init_method onehot --use_training2 --training2_ratio ${training_ratio}
                    CUDA_VISIBLE_DEVICES=${cuda} PYTHONHASHSEED=0 python eval.py --config_file configs/config_eval.json --exp_name ${exp_name} --use_sample4eval
                done

                # Hard
                for adv_weight in 0.0 0.1 10.0 30.0 100.0 ; do
                    exp_name=civil_half_${training_ratio}_bert_dp_adv_${adv_weight}weight_${seed}-${num_word}-dpVal-PostProcessing-${lr}lr-simplex-onehot-st-new
                    CUDA_VISIBLE_DEVICES=${cuda} PYTHONHASHSEED=0 python train.py --config_file configs/config_train.json --lr ${lr} --validation_metric ED_PO1_AcrossZ --random_seed ${seed} --exp_name ${exp_name}  --num_warmup_epoch 0 --adversary_loss_weight ${adv_weight} --num_trigger_word ${num_word} --only_optimize_trigger --dir_pretrain_model civil_half_bert_erm_${seed}-dpVal-8warm --sampling_method simplex --trigger_word_selector_init_method onehot --use_training2 --training2_ratio ${training_ratio} --use_straight_through
                    CUDA_VISIBLE_DEVICES=${cuda} PYTHONHASHSEED=0 python eval.py --config_file configs/config_eval.json --exp_name ${exp_name}
                done
            done
        done
    done


    #
    # In processing ADV
    #
    for adv_weight in 0.0 0.1 0.5 5.0 20.0 ; do
        exp_name=civil_half_bert_dp_advin_${adv_weight}weight_${seed}-dpVal-0warm
        CUDA_VISIBLE_DEVICES=${cuda} PYTHONHASHSEED=0 python train.py --config_file configs/config_train.json --validation_metric ED_PO1_AcrossZ --random_seed ${seed} --exp_name ${exp_name} --adversary_loss_weight ${adv_weight} --num_warmup_epoch 0
        CUDA_VISIBLE_DEVICES=${cuda} PYTHONHASHSEED=0 python eval.py --config_file configs/config_eval.json --exp_name ${exp_name}
    done


    #
    # Post processing ADV
    #
    for adv_weight in 0.0 0.1 0.5 1.0 5.0 ; do
        for training_ratio in 1.0 ; do
            exp_name=civil_half_${training_ratio}_bert_dp_advpost_${adv_weight}weight_${seed}-dpVal-0warm-tr2-new
            CUDA_VISIBLE_DEVICES=${cuda} PYTHONHASHSEED=0 python train.py --config_file configs/config_train.json --validation_metric ED_PO1_AcrossZ --random_seed ${seed} --exp_name ${exp_name} --adversary_loss_weight ${adv_weight} --use_training2 --num_warmup_epoch 0 --dir_pretrain_model civil_half_bert_erm_${seed}-dpVal-8warm --training2_ratio ${training_ratio}
            CUDA_VISIBLE_DEVICES=${cuda} PYTHONHASHSEED=0 python eval.py --config_file configs/config_eval.json --exp_name ${exp_name}
        done
    done


done