#!/bin/bash

#SBATCH --job-name=eend
#SBATCH --partition=2080ti
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G



module add anaconda/3
module add cmake/3.12.0
module add ffmpeg/3.4.8
module add cuda/11.2
module add gcc/9.3.0

source activate /mnt/lustre/sjtu/home/bhl66/miniconda3/envs/EEND
source activate pytorch
ulimit -S -n 4096



. path.sh

exp_dir=exp_large
conf_dir=conf/large

train_dir=/mnt/lustre/sjtu/home/bhl66/EEND/simu_data_2spk/label/swb_sre_tr_ns2_beta2_100000
dev_dir=/mnt/lustre/sjtu/home/bhl66/EEND/simu_data_2spk/label/swb_sre_cv_ns2_beta2_500
# model_dir=$exp_dir/models_debug1
model_dir=$exp_dir/models_load
# model_dir=$exp_dir/models
# train_conf=$conf_dir/train.yaml
train_conf=$conf_dir/train_init.yaml

train_adapt_dir=/mnt/lustre/sjtu/home/bhl66/EEND_PyTorch/data/eval/callhome1_spk2
dev_adapt_dir=/mnt/lustre/sjtu/home/bhl66/EEND_PyTorch/data/eval/callhome2_spk2
model_adapt_dir=$exp_dir/models_adapt
adapt_conf=$conf_dir/adapt.yaml
init_model=$model_dir/avg.th

infer_conf=$conf_dir/infer.yaml
test_dir=mnt/lustre/sjtu/home/bhl66/EEND_PyTorch/data/eval/callhome2_spk2
test_model=$model_adapt_dir/avg.th
infer_out_dir=$exp_dir/infer/callhome
# test_dir=data/simu/data/swb_sre_cv_ns2_beta2_500
# test_model=$model_dir/avg.th
# infer_out_dir=$exp_dir/infer/simu

work=$infer_out_dir/.work
scoring_dir=$exp_dir/score/callhome
# scoring_dir=$exp_dir/score/simu

stage=1
# debug=0
# debug_train_conf=/mnt/lustre/sjtu/home/bhl66/EEND_PyTorch/debug/conf/train_load.yaml
# debug_train_dir=/mnt/lustre/sjtu/home/bhl66/EEND/simu_data_2spk/label/swb_sre_tr_ns2_beta2_100000
# debug_dev_dir=/mnt/lustre/sjtu/home/bhl66/EEND/simu_data_2spk/label/swb_sre_cv_ns2_beta2_500
# debug_model_dir=/mnt/lustre/sjtu/home/bhl66/EEND_PyTorch/debug/model

# # Debug
# if [ $debug==1 ]; then
#     echo "Debugging~~~~"
#     python eend/bin/train.py -c $debug_train_conf $debug_train_dir $debug_dev_dir $debug_model_dir --subsampling 4 --num-frames 1000 --batchsize 16 --gradient-accumulation-steps 4
#     exit
# fi

# Training
if [ $stage -le 1 ]; then
    echo "Start training"
    python eend/bin/train.py -c $train_conf $train_dir $dev_dir $model_dir --subsampling 4 --num-frames 1000 --batchsize 16 --gradient-accumulation-steps 4
    # python eend/bin/train.py -c $train_conf $train_dir $dev_dir $model_dir --batchsize 16 --gradient-accumulation-steps 4
fi

# Model averaging
if [ $stage -le 2 ]; then
    echo "Start model averaging"
    ifiles=`eval echo $model_dir/transformer{38..40}.th`
    python eend/bin/model_averaging.py $init_model $ifiles
fi

# Adapting
if [ $stage -le 3 ]; then
    echo "Start adapting"
    python eend/bin/train.py -c $adapt_conf $train_adapt_dir $dev_adapt_dir $model_adapt_dir --initmodel $init_model --subsampling 4 --num-frames 1000 --batchsize 16 --gradient-accumulation-steps 4
    # python eend/bin/train.py -c $adapt_conf $train_adapt_dir $dev_adapt_dir $model_adapt_dir --initmodel $init_model
fi

# Model averaging
if [ $stage -le 4 ]; then
    echo "Start model averaging"
    ifiles=`eval echo $model_adapt_dir/transformer{91..100}.th`
    python eend/bin/model_averaging.py $test_model $ifiles
fi

# Inferring
if [ $stage -le 5 ]; then
    echo "Start inferring"
    python eend/bin/infer.py -c $infer_conf $test_dir $test_model $infer_out_dir
fi

# Scoring
if [ $stage -le 6 ]; then
    echo "Start scoring"
    mkdir -p $work
    mkdir -p $scoring_dir
	find $infer_out_dir -iname "*.h5" > $work/file_list
	for med in 1 11; do
	for th in 0.3 0.4 0.5 0.6 0.7; do
	python eend/bin/make_rttm.py --median=$med --threshold=$th \
		--frame_shift=80 --subsampling=10 --sampling_rate=8000 \
		$work/file_list $scoring_dir/hyp_${th}_$med.rttm
	md-eval.pl -c 0.25 \
		-r $test_dir/rttm \
		-s $scoring_dir/hyp_${th}_$med.rttm > $scoring_dir/result_th${th}_med${med}_collar0.25 2>/dev/null || exit
	done
	done
fi
