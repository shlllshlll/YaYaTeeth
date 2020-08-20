#! /bin/bash

build_data() {
    cmd_prefix=$1
    cmd=''${cmd_prefix}'python deeplab/datasets/build_voc2012_data.py'

    mkdir -p tfrecord
    eval $cmd
}

train_model() {
    cmd_prefix=$1
    postfix=$2

    cmd=''${cmd_prefix}'python deeplab/train.py
        --logtostderr
        --training_number_of_steps=420000
        --train_split="train"
        --model_variant="xception_65"
        --atrous_rates=6
        --atrous_rates=12
        --atrous_rates=18
        --output_stride=16
        --decoder_output_stride=4
        --train_crop_size="513,513"
        --tf_initial_checkpoint="./pretrain/model.ckpt"
        --train_batch_size=4
        --dataset="teeth"
        --train_logdir="./train_log_'${postfix}'"
        --dataset_dir="./tfrecord_'${postfix}'"'
    eval $cmd
}

train_model_bright() {
    cmd_prefix=$1
    cmd=''${cmd_prefix}'python deeplab/train.py
        --logtostderr
        --training_number_of_steps=420000
        --train_split="train"
        --model_variant="xception_65"
        --atrous_rates=6
        --atrous_rates=12
        --atrous_rates=18
        --output_stride=16
        --decoder_output_stride=4
        --train_crop_size="513,513"
        --train_batch_size=2
        --dataset="teeth"
        --tf_initial_checkpoint="./pretrain/model.ckpt"
        --train_logdir="./train_log_bright"
        --dataset_dir="./tfrecord_bright"
        --fine_tune_batch_norm=False'
    eval $cmd
}


train_model_ori() {
    cmd_prefix=$1
    cmd=''${cmd_prefix}'python deeplab/train.py
        --logtostderr
        --training_number_of_steps=620000
        --train_split="train"
        --model_variant="xception_65"
        --atrous_rates=6
        --atrous_rates=12
        --atrous_rates=18
        --output_stride=16
        --decoder_output_stride=4
        --train_crop_size="513,513"
        --train_batch_size=2
        --dataset="teeth21"
        --tf_initial_checkpoint="./pretrain/model.ckpt"
        --train_logdir="./train_log_ori"
        --dataset_dir="./tfrecord_ori"
        --fine_tune_batch_norm=False'
    eval $cmd
}

train_model_2166() {
    cmd_prefix=$1
    cmd=''${cmd_prefix}'python deeplab/train.py
        --logtostderr
        --training_number_of_steps=420000
        --train_split="train"
        --model_variant="xception_65"
        --atrous_rates=6
        --atrous_rates=12
        --atrous_rates=18
        --output_stride=16
        --decoder_output_stride=4
        --train_crop_size="513,513"
        --train_batch_size=8
        --dataset="teeth4"
        --train_logdir="./train_log_2166"
        --dataset_dir="./tfrecord_2166"'
    eval $cmd
}

export_model() {
    cmd_prefix=$1
    cmd=''${cmd_prefix}'python deeplab/export_model.py
        --atrous_rates=6
        --atrous_rates=12
        --atrous_rates=18
        --output_stride=16
        --decoder_output_stride=4
        --model_variant="xception_65"
        --num_classes=3
        --checkpoint_path="./train_log_crop/model.ckpt-243782"
        --export_path="./trained_model/243782.pb"'
    eval $cmd
}

eval_teeth() {
    cmd_prefix=$1
    shift
    cmd=''${cmd_prefix}'python deeplab/eval_teeth.py '${*}''
    eval $cmd
}

eval_metric() {
    cmd_prefix=$1
    shift
    cmd=''${cmd_prefix}'python deeplab/eval_metric.py '${*}''
    eval $cmd
}

#==================================
# 检查cmd标志位是否为1,为1则直接退出程序
# $1: cmd_flag
#==================================
check_cmd_flag() {
    cmd_flag=$1
    if [[ cmd_flag -eq 1 ]]
    then
        echo "Multiple command found."
        exit 1
    fi
}

if [[ $# -eq 0 ]]
then
    echo "No parameter specified."
    exit 1
fi

cmd_flag=0
cmd_prefix="pipenv run "

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -G|--gpu)
    if [[ cmd_flag -eq 1 ]]
    then
        echo "Please use option in front of command."
        exit 1
    fi
    GPU_DEVICE="$2"
    shift # past argument
    shift # past value
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export CUDA_VISIBLE_DEVICES=$GPU_DEVICE
    ;;
    -M|--mode)
    if [[ cmd_flag -eq 1 ]]
    then
        echo "Please use option in front of command."
        exit 1
    fi
    MODE="$2"
    shift
    shift
    case $MODE in
        s1)
        cmd_prefix=""
        export PYTHONPATH="${PYTHONPATH}:/home/shlll/Projects/teeth:/home/shlll/Projects/teeth/slim"
        ;;
        pc)
        export CUDA_HOME="/usr/local/cuda-10.0"
        export PYTHONPATH="${PYTHONPATH}:/home/shlll/Projects/teeth:/home/shlll/Projects/teeth/slim"
        ;;
        swf)
        cmd_prefix=""
        export PYTHONPATH="${PYTHONPATH}:/data/shl/teeth:/data/shl/teeth/slim"
        ;;
        *)
        echo "Path parameter '${MODE}' not supported."
        exit 1
        ;;
    esac
    ;;
    build)
    check_cmd_flag "$cmd_flag"
    echo "Generating tfrecord of dataset."
    cmd_flag=1
    build_data "$cmd_prefix"
    shift
    ;;
    train)
    check_cmd_flag "$cmd_flag"
    echo "Start training..."
    if [ ! -n "$2" ]
    then
        echo "Please specify train name."
        exit 1
    fi
    cmd_flag=1
    train_model "$cmd_prefix" $2
    shift
    shift
    ;;
    train_bright)
    check_cmd_flag "$cmd_flag"
    echo "Start training..."
    cmd_flag=1
    train_model_bright "$cmd_prefix"
    shift
    ;;
    train_ori)
    check_cmd_flag "$cmd_flag"
    echo "Start training..."
    cmd_flag=1
    train_model_ori "$cmd_prefix"
    shift
    ;;
    train_2166)
    check_cmd_flag "$cmd_flag"
    echo "Start training..."
    cmd_flag=1
    train_model_2166 "$cmd_prefix"
    shift
    ;;
    export)
    check_cmd_flag "$cmd_flag"
    echo "Start model exporting..."
    cmd_flag=1
    export_model "$cmd_prefix"
    shift
    ;;
    eval_teeth)
    check_cmd_flag "$cmd_flag"
    echo "Start evaluating test images..."
    cmd_flag=1
    eval_teeth "$cmd_prefix" $2 $3
    shift
    shift
    shift
    ;;
    eval)
    check_cmd_flag "$cmd_flag"
    cmd_flag=1
    eval_metric "$cmd_prefix" $2 $3
    shift
    shift
    shift
    ;;
    help)
    check_cmd_flag "$cmd_flag"
    cmd_flag=1
    echo "run.sh [-G|--gpu 0|1 -P|--path s1|swf] build|train_bright|train_ori|export|eval|help|eval"
    shift
    ;;
    *)
    echo "${1} cmd not supported, \n type 'help' for more information."
    exit 1
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters
