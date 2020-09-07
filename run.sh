#! /bin/bash

build_data() {
    cmd=''${cmd_prefix}'python deeplab/datasets/build_voc2012_data.py'

    mkdir -p tfrecord
    eval $cmd
}

train_model() {
    postfix=$1

    case $MODEL in
        deeplab)
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
            --tf_initial_checkpoint="./pretrain/deeplab/model.ckpt"
            --train_batch_size=4
            --dataset="teeth"
            --train_logdir="./train_log_'${postfix}'"
            --dataset_dir="./tfrecord_'${postfix}'"'
        ;;
        hrnet)
        cmd=''${cmd_prefix}'python hrnet/tools/train.py
            --cfg config/hrnet.yaml
            DATASET.ROOT '${dataset_path}'/'${postfix}''
        ;;
    esac

    eval $cmd
}

train_model_bright() {
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
        --tf_initial_checkpoint="./pretrain/deeplab/model.ckpt"
        --train_logdir="./train_log_bright"
        --dataset_dir="./tfrecord_bright"
        --fine_tune_batch_norm=False'
    eval $cmd
}

export_model() {
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
    cmd=''${cmd_prefix}'python deeplab/eval_teeth.py '${*}''
    eval $cmd
}

eval_metric() {
    cmd=''${cmd_prefix}'python eval_metric.py '${*}''
    eval $cmd
}

#==================================
# 检查cmd标志位是否为1,为1则直接退出程序
# $1: cmd_flag
#==================================
check_cmd_flag() {
    if [[ cmd_flag -eq 1 ]]
    then
        echo "Multiple command found."
        exit 1
    fi
}

check_option_flag() {
    if [[ cmd_flag -eq 1 ]]
    then
        echo "Please use option in front of command"
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
dataset_path="/home/shlll/Dataset/teeth"

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -G|--gpu)
    check_option_flag
    GPU_DEVICE="$2"
    shift # past argument
    shift # past value
    export CUDA_DEVICE_ORDER="PCI_BUS_ID"
    export CUDA_VISIBLE_DEVICES=$GPU_DEVICE
    ;;
    -M|--model)
    check_option_flag
    MODEL="$2"
    shift
    shift
    case $MODEL in
        deeplab)
        ;;
        hrnet)
        # conda activate hrnet
        ;;
        *)
        echo "Model parameter '${MODEL}' not supported."
    esac
    ;;
    -P|--pc)
    check_option_flag
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
    check_cmd_flag
    echo "Generating tfrecord of dataset."
    cmd_flag=1
    build_data
    shift
    ;;
    train)
    check_cmd_flag
    echo "Start training..."
    if [ ! -n "$2" ]
    then
        echo "Please specify train name."
        exit 1
    fi
    cmd_flag=1
    train_model $2
    shift
    shift
    ;;
    export)
    check_cmd_flag
    echo "Start model exporting..."
    cmd_flag=1
    export_model
    shift
    ;;
    eval_teeth)
    check_cmd_flag
    echo "Start evaluating test images..."
    cmd_flag=1
    eval_teeth $2 $3
    shift
    shift
    shift
    ;;
    eval)
    check_cmd_flag
    cmd_flag=1
    eval_metric $2 $3
    shift
    shift
    shift
    ;;
    help)
    check_cmd_flag
    cmd_flag=1
    echo "run.sh [-G|--gpu 0|1 -M|--model deeplab|hrnet -P|--pc s1|pc|swf] build|train_bright|train_ori|export|eval|help|eval"
    shift
    ;;
    *)
    echo "${1} cmd not supported, \n type 'help' for more information."
    exit 1
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters
