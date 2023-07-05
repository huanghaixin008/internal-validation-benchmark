#!/bin/bash
set -xe
# import common funcs
source ../../common.sh

# attention-ocr
function prepare_workload {
    #
    workload_dir="${PWD}"
    rm -rf tensorflow-models
    git clone https://github.com/tensorflow/models.git tensorflow-models
    cd tensorflow-models
    git reset --hard 32671b

    patch -f -p1 < ${workload_dir}/attention_ocr.patch || true
    pip install -r ${workload_dir}/requirements.txt
    cd research/attention_ocr/python/
    if [ ! -e attention_ocr ];then
        ln -sf /home2/tensorflow-broad-product/oob_tf_models/mlp/attention_ocr/ .
    fi
}

function main {
    # prepare workload
    prepare_workload

    # set common info
    init_params $@
    fetch_cpu_info
    set_environment


    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        # pre run
        python demo_inference.py --dataset_dir datasets/testdata/fsns --checkpoint ./attention_ocr/model.ckpt-399731 \
            --num_iter 3 --num_warmup 1 --eval_batch_size 1 --precision ${precision}
        #
        for batch_size in ${batch_size_list[@]}
        do
            logs_path_clean
            generate_core
            collect_perf_logs
        done
    done
}

# run
function generate_core {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${cpu_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        printf "numactl -m $(echo ${cpu_array[i]} |awk -F ';' '{print $2}') \
                    -C $(echo ${cpu_array[i]} |awk -F ';' '{print $1}') \
            python demo_inference.py --dataset_dir ./datasets/testdata/fsns \
                --checkpoint ./attention_ocr/model.ckpt-399731 \
                --num_iter ${num_iter} --num_warmup ${num_warmup} \
                --eval_batch_size ${batch_size} \
                --precision ${precision} \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
    done
    echo -e "\n wait" >> ${excute_cmd_file}
    echo -e "\n\n\n\n Running..."
    source ${excute_cmd_file}
    echo -e "Finished.\n\n\n\n"

    # latency and throughput
    # latency=$(grep 'Throughput:' ${log_dir}/rcpi* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk -v bs=${batch_size} '
    #     BEGIN {
    #         sum = 0;
    #         i = 0;
    #     }
    #     {
    #         sum = sum + bs / $1 * 1000;
    #         i++;
    #     }
    #     END {
    #         sum = sum / i;
    #         printf("%.3f", sum);
    #     }
    # ')
    throughput=$(grep 'Throughput:' ${log_dir}/rcpi* |sed -e 's/.*Throughput//;s/[^0-9.]//g' |awk '
        BEGIN {
            sum = 0;
        }
        {
            sum = sum + $1;
        }
        END {
            printf("%.2f", sum);
        }
    ')
}

# Start
main "$@"
