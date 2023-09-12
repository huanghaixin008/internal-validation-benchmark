export PYTHONPATH=${PYTHONPATH}:/home/haixin/code/end2end/ipex/vit-benchmark
export PYTHONPATH=${PYTHONPATH}:/home/haixin/code/end2end/ipex/huggingface-benchmark

export DNNL_GRAPH_VERBOSE=1
export ONEDNN_VERBOSE=1
export DNNL_GRAPH_DUMP=2
export _DNNL_DISABLE_DNNL_BACKEND=1
export _DNNL_GC_GENERIC_CONV_BLOCK=1
export _DNNL_GC_GENERIC_PARTITIONING=1
export _DNNL_GC_GENERIC_MHA=1
export _DNNL_FORCE_MAX_PARTITION_POLICY=0

sh /home/haixin/code/end2end/ipex/huggingface-benchmark/run_auto_cpu_jit.sh all int8 multi_instance ipex

# sh /home/haixin/code/end2end/ipex/vit-benchmark/run_auto_cpu.sh all int8_ipex multi_instance ipex
