from utils import *
import numpy as np

BATCH_SIZE = 1

input_1 = np.array([[[ 0,  1],
        [ 1,  1],
        [10, 10],
        [ 0,  1],
        [ 1,  1],
        [10, 10],
        [ 0,  1],
        [ 1,  1],
        [10, 10],
        [ 0,  1],
        [ 1,  1],
        [10, 10],
        [ 0,  1],
        [ 1,  1],
        [10, 10],
        [ 0,  1],
        [ 1,  1],
        [10, 10],
        [ 0,  1],
        [ 1,  1],
        [10, 10],
        [ 0,  1],
        [ 1,  1],
        [10, 10],
        [ 0,  1],
        [ 1,  1]]])

models = [

    # aipg-vdcnn
    {
        'model_name': 'aipg-vdcnn',
        'input_generator': lambda set_seed=True: {'input_x': generate_data([1024, 70], set_seed=set_seed), 'phase': False},
        'output': ['fc-3/predictions']
    },

    # arttrack-coco-multi
    {
        'model_name': 'arttrack-coco-multi',
        'input_generator': lambda set_seed=True: {'sub': generate_data([688,688,3], set_seed=set_seed)},
        'output': ['pose/part_pred/block4/BiasAdd','pose/locref_pred/block4/BiasAdd','pose/pairwise_pred/block4/BiasAdd']
    },

    # arttrack-mpii-single
    {
        'model_name': 'arttrack-mpii-single',
        'input_generator': lambda set_seed=True: {'sub': generate_data([688, 688, 3], set_seed=set_seed)},
        'output': ['pose/part_pred/block4/BiasAdd','pose/locref_pred/block4/BiasAdd']
    },

    # ava-person-vehicle-detection-stage1-2.0.0
    {
        'model_name': 'ava-person-vehicle-detection-stage1-2.0.0',
        'input_generator': lambda set_seed=True: {'data': generate_data([544, 992, 3], set_seed=set_seed), 'im_info': generate_data([4], set_seed=set_seed)},
        'output': ['cls_score','bbox_pred','head_score','head_bbox_pred','torso_score','torso_reg_pred','wp_score','wp_reg_pred']
    },

    # COVID-Net
    {
        'model_name': 'COVID-Net',
        'input_generator': lambda set_seed=True: {'input_1': generate_data([480, 480, 3], set_seed=set_seed)},
        'output': ['norm_dense_1/Softmax']
    },

    # Time_series_LSTM
    {
         'model_name': 'Time_series_LSTM',
         'input_generator': lambda set_seed=True: {'lstm_input': generate_data([6, 1], set_seed=set_seed)},
         'output': ['activation']
    },

    # cpm-pose
    {
        'model_name': 'cpm-pose',
        'input_generator': lambda set_seed=True: {'CPM/Placeholder_1': generate_data([368, 368, 3], set_seed=set_seed), 'CPM/Placeholder_2': generate_data([368, 368, 1], set_seed=set_seed)},
        'output': ['CPM/PoseNet/Mconv7_stage6/Conv2D']
    },

    # DLRM
    {
        'model_name': 'DLRM',
        'input_generator': lambda set_seed=True: {'dense_features': np.array([[0., 1.3862944, 1.3862944, 1.609438, 8.13798, 5.480639, 0.6931472,
                                               3.218876, 5.187386, 0., 0.6931472, 0., 3.6635616]], dtype=np.float32),
                  'sparse_features': np.array([[3, 93, 319, 272, 0, 5, 7898, 1, 0, 2, 3306, 310, 2528, 7,
                                                293, 293, 1, 218, 1, 2, 302, 0, 1, 120, 1, 2]], dtype=np.int32)
                  },
        'output': ['Identity',]
    },

    # deepvariant_wgs
    {
        'model_name': 'deepvariant_wgs',
        'input_generator': lambda set_seed=True: {'InceptionV3/InceptionV3/Conv2d_1a_3x3/ReadVariableOp': generate_data([100, 221, 6], set_seed=set_seed)},
        'output': ['InceptionV3/Predictions/Reshape_1']
    },

    # dense_vnet_abdominal_ct
    {
        'model_name': 'dense_vnet_abdominal_ct',
        'input_generator': lambda set_seed=True: {'worker_0/validation/Squeeze': generate_data([144, 144, 144, 1], set_seed=set_seed)},
        'output': ['worker_0/DenseVNet/trilinear_resize_3/transpose_1']
    },

    # east_resnet_v1_50
    {
        'model_name': 'east_resnet_v1_50',
        'input_generator': lambda set_seed=True: {'input_images': generate_data([1024, 1920, 3], set_seed=set_seed)},
        'output': ['feature_fusion/Conv_7/Sigmoid','feature_fusion/concat_3']
    },

    # facenet-20180408-102900
    {
        'model_name': 'facenet-20180408-102900',
        'input_generator': lambda set_seed=True: {'input': generate_data([160, 160, 3], set_seed=set_seed), 'phase_train': False},
        'output': ['embeddings']
    },

    # faster-rcnn-resnet101-coco-sparse-60-0001
    {
        'model_name': 'faster-rcnn-resnet101-coco-sparse-60-0001',
        'input_generator': lambda set_seed=True: {'image': generate_data([800, 1280, 3], set_seed=set_seed)},
        'output': ['openvino_outputs/cls_score','openvino_outputs/bbox_pred']
    },

    # GNMT
    # {
    #     'model_name': 'GNMT',
    #     'input_generator': lambda set_seed=True: {'IteratorGetNext:1{i32}[1],IteratorGetNext:0{i32}[1 50],dynamic_seq2seq/hash_table_Lookup_1:0[1]->[2],dynamic_seq2seq/hash_table_Lookup:0[1]->[1]': generate_data([688, 688, 3], set_seed=set_seed)},
    #     'output': ['dynamic_seq2seq/decoder/decoder/GatherTree']
    # },

    # handwritten-score-recognition-0003
    {
        'model_name': 'handwritten-score-recognition-0003',
        'input_generator': lambda set_seed=True: {'Placeholder': generate_data([32, 64, 1], set_seed=set_seed)},
        'output': ['shadow/LSTMLayers/transpose_time_major']
    },

    # license-plate-recognition-barrier-0007
    {
        'model_name': 'license-plate-recognition-barrier-0007',
        'input_generator': lambda set_seed=True: {'input': generate_data([24, 94, 3], set_seed=set_seed)},
        'output': ['d_predictions']
    },

    # lm_1b
    # {
    #     'model_name': 'lm_1b',
    #     'input_generator': lambda set_seed=True: {'char_embedding/EmbeddingLookupUnique/Unique:0,char_embedding/EmbeddingLookupUnique/Unique:1,Variable/read,Variable_1/read': generate_data([1], set_seed=set_seed),[50],[1,9216],[1,9216])},
    #     'output': ['softmax_out','lstm/lstm_0/concat_2','lstm/lstm_1/concat_2']
    # },

    # ncf
    {
        'model_name': 'NCF',
        'input_generator': lambda set_seed=True: {'0': False, '1': False},
        'output': ['add_2']
    },

    # optical_character_recognition-text_recognition-tf
    {
        'model_name': 'optical_character_recognition-text_recognition-tf',
        'input_generator': lambda set_seed=True: {'input': generate_data([32, 100, 3], set_seed=set_seed)},
        'output': ['shadow/LSTMLayers/transpose_time_major']
    },

    # pose-ae-multiperson
    {
        'model_name': 'pose-ae-multiperson',
        # 'input_generator': lambda set_seed=True: {'Placeholder_1': generate_data([512, 512, 3], set_seed=set_seed)},
        'input_generator': lambda set_seed=True: {'my_model/strided_slice': generate_data([512, 512, 3], set_seed=set_seed)},
        'output': ['my_model/out_3/add']
    },

    # pose-ae-refinement
    {
        'model_name': 'pose-ae-refinement',
        # 'input_generator': lambda set_seed=True: {'Placeholder_1': generate_data([512, 512, 3], set_seed=set_seed)},
        'input_generator': lambda set_seed=True: {'my_model/strided_slice': generate_data([512, 512, 3], set_seed=set_seed)},
        'output': ['my_model/out_3/add']
    },

    # PRNet
    {
        'model_name': 'PRNet',
        'input_generator': lambda set_seed=True: {'Placeholder': generate_data([256, 256, 3], set_seed=set_seed)},
        'output': ['resfcn256/Conv2d_transpose_16/Sigmoid']
    },

    # text-recognition-0012
    {
        'model_name': 'text-recognition-0012',
        'input_generator': lambda set_seed=True: {'Placeholder': generate_data([32, 120, 1], set_seed=set_seed)},
        'output': ['shadow/LSTMLayers/transpose_time_major']
    },

    # vggvox
    {
        'model_name': 'vggvox',
        'input_generator': lambda set_seed=True: {'input': generate_data([512, 1000, 1], set_seed=set_seed)},
        'output': ['fc8/BiasAdd']
    },

    # efficientnet
    {
        'model_name': 'efficientnet-b0',
        'input_generator': lambda set_seed=True: {'sub': generate_data([224, 224, 3], set_seed=set_seed)},
        'output': ['logits']
    },

    {
        'model_name': 'efficientnet-b0_auto_aug',
        'input_generator': lambda set_seed=True: {'sub': generate_data([224, 224, 3], set_seed=set_seed)},
        'output': ['logits']
    },

    {
        'model_name': 'efficientnet-b5',
        'input_generator': lambda set_seed=True: {'sub': generate_data([224, 224, 3], set_seed=set_seed)},
        'output': ['logits']
    },

    {
        'model_name': 'efficientnet-b7_auto_aug',
        'input_generator': lambda set_seed=True: {'sub': generate_data([224, 224, 3], set_seed=set_seed)},
        'output': ['logits']
    },

    # yolo_v3_mlp
    {
        'model_name': 'yolo_v3_mlp',
        'input_generator': lambda set_seed=True: {'input/input_data': generate_data([416, 416, 3], set_seed=set_seed)},
        'output': ['pred_sbbox/concat_2', 'pred_mbbox/concat_2', 'pred_lbbox/concat_2']
    },

    # Hierarchical_LSTM
    {
        'model_name': 'Hierarchical_LSTM',
        'input_generator': lambda set_seed=True: {'batch_in': generate_data([1024, 27], set_seed=set_seed), 'batch_out': generate_data([1024, 27], set_seed=set_seed)},
        'output': ['map_1/while/output_module_vars/prediction']
    },

    # resnet_v2_200
    {
        'model_name': 'resnet_v2_200',
        'input_generator': lambda set_seed=True: {'fifo_queue_Dequeue': generate_data([224, 224, 3], set_seed=set_seed)},
        'output': ['resnet_v2_200/SpatialSqueeze']
    },
    # TextCNN
    {
        'model_name': 'TextCNN',
        'input_generator': lambda set_seed=True: {'is_training_flag': False, 'input_x': generate_data([200,], set_seed=set_seed, input_dtype="int32"), 'dropout_keep_prob': generate_data([384], set_seed=set_seed)},
        'output': ['Sigmoid']
    },
    # TextRNN
    {
        'model_name': 'TextRNN',
        'input_generator': lambda set_seed=True: {'input_x': generate_data([100,], set_seed=set_seed, input_dtype="int32"), 'input_y': generate_data([1,], set_seed=set_seed, input_dtype="int32", newaxis = False), 'dropout_keep_prob': generate_data([100], set_seed=set_seed)},
        'output': ['Accuracy']
    },
    # TextRCNN, need bs=512
    {
        'model_name': 'TextRCNN',
        'input_generator': lambda set_seed=True: {'input_x': generate_data([100,], set_seed=set_seed, input_dtype="int32"), 'input_y': generate_data([1,], set_seed=set_seed, input_dtype="int32"), 'dropout_keep_prob': generate_data([300], set_seed=set_seed)},
        'output': ['Accuracy']
    },
    # CapsuleNet
    {
        'model_name': 'CapsuleNet',
        'input_generator': lambda set_seed=True: {'input/x': generate_data([128, 28, 28, 1], set_seed=set_seed, newaxis=False), 'input/label': generate_data([128, 10], set_seed=set_seed, newaxis=False)},
        'output': ['Decoder/fully_connected_2/Sigmoid']
    },
    # CharCNN
    {
        'model_name': 'CharCNN',
        'input_generator': lambda set_seed=True: {'Model/input': generate_data([35, 21], set_seed=set_seed,input_dtype="int32")},
        'output': ['Model/LSTM/WordEmbedding/add']
    },
    # CenterNet
    {
        'model_name': 'CenterNet',
        'input_generator': lambda set_seed=True: {'inputs': generate_data([224, 224, 3], set_seed=set_seed)},
        'output': ['detector/hm/Sigmoid']
    },
    # VNet
    {
        'model_name': 'VNet',
        'input_generator': lambda set_seed=True: {'input': generate_data([190,190,20,6], set_seed=set_seed)},
        'output': ['vnet/output_layer/add']
    },
    # DIEN
    {
        'model_name': 'DIEN_Deep-Interest-Evolution-Network',
        'input_generator': lambda set_seed=True: {'Inputs/mid_his_batch_ph': generate_data([300], set_seed=set_seed,input_dtype="int32"),'Inputs/cat_his_batch_ph': generate_data([300], set_seed=set_seed,input_dtype="int32"),
        "Inputs/uid_batch_ph":np.array([1],dtype=np.int32),"Inputs/cat_batch_ph":np.array([1],dtype=np.int32),'Inputs/mask': generate_data([300], set_seed=set_seed),
        "Inputs/seq_len_ph":np.array([1],dtype=np.int32),"Inputs/mid_batch_ph":np.array([1],dtype=np.int32)},
        'output': ['add_9']
    },
    # CRNN
    {
        'model_name': 'CRNN',
        'input_generator': lambda set_seed=True: {'input': generate_data([32, 100, 3], set_seed=set_seed)},
        'output': ['CTCGreedyDecoder']
    },
    # yolo-v3-tiny
    {
        'model_name': 'yolo-v3-tiny',
        'input_generator': lambda set_seed=True: {'image_input': generate_data([416, 416, 3], set_seed=set_seed)},
        'output':['conv2d_9/BiasAdd', 'conv2d_12/BiasAdd']
    },
    # wide_deep
    {
        'model_name': 'wide_deep',
        'input_generator': lambda set_seed=True: {'new_categorical_placeholder': input_1, 'new_numeric_placeholder':generate_data([13], set_seed=set_seed)},
        'output':['import/head/predictions/probabilities']
    },
    # show and talk
    {
        'model_name': 'show_and_tell',
        'input_generator': lambda set_seed=True: {'batch_and_pad' :generate_data([299,299,3], set_seed=set_seed)},
        'output':['lstm/basic_lstm_cell/Sigmoid_2']
    },
    # deepspeech
    { 
        'model_name': 'deepspeech',
        'input_generator': lambda set_seed=True: {'previous_state_c/read': generate_data([2048], set_seed=set_seed), 'previous_state_h/read': generate_data([2048], set_seed=set_seed), 'input_node': generate_data([16, 19, 26], set_seed=set_seed),"input_lengths":np.array([16],dtype=np.int32)},
        'output': ['raw_logits','lstm_fused_cell/GatherNd','lstm_fused_cell/GatherNd_1']
    },
    # AttRec
    {
        'model_name': 'AttRec',
        'input_generator': lambda set_seed=True: {'keep_prob': np.array([.5], dtype=np.float32), 'Placeholder': generate_data([5], set_seed=set_seed, input_dtype="int32"), 'Placeholder_1': np.array([3], dtype=np.int32),"Placeholder_3": np.array([1],dtype=np.int32)},
        'output': ['TopKV2']
    },
    # MiniGo
    {
        'model_name': 'MiniGo',
        'input_generator': lambda set_seed=True: {'pos_tensor': np.random.choice(a=[False, True], size=(1, 13, 19, 19), p=[0.5, 1-0.5]) },
        'output': ['policy_output', 'value_output']
    },
    # MANN
    {
        'model_name': 'MANN',
        'input_generator': lambda set_seed=True: {'nn_Y': generate_data([363,], set_seed=set_seed),
                  'nn_X': generate_data([480,], set_seed=set_seed),
                  'nn_keep_prob': np.array([.7], dtype='float32')},
        'output': ['Mean']
    },
    # context_rcnn_resnet101_snapshot_serenget
    {
        'model_name': 'context_rcnn_resnet101_snapshot_serenget',
        'input_generator': lambda set_seed=True: {'image_tensor': generate_data([300, 300, 3], set_seed=set_seed, input_dtype="uint8"),
                  'context_features': generate_data([ 2000, 2057], set_seed=set_seed),
                  'valid_context_size': np.array([1], dtype='int32')},
        'output': ['detection_boxes', 'detection_scores', 'detection_multiclass_scores', 'num_detections',]
    },
    # NeuMF
    {
        'model_name': 'NeuMF',
        'input_generator': lambda set_seed=True: {'input/user_onehot':np.array([[0] * 6040] * 10, dtype='float32'),
                  'input/item_onehot':np.array([[0] * 3706] * 10, dtype='float32')
                  },
        'output': ['evaluation/TopKV2',]
    }
]

