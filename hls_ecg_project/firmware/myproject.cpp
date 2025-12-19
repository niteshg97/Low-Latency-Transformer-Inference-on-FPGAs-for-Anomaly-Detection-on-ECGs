#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t ecg_input[140*1],
    result_t layer24_out[2]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=ecg_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer24_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=ecg_input,layer24_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<embedding_weight_t, 16>(w25, "w25.txt");
        nnet::load_weights_from_txt<embedding_bias_t, 16>(b25, "b25.txt");
        nnet::load_weights_from_txt<att_dense1_weight_t, 128>(w7, "w7.txt");
        nnet::load_weights_from_txt<att_dense1_bias_t, 8>(b7, "b7.txt");
        nnet::load_weights_from_txt<att_dense2_weight_t, 128>(w9, "w9.txt");
        nnet::load_weights_from_txt<att_dense2_bias_t, 16>(b9, "b9.txt");
        nnet::load_weights_from_txt<ffn1_weight_t, 512>(w26, "w26.txt");
        nnet::load_weights_from_txt<ffn1_bias_t, 32>(b26, "b26.txt");
        nnet::load_weights_from_txt<ffn2_weight_t, 512>(w27, "w27.txt");
        nnet::load_weights_from_txt<ffn2_bias_t, 16>(b27, "b27.txt");
        nnet::load_weights_from_txt<final_bn_scale_t, 16>(s21, "s21.txt");
        nnet::load_weights_from_txt<final_bn_bias_t, 16>(b21, "b21.txt");
        nnet::load_weights_from_txt<output_weight_t, 32>(w23, "w23.txt");
        nnet::load_weights_from_txt<output_bias_t, 2>(b23, "b23.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    embedding_result_t layer25_out[140*16];
    #pragma HLS ARRAY_PARTITION variable=layer25_out complete dim=0

    layer5_t layer5_out[140*16];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    layer6_t layer6_out[16];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    att_dense1_result_t layer7_out[8];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0

    layer8_t layer8_out[8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0

    att_dense2_result_t layer9_out[16];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0

    layer10_t layer10_out[16];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0

    auto& layer11_out = layer10_out;
    att_mult_result_t layer12_out[140*16];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0

    att_add_result_t layer13_out[140*16];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0

    ffn1_result_t layer26_out[140*32];
    #pragma HLS ARRAY_PARTITION variable=layer26_out complete dim=0

    layer17_t layer17_out[140*32];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0

    ffn2_result_t layer27_out[140*16];
    #pragma HLS ARRAY_PARTITION variable=layer27_out complete dim=0

    ffn_add_result_t layer20_out[140*16];
    #pragma HLS ARRAY_PARTITION variable=layer20_out complete dim=0

    final_bn_result_t layer21_out[140*16];
    #pragma HLS ARRAY_PARTITION variable=layer21_out complete dim=0

    layer22_t layer22_out[16];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0

    output_result_t layer23_out[2];
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0

    nnet::pointwise_conv_1d_cl<input_t, embedding_result_t, config28>(ecg_input, layer25_out, w25, b25); // embedding

    nnet::relu<embedding_result_t, layer5_t, relu_config5>(layer25_out, layer5_out); // emb_relu

    nnet::global_pooling1d_cl<layer5_t, layer6_t, config6>(layer5_out, layer6_out); // att_pool

    nnet::dense<layer6_t, att_dense1_result_t, config7>(layer6_out, layer7_out, w7, b7); // att_dense1

    nnet::relu<att_dense1_result_t, layer8_t, relu_config8>(layer7_out, layer8_out); // att_dense1_relu

    nnet::dense<layer8_t, att_dense2_result_t, config9>(layer8_out, layer9_out, w9, b9); // att_dense2

    nnet::sigmoid<att_dense2_result_t, layer10_t, sigmoid_config10>(layer9_out, layer10_out); // att_dense2_sigmoid

    nnet::multiply<layer5_t, layer10_t, att_mult_result_t, config12>(layer5_out, layer11_out, layer12_out); // att_mult

    nnet::add<layer5_t, att_mult_result_t, att_add_result_t, config13>(layer5_out, layer12_out, layer13_out); // att_add

    nnet::pointwise_conv_1d_cl<att_add_result_t, ffn1_result_t, config29>(layer13_out, layer26_out, w26, b26); // ffn1

    nnet::relu<ffn1_result_t, layer17_t, relu_config17>(layer26_out, layer17_out); // ffn_relu

    nnet::pointwise_conv_1d_cl<layer17_t, ffn2_result_t, config30>(layer17_out, layer27_out, w27, b27); // ffn2

    nnet::add<att_add_result_t, ffn2_result_t, ffn_add_result_t, config20>(layer13_out, layer27_out, layer20_out); // ffn_add

    nnet::normalize<ffn_add_result_t, final_bn_result_t, config21>(layer20_out, layer21_out, s21, b21); // final_bn

    nnet::global_pooling1d_cl<final_bn_result_t, layer22_t, config22>(layer21_out, layer22_out); // global_pool

    nnet::dense<layer22_t, output_result_t, config23>(layer22_out, layer23_out, w23, b23); // output

    nnet::softmax<output_result_t, result_t, softmax_config24>(layer23_out, layer24_out); // output_softmax

}

