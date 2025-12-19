#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<33,13> embedding_result_t;
typedef ap_fixed<16,6> embedding_weight_t;
typedef ap_fixed<16,6> embedding_bias_t;
typedef ap_fixed<16,6> layer5_t;
typedef ap_fixed<18,8> emb_relu_table_t;
typedef ap_fixed<16,6> layer6_t;
typedef ap_fixed<37,17> att_dense1_result_t;
typedef ap_fixed<16,6> att_dense1_weight_t;
typedef ap_fixed<16,6> att_dense1_bias_t;
typedef ap_uint<1> layer7_index;
typedef ap_fixed<16,6> layer8_t;
typedef ap_fixed<18,8> att_dense1_relu_table_t;
typedef ap_fixed<36,16> att_dense2_result_t;
typedef ap_fixed<16,6> att_dense2_weight_t;
typedef ap_fixed<16,6> att_dense2_bias_t;
typedef ap_uint<1> layer9_index;
typedef ap_fixed<16,6> layer10_t;
typedef ap_fixed<18,8> att_dense2_sigmoid_table_t;
typedef ap_fixed<32,12> att_mult_result_t;
typedef ap_fixed<33,13> att_add_result_t;
typedef ap_fixed<54,24> ffn1_result_t;
typedef ap_fixed<16,6> ffn1_weight_t;
typedef ap_fixed<16,6> ffn1_bias_t;
typedef ap_fixed<16,6> layer17_t;
typedef ap_fixed<18,8> ffn_relu_table_t;
typedef ap_fixed<38,18> ffn2_result_t;
typedef ap_fixed<16,6> ffn2_weight_t;
typedef ap_fixed<16,6> ffn2_bias_t;
typedef ap_fixed<39,19> ffn_add_result_t;
typedef ap_fixed<56,26> final_bn_result_t;
typedef ap_fixed<16,6> final_bn_scale_t;
typedef ap_fixed<16,6> final_bn_bias_t;
typedef ap_fixed<16,6> layer22_t;
typedef ap_fixed<37,17> output_result_t;
typedef ap_fixed<16,6> output_weight_t;
typedef ap_fixed<16,6> output_bias_t;
typedef ap_uint<1> layer23_index;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<18,8> output_softmax_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> output_softmax_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> output_softmax_inv_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> output_softmax_inv_inp_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
