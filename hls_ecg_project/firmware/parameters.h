#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"
#include "nnet_utils/nnet_conv1d.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_stream.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"
#include "nnet_utils/nnet_sepconv1d_stream.h"

// hls-fpga-machine-learning insert weights
#include "weights/w25.h"
#include "weights/b25.h"
#include "weights/w7.h"
#include "weights/b7.h"
#include "weights/w9.h"
#include "weights/b9.h"
#include "weights/w26.h"
#include "weights/b26.h"
#include "weights/w27.h"
#include "weights/b27.h"
#include "weights/s21.h"
#include "weights/b21.h"
#include "weights/w23.h"
#include "weights/b23.h"


// hls-fpga-machine-learning insert layer-config
// embedding
struct config28_mult : nnet::dense_config {
    static const unsigned n_in = 1;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef embedding_bias_t bias_t;
    typedef embedding_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config28 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 140;
    static const unsigned n_chan = 1;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 140;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 140;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 140;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_28<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef embedding_bias_t bias_t;
    typedef embedding_weight_t weight_t;
    typedef config28_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = nnet::Conv1DLatency<data_T, res_T, CONFIG_T>;
};
const ap_uint<config28::filt_width> config28::pixels[] = {0};

// emb_relu
struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = 2240;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef emb_relu_table_t table_t;
};

// att_pool
struct config6 : nnet::pooling1d_config {
    static const unsigned n_in = 140;
    static const unsigned n_filt = 16;
    static const nnet::Pool_Op pool_op = nnet::Average;
    static const unsigned reuse_factor = 1;
    typedef model_default_t accum_t;
};

// att_dense1
struct config7 : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 8;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 128;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef att_dense1_bias_t bias_t;
    typedef att_dense1_weight_t weight_t;
    typedef layer7_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// att_dense1_relu
struct relu_config8 : nnet::activ_config {
    static const unsigned n_in = 8;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef att_dense1_relu_table_t table_t;
};

// att_dense2
struct config9 : nnet::dense_config {
    static const unsigned n_in = 8;
    static const unsigned n_out = 16;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 128;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef att_dense2_bias_t bias_t;
    typedef att_dense2_weight_t weight_t;
    typedef layer9_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// att_dense2_sigmoid
struct sigmoid_config10 : nnet::activ_config {
    static const unsigned n_in = 16;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef att_dense2_sigmoid_table_t table_t;
};

// att_mult
struct config12 : nnet::merge_config {
    static const unsigned n_elem = 140*16;
    static const unsigned reuse_factor = 1;
};

// att_add
struct config13 : nnet::merge_config {
    static const unsigned n_elem = 140*16;
    static const unsigned reuse_factor = 1;
};

// ffn1
struct config29_mult : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef ffn1_bias_t bias_t;
    typedef ffn1_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config29 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 140;
    static const unsigned n_chan = 16;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 32;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 140;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 140;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 140;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_29<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef ffn1_bias_t bias_t;
    typedef ffn1_weight_t weight_t;
    typedef config29_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = nnet::Conv1DLatency<data_T, res_T, CONFIG_T>;
};
const ap_uint<config29::filt_width> config29::pixels[] = {0};

// ffn_relu
struct relu_config17 : nnet::activ_config {
    static const unsigned n_in = 4480;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef ffn_relu_table_t table_t;
};

// ffn2
struct config30_mult : nnet::dense_config {
    static const unsigned n_in = 32;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    typedef model_default_t accum_t;
    typedef ffn2_bias_t bias_t;
    typedef ffn2_weight_t weight_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

struct config30 : nnet::conv1d_config {
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_width = 140;
    static const unsigned n_chan = 32;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_width;
    static const unsigned n_filt = 16;
    static const unsigned stride_width = 1;
    static const unsigned dilation = 1;
    static const unsigned out_width = 140;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned multiplier_limit =
        DIV_ROUNDUP(kernel_size * n_chan * n_filt, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_width = 140;
    static const ap_uint<filt_width> pixels[min_width];
    static const unsigned n_partitions = 140;
    static const unsigned n_pixels = out_width / n_partitions;
    template<class data_T, class CONFIG_T>
    using fill_buffer = nnet::fill_buffer_30<data_T, CONFIG_T>;
    typedef model_default_t accum_t;
    typedef ffn2_bias_t bias_t;
    typedef ffn2_weight_t weight_t;
    typedef config30_mult mult_config;
    template<unsigned K, unsigned S, unsigned W>
    using scale_index = nnet::scale_index_regular<K, S, W>;
    template<class data_T, class res_T, class CONFIG_T>
    using conv_kernel = nnet::Conv1DLatency<data_T, res_T, CONFIG_T>;
};
const ap_uint<config30::filt_width> config30::pixels[] = {0};

// ffn_add
struct config20 : nnet::merge_config {
    static const unsigned n_elem = 140*16;
    static const unsigned reuse_factor = 1;
};

// final_bn
struct config21 : nnet::batchnorm_config {
    static const unsigned n_in = 140*16;
    static const unsigned n_filt = 16;
    static const unsigned n_scale_bias = (n_filt == -1) ? n_in : n_filt;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    static const bool store_weights_in_bram = false;
    typedef final_bn_bias_t bias_t;
    typedef final_bn_scale_t scale_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// global_pool
struct config22 : nnet::pooling1d_config {
    static const unsigned n_in = 140;
    static const unsigned n_filt = 16;
    static const nnet::Pool_Op pool_op = nnet::Average;
    static const unsigned reuse_factor = 1;
    typedef model_default_t accum_t;
};

// output
struct config23 : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 2;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 32;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef output_bias_t bias_t;
    typedef output_weight_t weight_t;
    typedef layer23_index index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::DenseLatency<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// output_softmax
struct softmax_config24 : nnet::activ_config {
    static const unsigned n_in = 2;
    static const unsigned n_slice = 2;
    static const unsigned n_outer = 1;
    static const unsigned n_inner = 1;
    static const unsigned parallelization_factor = -1;
    static const unsigned exp_table_size = 1024;
    static const unsigned inv_table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned axis = -1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::stable;
    static constexpr float exp_scale = 1.0;
    typedef output_softmax_exp_table_t exp_table_t;
    typedef output_softmax_inv_table_t inv_table_t;
    typedef model_default_t accum_t;
    typedef output_softmax_inv_inp_t inv_inp_t;
    typedef ap_ufixed<36, 16> inp_norm_t;
};



#endif
