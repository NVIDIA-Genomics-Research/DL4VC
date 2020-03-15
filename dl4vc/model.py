# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Needs to be included *before* h5py to inherit multithreading support
import torch.multiprocessing
import os
import math
from dl4vc.dataset import MAX_READS
from dl4vc.base_enum import *

# Multiply strand enum, in use by neural network input
STRAND_ENCODE_FACTOR = 0.5

NONCE = 0.0001
READ_MIDPOINT = 100
READ_MIDPOINT_DISTANCE = 10
# Re-scale coverage average to something close to 1.0
COVER_AVERAGE_NORM = 1./100.
CONV_CHANNELS = 100
Q_SCORE_SCALE_FACTOR = 1./100.
SINGLE_READ_LENGTH = 201
NUM_SINGLE_READS = 100
BOTTLENECK_SIZE = 16
MIN_RESIDUAL_LAYER = 2 # do not support residuals too early in the network
# big share of final parameters in the FCN -- try reducing FCN size...
# layer_sizes = [1024, 256]
class Basic2DNet(nn.Module):
    """
    Fully conv (2D) network for variant calling -- also a "Conv1D" version that only convolves in "reads" direction -- and pools across reads
    """
    def __init__(self, target_size, layer_sizes = [1024, 256], pre_conv_dropout=0.1, hidden_dropout=0.1,
                 embed_dim=20, pos_embeddings=True, init_conv_channels=CONV_CHANNELS, final_conv_channels=CONV_CHANNELS,
                 ref_concat_at_reads=True, split_ref_reads_groups=False,
                 use_q_scores=False, use_strands=False,
                 use_naive_variant_encoding=False, expand_bases_naive_variant_encoding=True,
                 use_reads_ref_var_mask = True, ref_var_mask_all = False,
                 single_read_len = SINGLE_READ_LENGTH, num_single_reads = NUM_SINGLE_READS,
                 bottleneck_channels=BOTTLENECK_SIZE, bottleneck_linear_outputs=BOTTLENECK_SIZE,
                 append_bottleneck_highway_reads=True, concat_hw_reads=True,
                 reads_sum_concat_at_reads=False,
                 total_conv_layers = 5, residual_layer_start = 0, conv_kernel_size = 3,
                 use_conv_1d=True, conv_1d_pool_append=False, conv_1d_pool_add=True,
                 # use_transformer=True, transformer_encoder_heads = 2, num_transformer_layers = 3,
                 # final_transformer_dims = 64, transformer_residual = True, transformer_feedforward_dim=64,
                 conv_1d_pool_layers=[2], use_batchnorm=False,
                 early_loss_layers=[], learn_context_early_loss_balance=True,
                 pool_combine_dimension = 0, skip_final_maxpool=False,
                 final_layer_dilation = 1, middle_layer_dilation = 1,
                 append_trust_region=False, append_num_reads=False, append_allele_frequency=False, args={}):
        super(Basic2DNet, self).__init__()
        # Channels for the convolutions -- expand? Double at every layer?
        self.init_conv_channels = init_conv_channels
        self.final_conv_channels = final_conv_channels
        # How long are the reads?
        self.single_read_len = single_read_len
        self.num_single_reads = num_single_reads
        # Only mean pool? Saves params
        self.skip_final_maxpool = skip_final_maxpool
        # From pooling to X dimension (need defined since pooling dimension could be different by layer)
        self.pool_combine_dimension = pool_combine_dimension
        # Always use learned embeddings.
        # Do we += positional embeddings? May not be necessary in no-pool or low-pool case
        self.pos_embeddings = pos_embeddings
        # Option: do we concat ref @ all reads? Instead of appending it once as "N+1st read"?
        self.ref_concat_at_reads = ref_concat_at_reads
        # If concat ref and reads -- do we split into groups at conv2D concat level? [separately convolve ref and reads]
        self.split_ref_reads_groups = split_ref_reads_groups
        # Does model expect Q-scores along with reads and reference?
        self.use_q_scores = use_q_scores
        # Does model expect strand direction, same as Q-scores
        self.use_strands = use_strands
        # HACK -- pass AF, ref, variant (base seq) to the model -- useful in multi-allele case
        # [Just pass to the FCN]
        # TODO -- *mark* ref and *var*(!) bases individually in the reads -- as 42nd channel. But do it carefully
        self.use_naive_variant_encoding = use_naive_variant_encoding
        # Do we expand those bases in the encoding?
        self.expand_bases_naive_variant_encoding = expand_bases_naive_variant_encoding
        # More sophisticated -- mask the reference and variants -- in each read
        self.use_reads_ref_var_mask = use_reads_ref_var_mask
        # For ^^ ref/var mask -- can it work if we just pass (1) for all masks, for all reads? [instead on finding what matches]
        #self.ref_var_mask_all = ref_var_mask_all

        # Dimensions for bottleneck and Linear layer -- useful to pass a small amount of info from single reads directly to FCN
        self.bottleneck_channels = bottleneck_channels
        self.bottleneck_linear_outputs = bottleneck_linear_outputs
        self.append_bottleneck_highway_reads = append_bottleneck_highway_reads
        # Concat HW -- or just average then followed by RelU? [sum/average is WaveNet style, but seems to train less well?]
        self.concat_hw_reads = concat_hw_reads
        # Do we concat the *sum* for reads to every read? Make it easier to spot disagreements -- and run a "average stats" model implicitly
        self.reads_sum_concat_at_reads = reads_sum_concat_at_reads
        # How many conv layers? Try 3-6 and compare results
        # TODO: Declare these in a loop -- rather than named and numbered
        self.total_conv_layers = total_conv_layers
        # Do we make last X layers residuals?
        # NOTE: Will not make last conv layer residual if # output channels doesn't match
        # NOTE: Keep 0 for no residual connections
        self.residual_layer_start = residual_layer_start
        # Early Loss -- backprop loss fror earlier layers -- as an aux to get better initialization
        self.early_loss_layers = early_loss_layers
        for l in early_loss_layers:
            assert l < self.total_conv_layers, "Can not allow early loss layers > allowed conv layers %s" % str(early_loss_layers)
        # If we learn to balance late and early loss -- buckle up -- Mixture of Softmax
        self.learn_context_early_loss_balance = learn_context_early_loss_balance

        # HACK: Transformer (encoder) layers *after* init convolutions?
        self.use_transformer = args.use_transformer
        self.transformer_encoder_nheads = args.transformer_encoder_heads
        # TODO: Option to change Transformer dimension from output of the Conv (128 works well enough)
        # [or maybe using self.final_conv_channels] works well enough.
        # NOTE: Model will crash if not( self.final_conv_channels // args.transformer_encoder_heads )
        self.num_transformer_layers = args.num_transformer_layers
        self.transformer_feedforward_dim = args.transformer_feedforward_dim
        self.final_transformer_dims = args.final_transformer_dims if args.final_transformer_dims > 0 else final_conv_channels
        self.transformer_residual = args.transformer_residual
        self.transformer_encoder_dropout = args.transformer_encoder_dropout

        # Do we use conv2D or conv1D (pool across reads only)? Good arguments for both
        # [or do we combine both?]
        self.use_conv_1d = use_conv_1d
        # How big a kernel? TODO: Support wider kernels by layer?
        self.conv_kernel_size = conv_kernel_size
        # Option to supply "average pool" information from lower levels to higher convolutions.
        # Why? So that read-level disagreement can be captured before final pooling & FCN
        self.conv_1d_pool_append = conv_1d_pool_append
        self.conv_1d_pool_add = conv_1d_pool_add
        # Which pool append layers? Ex [2] means pool at 2nd layer only. Valid 1-3 [always pool after final layer also]
        self.conv_1d_pool_layers = conv_1d_pool_layers
        # Do we apply batch norm to 1D, or 2D model?
        self.use_batchnorm = use_batchnorm
        # Do we inform network that this is (more certain) Trust Region example?
        self.append_trust_region = append_trust_region
        # Do we inform the network how many non-zero reads? Could simply calculate it, but can also supply
        self.append_num_reads = append_num_reads
        # HACK -- append AF -- to disambiguate long inserts until we fix length cap issues
        self.append_AF = append_allele_frequency
        # (4x for ATCG, 1x for "insert") --  and 1x  padding, if needed
        self.vocab_size = len(enum_base)
        self.embed_dim = embed_dim
        self.embeddings = nn.Embedding(self.vocab_size, self.embed_dim,
                                        padding_idx=base_enum['pad'], sparse=False,
                                        scale_grad_by_freq=True)

        # Apply dropout, but only if we expand dimensions to embedding space!
        self.pre_conv_dropout = pre_conv_dropout
        self.bottleneck_linear_outputs = bottleneck_linear_outputs

        # Positional embeddings -- via the Transformer
        # http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
        # Why? Else pooling or attention layers lose position information -- especially the important center position
        max_len = single_read_len
        d_model = embed_dim
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        # Four (why not more?) layers of 2D conv
        # TODO: Specify number of layers
        # TODO: Batch norm!
        # TODO: Dilation, for bigger receptive fields...
        # TODO: Non-square kernels (why not)
        channel_multiplier = 1 + int(self.ref_concat_at_reads) + int(self.reads_sum_concat_at_reads)
        channel_adder = 0 + (1 if self.use_q_scores else 0) + (1 if self.use_strands else 0)
        if self.use_q_scores:
            print('Input using Q-scores')
        if self.use_strands:
            print('Input using Strand')
        if self.use_reads_ref_var_mask:
            print('Input using reads_ref_var_mask')
            channel_adder += 3
        input_channels = channel_multiplier * self.embed_dim + channel_adder
        print('Init conv1D input will be %d channels (%d added channels on top of embeddings)' % (input_channels, channel_adder))

        # Alternative -- Try 1D convs...
        # One easy way to do this is to do 2D conv of shape (height x N) -- where height == number of operations
        # Another way is (1 x N) -- and we pool in higher dimensions
        # NOTE: We also (average) pool lower levels, and supply that as side-input (optionally) at higher dimension

        # For variable conv depth -- store several item
        # Conv, batchnorm for each layer
        # *single* average pool -- since we only support this once (for now)
        # Bottleneck for each layer -- if bottleneck requested
        self.conv1D_layers = []
        self.bn1D_layers = []
        self.inter_ave_pool1D_layers = []
        for _ in self.conv_1d_pool_layers:
            self.inter_ave_pool1D_layers.append(nn.AvgPool2d(kernel_size=(MAX_READS, 1), padding=0, ceil_mode=True))
        self.inter_ave_pool1D_layers = nn.ModuleList(self.inter_ave_pool1D_layers)
        self.conv1D_bottleneck_layers = []
        self.conv1D_compression_layers = []
        self.is_residual_layer = []
        # Need extra 1x1 for residual layers [after the ReLU] -- to make switch-off easier
        self.residual_conv_layers = []
        self.add_pooling_layer = []
        # HACK -- Try dilated layers... bigger reception field. Perhaps still padding?
        self.middle_layer_dilation = middle_layer_dilation
        self.final_layer_dilation = final_layer_dilation
        # Iterate through 1..total_layers (inclusive)
        print('Initializing Conv1D model with %d conv layers' % self.total_conv_layers)
        if self.residual_layer_start > 0:
            print('Will start residual connects after layer %s' % self.residual_layer_start)
            assert self.residual_layer_start >= MIN_RESIDUAL_LAYER, "Do not allow residuals starting at conv layer %s" % self.residual_layer_start
        pool_layer_index = 0
        for l_num in range(1,self.total_conv_layers+1):
            # special rules for first (init) layer
            if l_num == 1:
                conv_layer = nn.Conv2d(in_channels=input_channels, out_channels=self.init_conv_channels,
                                        kernel_size=(1, self.conv_kernel_size), stride=1, padding=(0,1), dilation=1,
                                        groups=(channel_multiplier if self.split_ref_reads_groups else 1), bias=True)
                bn_layer = nn.BatchNorm2d(self.init_conv_channels)
            elif l_num < self.total_conv_layers:
                print('Dilation %d at layer %d' % (self.middle_layer_dilation, l_num))
                conv_layer = nn.Conv2d(in_channels=(2 if self.conv_1d_pool_append and (l_num-1) in self.conv_1d_pool_layers else 1)*self.init_conv_channels,
                                        out_channels=self.init_conv_channels, kernel_size=(1, self.conv_kernel_size),
                                        stride=1, padding=(0,self.middle_layer_dilation), dilation=self.middle_layer_dilation, groups=1, bias=True)
                bn_layer = nn.BatchNorm2d(self.init_conv_channels)
            else:
                print('Dilation %d at final layer %d' % (self.final_layer_dilation, l_num))
                conv_layer = nn.Conv2d(in_channels=(2 if self.conv_1d_pool_append and (l_num-1) in self.conv_1d_pool_layers else 1)*self.init_conv_channels,
                                        out_channels=self.final_conv_channels, kernel_size=(1, self.conv_kernel_size),
                                        stride=1, padding=(0,self.final_layer_dilation), dilation=self.final_layer_dilation, groups=1, bias=True)
                bn_layer = nn.BatchNorm2d(self.final_conv_channels)
            self.conv1D_layers.append(conv_layer)
            self.bn1D_layers.append(bn_layer)

            # If conv pool layer should be applied *after* this layer, note which index
            if l_num in self.conv_1d_pool_layers:
                self.add_pooling_layer.append(pool_layer_index)
                print('Adding pooling += after layer %d -- pooling layer %d of %d' % (l_num, pool_layer_index, len(self.inter_ave_pool1D_layers)))
                pool_layer_index += 1
            else:
                # Should crash, if called -- make sure we call the right pooling layer
                self.add_pooling_layer.append(999)

            # Compute whether layer should be residual
            # TODO: Initialization?
            is_res_layer = False

            if self.residual_layer_start > 0 and l_num >= self.residual_layer_start and not (l_num == self.total_conv_layers and self.init_conv_channels != self.final_conv_channels):
                is_res_layer = True
                print('Layer %d is residual' % l_num)
                residual_conv_layer = nn.Conv2d(in_channels=(self.init_conv_channels if l_num < self.total_conv_layers else self.final_conv_channels),
                    out_channels=(self.init_conv_channels if l_num < self.total_conv_layers else self.final_conv_channels),
                    kernel_size=(1,1))
                self.residual_conv_layers.append(residual_conv_layer)
            self.is_residual_layer.append(is_res_layer)

            # Add bottleneck highway, upon request
            if self.append_bottleneck_highway_reads:
                bottleneck_layer = nn.Conv2d(in_channels=(self.init_conv_channels if l_num < self.total_conv_layers else self.final_conv_channels),
                                        out_channels=self.bottleneck_channels, kernel_size=(1,1))
                compression_layer = nn.Conv2d(in_channels=self.bottleneck_channels,
                                        out_channels=self.bottleneck_linear_outputs, kernel_size=(1,self.single_read_len))
                self.conv1D_bottleneck_layers.append(bottleneck_layer)
                self.conv1D_compression_layers.append(compression_layer)

        # Make sure CUDA picks up all of the modules
        self.conv1D_layers = nn.ModuleList(self.conv1D_layers)
        self.bn1D_layers = nn.ModuleList(self.bn1D_layers)
        if self.append_bottleneck_highway_reads:
            self.conv1D_bottleneck_layers = nn.ModuleList(self.conv1D_bottleneck_layers)
            self.conv1D_compression_layers = nn.ModuleList(self.conv1D_compression_layers)
        if len(self.residual_conv_layers) > 0:
            self.residual_conv_layers = nn.ModuleList(self.residual_conv_layers)

        # Optionally, add Transformer (encoder) after conv layers
        # TODO: Also affects averaging, and final output to the fully connect layers...
        conv_out_dim = self.final_conv_channels
        final_out_dims = self.final_transformer_dims
        feedforward_dims = self.transformer_feedforward_dim
        # Naive -- just take input from 1D conv outputs, each layer do encoder then LayerNorm. Use final output.
        if self.use_transformer:
            self.transformer_encoder_layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=conv_out_dim,
                nhead=self.transformer_encoder_nheads,
                dim_feedforward=feedforward_dims,
                dropout=self.transformer_encoder_dropout) for i in range(self.num_transformer_layers)])
            self.transformer_layer_norms = nn.ModuleList([nn.LayerNorm(conv_out_dim) for i in range(self.num_transformer_layers)])
            print('For Transformer model, using %d layers' % self.num_transformer_layers)
            # Reduce dimension going into the pooling layers? This is useful because it can greatly reduce total param count
            if final_out_dims != conv_out_dim:
                print('Adding dimenion reduction layer %d -> %d after Transformer' % (conv_out_dim, final_out_dims))
                self.transformer_dim_reduction_layer = nn.Conv2d(
                    in_channels=conv_out_dim, out_channels=final_out_dims, kernel_size=(1,1))
            else:
                self.transformer_dim_reduction_layer = nn.Identity()
            print('Final Transformer output dim: %d' % final_out_dims)
            conv_out_dim = final_out_dims

        conv_total_out = (1 if self.skip_final_maxpool else 2) * conv_out_dim * self.single_read_len
        conv_total_out_early = (1 if self.skip_final_maxpool else 2) * self.init_conv_channels * self.single_read_len
        print('Using 1D conv model, adding %d outputs' % conv_total_out)

        # Now the tricky part -- pooling
        # 1D Conv -- do a *full pool* across # reads only.
        if not self.skip_final_maxpool:
            self.maxPool1D = nn.MaxPool2d(kernel_size=(MAX_READS, 1), padding=0, dilation=1, return_indices=False, ceil_mode=True)
        self.avgPool1D = nn.AvgPool2d(kernel_size=(MAX_READS, 1), padding=0, ceil_mode=True)
        # Add 1x1 conv *after* concat the pool
        # Why? 1. Allow averaging pool outputs across layers 2. Learn to weight these appropriately 3. Control FCN dimensions
        # If zero -- then no layer at all! Pass through like we had before; saves parameters?
        if self.pool_combine_dimension > 0:
            print('reducing pooled output to %d dimensions' % self.pool_combine_dimension)
            self.post_pool_conv1D = nn.Linear(conv_total_out, self.pool_combine_dimension)

        # Early loss -- also a pool & FCN
        if not self.skip_final_maxpool:
            self.maxPool1DEarly = [nn.MaxPool2d(kernel_size=(MAX_READS, 1), padding=0, dilation=1, return_indices=False, ceil_mode=True) for i in self.early_loss_layers]
            self.maxPool1DEarly = nn.ModuleList(self.maxPool1DEarly)
        self.avgPool1DEarly = [nn.AvgPool2d(kernel_size=(MAX_READS, 1), padding=0, ceil_mode=True) for i in self.early_loss_layers]
        self.avgPool1DEarly = nn.ModuleList(self.avgPool1DEarly)
        if self.pool_combine_dimension > 0:
            self.post_pool_conv1D_early = [nn.Linear(conv_total_out_early, self.pool_combine_dimension) for i in self.early_loss_layers]
            self.post_pool_conv1D_early = nn.ModuleList(self.post_pool_conv1D_early)

        # All of the convolutions are joined into a final layer.
        # NOTE: Conv2D no longer supported -- look for older versions of the code with this option.
        assert self.use_conv_1d, "Need use_conv_1d as other methods no longer supported."

        # FCN will start with output from 1x1 pool output
        input_layer_size =  self.pool_combine_dimension if self.pool_combine_dimension > 0 else conv_total_out
        input_layer_size_early = [(self.pool_combine_dimension if self.pool_combine_dimension > 0 else conv_total_out_early) for i in self.early_loss_layers]

        # Append highway from single reads?
        num_highway_layers = self.total_conv_layers
        if self.concat_hw_reads:
            print('Using concat (not average) for HW outputs from every layer [more params]')
        if self.append_bottleneck_highway_reads:
            # Averaging all HW outputs -- don't grow the input size as # layer increases
            total_highway_inputs = (num_highway_layers if self.concat_hw_reads else 1) * self.bottleneck_linear_outputs * self.num_single_reads
            print('Appending %d highway inputs from %d layers' % (total_highway_inputs, self.total_conv_layers))
            input_layer_size += total_highway_inputs
            for i,l in enumerate(self.early_loss_layers):
                # Averaging all HW outputs -- don't grow the input size as # layer increases
                early_highway_inputs = (l if self.concat_hw_reads else 1) * self.bottleneck_linear_outputs * self.num_single_reads
                print('Appending %d early highway inputs from %d layers' % (early_highway_inputs, l))
                input_layer_size_early[i] += early_highway_inputs

        # Append extra information to FCN input? -- reads coverage (number of non-zero reads present)
        if self.append_num_reads:
            assert False, "append_num_reads is deprecated"

        # Append extra information to FCN input? -- is example in trust region [thus higher confidence]
        if self.append_trust_region:
            assert False, "append_trust_region is deprecated"

        # Append AF (allele frequency) -- an ugly hack -- but a necessary one to
        # disambiguate very long inserts (until we fix data)
        if self.append_AF and not self.use_naive_variant_encoding:
            assert False, "append_AF is deprecated"

        # Append naive info about the variant? For multi-allele (different predictions for same pileup)
        if self.use_naive_variant_encoding:
            assert False, 'use_naive_variant_encoding is deprecated'

        print('Adding fully connected hidden layers %s' % str(layer_sizes))
        self.layer_sizes = [input_layer_size] + list(map(int, layer_sizes))
        self.final_hidden_size = self.layer_sizes[-1]
        self.dropout = hidden_dropout
        self.nonlinearity = nn.ReLU()

        # Build the connector, between conv layers, and final output to output layers
        layer_list = []
        # Dropout layer from conv, before fully connected
        if self.dropout:
            layer_list.extend([nn.Dropout(p=self.dropout)])
        layer_list.extend(list(chain.from_iterable(
            [[nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]), self.nonlinearity, nn.Dropout(p=self.dropout)] for i in range(len(self.layer_sizes) - 1)]
        )))
        print(layer_list)
        self.conv2hidden = nn.Sequential(*layer_list)

        # Similar FCN for each intermediate (early) output
        self.conv2hidden_early = []
        self.fcHidden2Bin_early = []
        self.fcHidden2VT_early = []
        for i,early_input_size in enumerate(input_layer_size_early):
            early_layer_sizes = [early_input_size] + list(map(int, layer_sizes))
            early_layer_list = []
            if self.dropout:
                early_layer_list.extend([nn.Dropout(p=self.dropout)])
            early_layer_list.extend(list(chain.from_iterable(
                [[nn.Linear(early_layer_sizes[i], early_layer_sizes[i+1]), self.nonlinearity, nn.Dropout(p=self.dropout)] for i in range(len(self.layer_sizes) - 1)]
            )))
            print(early_layer_list)
            early_conv2hidden = nn.Sequential(*early_layer_list)
            self.conv2hidden_early.append(early_conv2hidden)

            # Also final early outputs -- for binary and VT only
            self.fcHidden2Bin_early.append(nn.Linear(self.final_hidden_size, 2))
            self.fcHidden2VT_early.append(nn.Linear(self.final_hidden_size, 3))

        # Module list, for CUDA
        if len(input_layer_size_early) > 0:
            self.conv2hidden_early = nn.ModuleList(self.conv2hidden_early)
            self.fcHidden2Bin_early = nn.ModuleList(self.fcHidden2Bin_early)
            self.fcHidden2VT_early = nn.ModuleList(self.fcHidden2VT_early)

        # Binary output [binary = mutation or not]
        self.fcHidden2BinTarget = nn.Linear(self.final_hidden_size, 2)
        # Auxillary loss pedictions
        # Variant Type, Allele Freq, Coverage, Variant Base (for SNP), Variant Ref (base)
        self.fcHidden2VT = nn.Linear(self.final_hidden_size, 3)
        #self.vt_softmax = nn.LogSoftmax(dim=1)
        self.fcHidden2AF = nn.Linear(self.final_hidden_size, 1)
        self.fcHidden2Coverage = nn.Linear(self.final_hidden_size, 1)
        self.fcHidden2VB = nn.Linear(self.final_hidden_size, len(enum_base))
        #self.vb_softmax = nn.LogSoftmax(dim=1)
        self.fcHidden2VR = nn.Linear(self.final_hidden_size, len(enum_base))
        #self.vr_softmax = nn.LogSoftmax(dim=1)

        # If learning context-based early/later loss balance...
        # A. Predict *softmax* over output heads (early/final output) based on [final] context
        # B. Get softmax predictions on output heads (not just logits)
        # C. Multiply the "MoS" -- for single ouputs
        # D. Make sure receiving function can handle softmax > logits on the other end
        if self.learn_context_early_loss_balance and len(self.early_loss_layers) > 0:
            self.binary_context_loss_balance = nn.Linear(self.final_hidden_size, len(self.early_loss_layers) + 1)
            self.vt_context_loss_balance = nn.Linear(self.final_hidden_size, len(self.early_loss_layers) + 1)

        # HACK -- learn output weights? -- weight between preliminary and final outputs
        print('Initializing model parameters for %d early loss layers' % len(self.early_loss_layers))
        self.bin_output_weights = torch.nn.Parameter(torch.ones(len(self.early_loss_layers)+1) * 0.1)
        print(self.bin_output_weights)
        self.vt_output_weights = torch.nn.Parameter(torch.ones(len(self.early_loss_layers)+1) * 0.1)
        print(self.vt_output_weights)

    def forward(self, reads, ref, q_scores, strands, binary_trust_vector,
        af_scores, ref_bases, var_bases, ref_masks, var_masks,
        rm_non_var_reads=0, rm_var_reads=0, debug=False):
        # With single reads, need a bit of conversion
        # TODO: Should be separate function
        # A. Translate reads and reference to embeddings via lookup table
        # B. Compute summary stats from reads -- networks should predict these
        # C. Encode reads to produce 201-len output predicting summary stats
        # D. Basic encoding for each single read: CNN(cat(summary, read, reference)+positional encoding) + FCN
        # E. Sum all read encodings
        # F. FCN from encodings sum, to final output (mutation or not)
        if debug:
            print('input reads and reference:')
            print(reads.shape)
            print(ref.shape)
            print('converting to %d dim base embeddings' % self.embed_dim)
        reads_emb = self.embeddings(reads)
        ref_emb = self.embeddings(ref)
        if debug:
            print('reads and references shape:')
            print(reads_emb.shape)
            print(ref_emb.shape)

        # Encode empty read -- to use for dynamic down-sampling (or directional augmentation)
        # HACK: Hardcoded that *pad* == 0
        empty_read = torch.zeros((1, reads_emb.shape[1], 1)).long().cuda()
        empty_read_emb = self.embeddings(empty_read)

        # If requested, add (+=) positional embeddings to all reads
        if self.pos_embeddings:
            # Compute positional embeddings and += these to the reference and reads
            # Expand into two dimensions -- batch size, and num reads
            pos_ref = self.pe[:, :reads_emb.size(1)]
            pos_ref = pos_ref.expand(reads_emb.shape[0], pos_ref.shape[1], pos_ref.shape[2])
            pos_ref = pos_ref.unsqueeze(2)
            pos_ref = pos_ref.expand(pos_ref.shape[0], pos_ref.shape[1], reads_emb.shape[2] + (0 if self.ref_concat_at_reads else 1), pos_ref.shape[3])
            pos_ref.requires_grad = False

        # Compute a weighted sum of the read embeddings (summary network)
        # WARNING -- empty_read_emb -- not supported
        if self.reads_sum_concat_at_reads:
            # For each input element (base, insert, etc):
            # A. Mask (equals this element)
            # B. Sum (in num reads dimension)
            # C. Concat
            # D. Multiply by element embeddings matrix
            # E. Sum and divide by num reads
            embedding_translation_matrix = self.embeddings(torch.Tensor([ix for ix in range(len(enum_base))]).long().cuda())
            read_counts = []
            for r_i in range(len(enum_base)):
                reads_mask = reads.eq(r_i)
                reads_mask_sum = torch.sum(reads_mask, dim=2).unsqueeze(dim=2)
                read_counts.append(reads_mask_sum)
            read_counts = torch.cat(read_counts, dim=2)
            read_counts_sum = torch.matmul(read_counts.float(), embedding_translation_matrix)
            # Divide by number of reads
            read_counts_sum /= reads.shape[2]

        # Naive hack network:
        # 1. combine reads and reference (and maybe sum of the channels too)
        # 2. transpose to correct dimensions (#channel should be #embedding dim)
        # 3. 2D conv with no max pool
        # 4. FCN to output

        # Either concat the references as "N+1st" dimension of reads
        # ...or append references at every every dimension, and expand to 2x channels
        # ...also pass the average embeddings, for 3x the channels
        if self.ref_concat_at_reads:
            ref_emb = ref_emb.unsqueeze(2)
            ref_emb = ref_emb.expand(reads_emb.shape)
            # Add positional embeddings
            if self.pos_embeddings:
                reads_emb = reads_emb + pos_ref
                ref_emb = ref_emb + pos_ref
                empty_read_emb = empty_read_emb + pos_ref.narrow(2,0,1).narrow(0,0,1)
            # Add sum by position
            if self.reads_sum_concat_at_reads:
                read_counts_sum = read_counts_sum.unsqueeze(2)
                read_counts_sum = read_counts_sum.expand(reads_emb.shape)
                if self.pos_embeddings:
                    read_counts_sum = read_counts_sum + pos_ref
                combo_input = torch.cat((reads_emb, ref_emb, read_counts_sum), dim=3)
            else:
                combo_input = torch.cat((reads_emb, ref_emb), dim=3)
                # Perform similar copy for empty read -- make sure we have position and reference
                empty_combo_input = torch.cat((empty_read_emb, ref_emb.narrow(2,0,1).narrow(0,0,1)), dim=3)
                if debug:
                    print('empty_combo_input')
                    print(empty_combo_input.shape)

        else:
            ref_emb = ref_emb.view(ref_emb.shape[0], ref_emb.shape[1], 1, -1)
            combo_input = torch.cat((reads_emb, ref_emb), dim=2)
            # Add positional embeddings
            if self.pos_embeddings:
                combo_input = combo_input + pos_ref

        # Concat other per-read channel inputs

        # Q(uality) scores
        if self.use_q_scores:
            # Normalize and contact Q-scores (log scale from 0 to 100 (50 max in practice))
            q_scores = q_scores.float() * Q_SCORE_SCALE_FACTOR
            q_scores = q_scores.unsqueeze(3)
            q_scores_empty = torch.zeros((empty_combo_input.shape[0], q_scores.shape[1], 1, q_scores.shape[3])).cuda()
            if debug:
                print('Q-scores')
                print(q_scores.shape)
                print(combo_input.shape)
            combo_input = torch.cat((combo_input, q_scores), dim=3)
            empty_combo_input = torch.cat((empty_combo_input, q_scores_empty), dim=3)
            if debug:
                print(combo_input.shape)

        # Strand (direction of single read)
        if self.use_strands:
            # Normalize and concant strand (0/1/2 enum)
            strands = strands.float() * STRAND_ENCODE_FACTOR
            strands = strands.unsqueeze(3)
            strands_empty = torch.zeros((empty_combo_input.shape[0], strands.shape[1], 1, strands.shape[3])).cuda()
            if debug:
                print('Strand')
                print(strands.shape)
                print(combo_input.shape)
            combo_input = torch.cat((combo_input, strands), dim=3)
            empty_combo_input = torch.cat((empty_combo_input, strands_empty), dim=3)
            if debug:
                print(combo_input.shape)

        # For empty read, neither reference nor variant will match
        if self.use_reads_ref_var_mask:
            var_mask_empty = torch.zeros((empty_combo_input.shape[0], empty_combo_input.shape[1], 1, 3)).cuda()
            empty_combo_input = torch.cat((empty_combo_input, var_mask_empty), dim=3)
            # squeeze may be necessary depending on how we do assigmnent (replacement)
            empty_combo_input = empty_combo_input.squeeze()
            if debug:
                print('empty_combo_input -- final dimension expanded')
                print(empty_combo_input.shape)

        # Mask match -- for reference? for proposed variant
        # A: Match which reads agree with ref (or var) -- ignore zero-values but match all others, exactly
        # B. If match, post '1' at every position for the mask
        if self.use_reads_ref_var_mask:
            # Add *mask* simply showing *length* of the variant/reference -- could help with partial matches (longer variants)?
            ref_masks = ref_masks.unsqueeze(2).expand(reads.shape)
            bin_mask = (~(ref_masks == 0)).long()
            if debug:
                print('Encoding binary mask for ref & var length:')
                print(bin_mask.shape)
                print(bin_mask.sum(dim=2).sum(dim=1))
            var_length_mask = bin_mask.unsqueeze(3).float()
            # Reference first
            if debug:
                print('Expanding ref & variant masks...')
                print(ref_masks.shape)
                print(reads.shape)
            #ref_masks = ref_masks.unsqueeze(2).expand(reads.shape)
            bin_mask = (~(ref_masks == 0)).long()
            bin_agree = torch.eq(reads*bin_mask, ref_masks)
            bin_agree = bin_agree.sum(dim=1) == reads.shape[1]
            if debug:
                print('Counting agreement per row (with reference)')
                print(bin_agree.sum(dim=1))
                print(bin_agree.shape)
            # Now -- take the bin_mask -- and multiply by binary match with reads
            ref_match_mask = bin_mask * bin_agree.long().unsqueeze(1)
            if debug:
                print('multiplying mask by rows that matched')
                print(ref_match_mask.shape)
                print(ref_match_mask.sum(dim=1).sum(dim=1))
            # Same with variant
            var_masks = var_masks.unsqueeze(2).expand(reads.shape)
            bin_mask = (~(var_masks == 0)).long()
            bin_agree = torch.eq(reads*bin_mask, var_masks)
            bin_agree = bin_agree.sum(dim=1) == reads.shape[1]
            if debug:
                print('Counting agreement per row (with variant)')
                print(bin_agree.sum(dim=1))
                # Example of where these indices come from
                print(bin_agree[0])
                print(bin_agree.shape)
            # Now -- take the bin_mask -- and multiply by binary match with reads
            var_match_mask = bin_mask * bin_agree.long().unsqueeze(1)
            if debug:
                print('multiplying mask by rows that matched')
                print(var_match_mask.shape)
                print(var_match_mask.sum(dim=1).sum(dim=1))
            # Append final product to the combo_input
            ref_match_mask = ref_match_mask.unsqueeze(3).float()
            var_match_mask = var_match_mask.unsqueeze(3).float()
            # NOTE: Could append *second* variant here -- to evaluate multi-allele directly...
            combo_input = torch.cat((combo_input, ref_match_mask, var_match_mask, var_length_mask), dim=3)
            if debug:
                print(combo_input.shape)

        # -------------------------------------
        # Now -- hack -- use indices to compute ID of rows to delete in support, or non-support of the variant

        # Remove up to X reads supporting the variant
        while rm_var_reads > 0:
            rm_var_reads -= 1
            # Get indices of non-zero elements -- ID of rows that *match* the variant
            var_agree_indices = torch.nonzero(bin_agree)
            if debug:
                print('Indices of elements agreeing with variant')
                print(var_agree_indices.shape)

            # Shuffle agreement indices -- for random sampling below
            perm = torch.randperm(var_agree_indices.shape[0])
            random_var_agree_indices = var_agree_indices[perm, :]

            # Use "unique" to return the first element ^^ at every batch position
            unique_rows, unique_row_index = torch.unique(torch.narrow(random_var_agree_indices,1,0,1), return_inverse=True)

            # Loop over candidate indices -- keep track of which batches can be modified, and take at most 1 of each
            if debug:
                print('Found matching var matches in %s rows' % unique_rows.shape[0])
            first_match_indices = []
            for i in range(unique_rows.shape[0]):
                first_match_idx = (unique_row_index == i).nonzero()[0,0]
                first_match_idx = random_var_agree_indices[first_match_idx,:]
                first_match_indices.append(first_match_idx)
            # Now we have a list of reads (at most one per batch example -- that can be removed to support fewer variant)
            if debug:
                print('zeroing out %d indices agreeing with variant (one per example)' % len(first_match_indices))
            for i in range(len(first_match_indices)):
                # zero out selected row, by index
                match_index = first_match_indices[i]
                if debug:
                    print('Zeroing out read index %s' % str(match_index))
                    #print(combo_input[match_index[0], 100, match_index[1], 42:])
                combo_input[match_index[0], :, match_index[1], :] = empty_combo_input

        # Same loop if we want to remove non-variant matching reads
        while rm_non_var_reads > 0:
            rm_non_var_reads -= 1
            # For negative var matches (to downsample) -- first mask by
            # A. *has* a read at pos 100
            # B. not variant
            bin_has_read = (reads[:,100,:] != 0)
            bin_has_read_not = (bin_has_read & (~bin_agree))
            if debug:
                print('Looking for reads with non-zero read at 100')
                print(bin_has_read.sum(dim=1))
                print(bin_has_read.shape)
                print('compare with bin agree')
                print(bin_agree[0])
                print(bin_agree.shape)
                print(bin_has_read_not[0])
                print(bin_has_read_not.shape)
                print(bin_has_read_not.sum(dim=1))

            # Now expand, index, shuffle and loop like for matching var reads
            var_agree_indices = torch.nonzero(bin_has_read_not)
            if debug:
                print('Indices of elements agreeing with variant')
                print(var_agree_indices.shape)

            # Shuffle agreement indices -- for random sampling below
            perm = torch.randperm(var_agree_indices.shape[0])
            random_var_agree_indices = var_agree_indices[perm, :]

            # Use "unique" to return the first element ^^ at every batch position
            unique_rows, unique_row_index = torch.unique(torch.narrow(random_var_agree_indices,1,0,1), return_inverse=True)

            # Loop over candidate indices -- keep track of which batches can be modified, and take at most 1 of each
            if debug:
                print('Found matching var matches in %s rows' % unique_rows.shape[0])
            first_match_not_var_indices = []
            for i in range(unique_rows.shape[0]):
                first_match_idx = (unique_row_index == i).nonzero()[0,0]
                first_match_idx = random_var_agree_indices[first_match_idx,:]
                first_match_not_var_indices.append(first_match_idx)
            # Now we have a list of reads (at most one per batch example -- that can be removed to support fewer variant)
            if debug:
                print('zeroing out %d indices *not* agree with variant (one per example)' % len(first_match_not_var_indices))
            for i in range(len(first_match_not_var_indices)):
                # zero out selected row, by index
                match_index = first_match_not_var_indices[i]
                if debug:
                    print('Zeroing out read index %s' % str(match_index))
                    #print(combo_input[match_index[0], 100, match_index[1], 42:])
                combo_input[match_index[0], :, match_index[1], :] = empty_combo_input

        # Dimensions [(batch), 201, 101, 20] -> [(batch), channels, reads, read_len]
        combo_input = combo_input.transpose(1,3)
        if debug:
            print('conv input shape')
            print(combo_input.shape)

        # Iterate over X layers, of conv, batch norm, and optional highway
        conv_out_1D = combo_input
        conv_1D_outputs = []
        highway_outputs = []
        for l_num in range(1,self.total_conv_layers+1):
            if debug:
                print('Pool layers -- %s' % str(self.conv_1d_pool_layers))
                print('---- Conv Round %d ----' % l_num)
            residual = conv_out_1D
            # Append previous round's average values -- if requested
            if (l_num-1) in self.conv_1d_pool_layers:
                if debug:
                    print('transforming and combining prev layer outputs')
                conv_out_pool = conv_out_pool.expand(conv_out_1D.shape)
                # Append pool average
                if self.conv_1d_pool_append:
                    conv_out_1D = torch.cat((conv_out_1D, conv_out_pool), dim=1)
                elif self.conv_1d_pool_add:
                    conv_out_1D = conv_out_1D + conv_out_pool
                else:
                    assert False, "Require conv_1d_pool_append or conv_1d_pool_add for appending intermediateAvePool1D"
                if debug:
                    print('combining (concat/add) 1D conv with average pool (across reads)')
                    print(conv_out_1D.shape)
            # Perform conv, BN, and highway (upon request)
            conv_out_1D = F.relu(self.conv1D_layers[l_num-1](conv_out_1D))
            if self.use_batchnorm:
                conv_out_1D = self.bn1D_layers[l_num-1](conv_out_1D)
            # Residual, if requested
            if self.is_residual_layer[l_num-1]:
                if debug:
                    print('Residual connection')
                # In the case of residual connections -- do final 1x1 layer *after* non-linearity
                # Why? Make it easier to turn output off...
                if debug:
                    print('Applying residual 1x1 number %d' % int(l_num -self.residual_layer_start))
                conv_out_1D = self.residual_conv_layers[int(l_num - self.residual_layer_start)](conv_out_1D)
                conv_out_1D += residual
            # Save ouput -- in case we need it for intermediate losses
            # NOTE: *keep* the 1x1 if residual -- think of it as a weighting/dimension switch operation
            conv_1D_outputs.append(conv_out_1D)
            # Do we pool this layer, for next round +=?
            if (self.conv_1d_pool_append or self.conv_1d_pool_add) and l_num in self.conv_1d_pool_layers:
                if debug:
                    print('computing all-dimensions pool for layer %d' % l_num)
                pooling_index = self.add_pooling_layer[l_num-1]
                if debug:
                    print('pooling layer index %d' % pooling_index)
                conv_out_pool = self.inter_ave_pool1D_layers[pooling_index](conv_out_1D)
            if self.append_bottleneck_highway_reads:
                hw_out = F.relu(self.conv1D_bottleneck_layers[l_num-1](conv_out_1D))
                # *Remove* nonlinearity after final HW output -- we will add nonlinearity after *averaging* HW outputs
                hw_out = self.conv1D_compression_layers[l_num-1](hw_out).squeeze(dim=3)
                hw_out = hw_out.view(hw_out.size(0), -1)
                highway_outputs.append(hw_out)

        # Transformer after Conv layers
        if self.use_transformer:
            if debug:
                print('Using Transformer %d layers' % self.num_transformer_layers)
            transformer_out = conv_out_1D
            # Conv out: {batch, dims, num_rows, seq_len}
            # Transformer requires: {seq_len, batch, dims} [combine batch*rows]
            transformer_out = transformer_out.permute(3,0,2,1).contiguous()
            if debug:
                print(transformer_out.shape)
            transformer_out = transformer_out.view((transformer_out.shape[0],-1,transformer_out.shape[-1]))
            if debug:
                print(transformer_out.shape)
            for i in range(1,self.num_transformer_layers+1):
                if debug:
                    print('Attempting Transformer layer %d' % i)
                    print(transformer_out.shape)
                residual = transformer_out
                transformer_out = self.transformer_encoder_layers[i-1](transformer_out)
                transformer_out = self.transformer_layer_norms[i-1](transformer_out)
                if self.transformer_residual:
                    if debug:
                        print('adding residual connection')
                    transformer_out += residual
            if debug:
                print('Final Transformer output:')
                print(transformer_out.shape)
            # Now decode back to conv dimensions.
            # input: {len, batch, dim}
            # output: {batch, dim, num_reads, dim}
            transformer_out = transformer_out.view((transformer_out.shape[0], conv_out_1D.shape[0], -1, transformer_out.shape[-1]))
            transformer_out = transformer_out.permute(1,3,2,0).contiguous()
            if debug:
                print(transformer_out.shape)
            # Final layer to reduce dimension out of the transformer (save params, need to do after because fo resnet)
            transformer_out = self.transformer_dim_reduction_layer(transformer_out)
            if debug:
                print('Reducing dimension before entering pooling network:')
                print(transformer_out.shape)
            conv_out_1D = transformer_out

        # Calculate mean and max pool, across all dimensions.
        if debug:
            print('---- Pooling Round ----')
        if not self.skip_final_maxpool:
            conv_out_max_1D = self.maxPool1D(conv_out_1D)
        conv_out_avg_1D = self.avgPool1D(conv_out_1D)
        if debug:
            print('1D conv pools')
            if not self.skip_final_maxpool:
                print(conv_out_max_1D.shape)
            print(conv_out_avg_1D.shape)
        if not self.skip_final_maxpool:
            conv_out_1D = torch.cat((conv_out_max_1D, conv_out_avg_1D), dim = 1)
        else:
            conv_out_1D = conv_out_avg_1D

        # Flatten pooled 1D outputs.
        xc = conv_out_1D
        xc = xc.view(xc.size(0), -1)
        # Apply final 1x1 after the pooling
        if self.pool_combine_dimension > 0:
            xc = self.post_pool_conv1D(xc)
            xc = F.relu(xc)
        hidden = xc

        # Append highway small-dimension embedding directly from each read [at low and multiple layers]
        # NOTE: Why? Make it easy to focus on the key region, to pass allele counts, variation bases
        if self.append_bottleneck_highway_reads:
            if debug:
                print('Appending bottleneck highway reads -- small D info for each individual read')
                print(highway_outputs[0].shape)
            # Average, or concat, HW outputs:
            if self.concat_hw_reads:
                hw_hiddens = torch.cat(highway_outputs, dim=1)
            else:
                hw_hiddens = sum(highway_outputs)
                hw_hiddens /= len(highway_outputs)
            # Add non-linearity, after the HW sums (since we remove ReLU before the sum)
            hw_hiddens = F.relu(hw_hiddens)
            if debug:
                print(hw_hiddens.shape)

        # Similarly -- calculate "early" conv ouputs -- earlier Conv1D outputs, and varying number of highways
        early_bin_outputs = []
        early_vt_outputs = []
        for i,l in enumerate(self.early_loss_layers):
            # Get correct layer output, run it through the pools
            inter_conv_out = conv_1D_outputs[l-1]
            if not self.skip_final_maxpool:
                inter_conv_max = self.maxPool1DEarly[i](inter_conv_out)
            inter_conv_avg = self.avgPool1DEarly[i](inter_conv_out)
            if not self.skip_final_maxpool:
                inter_conv_out = torch.cat((inter_conv_max, inter_conv_avg), dim=1)
            else:
                inter_conv_out = inter_conv_avg
            early_hidden = inter_conv_out.view(inter_conv_out.size(0), -1)
            if self.pool_combine_dimension > 0:
                early_hidden = self.post_pool_conv1D_early[i](early_hidden)
                early_hidden = F.relu(early_hidden)

            # Highway, if used
            if self.append_bottleneck_highway_reads:
                # Average, or cat the highways
                if debug:
                    print('Early-summing HW layers [0:%d]' % l)
                if self.concat_hw_reads:
                    hw_hiddens_early = torch.cat(highway_outputs[:l], dim=1)
                else:
                    hw_hiddens_early = sum(highway_outputs[:l])
                    hw_hiddens_early /= l
                # Take the non-linearity here, after summing the HW outputs (before FCN)
                hw_hiddens_early = F.relu(hw_hiddens_early)
                early_hidden = torch.cat((early_hidden, hw_hiddens_early), dim=1)
            # FCN
            early_hidden = self.conv2hidden_early[i](early_hidden)
            # Two basic outputs -- binary and VT
            early_bin_out = self.fcHidden2Bin_early[i](early_hidden)
            early_vt_out = self.fcHidden2VT_early[i](early_hidden)
            early_bin_outputs.append(early_bin_out)
            early_vt_outputs.append(early_vt_out)

        # TODO: Proper WaveNet-like Resnet:
        # A. Sum in all dimensions (hidden)
        # B. Take ReLU of the sum
        # C. Concat highway outputs (already ReLU)
        # D. *this* goes into the single FCN...
        # NOTE: Early attempt do not work -- diverges. For now, keep returning early output loss, separately.
        if debug:
            print('Final combine and ReLU -- for 1x1 outputs from pooling -- and also the (averaged) highways')
            print(hidden.shape)
        if self.append_bottleneck_highway_reads:
            hidden = torch.cat((hidden, hw_hiddens), dim=1)
        if debug:
            print(hidden.shape)

        # Fully connected network, on flattened outputs
        hidden = self.conv2hidden(hidden)
        # output as logits -- for SoftmaxWithLogits
        xbinary = self.fcHidden2BinTarget(hidden)
        # VT (Variant type) output
        xVT = self.fcHidden2VT(hidden)

        # *if* learning context-based loss balance,
        # A. compute softmax (between losses)
        # B. take softmax of logits
        # C. combine weighted softmax
        # D. output final *predictions* plus softmax weights for debugging/viz
        if self.learn_context_early_loss_balance and len(self.early_loss_layers) > 0:
            bin_layers_softmax = F.softmax(self.binary_context_loss_balance(hidden), dim=1)
            vt_layers_softmax = F.softmax(self.vt_context_loss_balance(hidden), dim=1)
            if debug:
                print('calculating context weighted softmax')
                print(loss_layers_softmax.shape)
                print(vt_layers_softmax.shape)
            # Binary outputs
            bin_outputs = torch.stack(early_bin_outputs + [xbinary]).cuda()
            soft_bin_outputs = F.softmax(bin_outputs, dim=2).transpose(0,1)
            soft_bin_outputs = soft_bin_outputs * bin_layers_softmax.unsqueeze(2).expand(soft_bin_outputs.shape)
            soft_bin_outputs = torch.sum(soft_bin_outputs, dim=1)
            # VT outputs
            vt_outputs = torch.stack(early_vt_outputs + [xVT]).cuda()
            soft_vt_outputs = F.softmax(vt_outputs, dim=2).transpose(0,1)
            soft_vt_outputs = soft_vt_outputs * vt_layers_softmax.unsqueeze(2).expand(soft_vt_outputs.shape)
            soft_vt_outputs = torch.sum(soft_vt_outputs, dim=1)
            if debug:
                print(soft_bin_outputs.shape)
                print(soft_vt_outputs.shape)
                print('------------------------------')
        else:
            bin_layers_softmax, vt_layers_softmax, soft_bin_outputs, soft_vt_outputs = None, None, None, None

        # Auxillary loss predictions
        xAF = self.fcHidden2AF(hidden)
        xAF = torch.sigmoid(xAF)
        xCov = self.fcHidden2Coverage(hidden)
        xCov = F.leaky_relu(xCov)
        xVB = self.fcHidden2VB(hidden)
        xVR = self.fcHidden2VR(hidden)
        return (xbinary, xVT, xAF, xCov, xVB, xVR, early_bin_outputs, early_vt_outputs,
            self.bin_output_weights, self.vt_output_weights,
            bin_layers_softmax, vt_layers_softmax, soft_bin_outputs, soft_vt_outputs)

