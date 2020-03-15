# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

import time
import pysam
import os
import math
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing
import numpy as np
import pandas as pd
# For label smoothing & other custom loss functions
from dl4vc.objectives import SoftBCEWithLogitsLoss, SoftBCEWithLogitsFocalLoss
from dl4vc.utils import initialize_vcf_tempfile, append_vcf_records
from sklearn.metrics import confusion_matrix, auc, roc_curve, precision_recall_curve

# Update data loader with close examples
# Optionally -- also write up to MAX hard examples to table.
# TODO: Add info to save, other than VCFrec? (combines location and variant info)
MAX_HARD_RECS = 100000
HARD_RECS_DTYPE = np.dtype([('vcfrec', np.string_, 128)])
def update_close_matches(train_dataset, idxs, scores, recs=[], hard_recs=None, hard_recs_count=0):
    #print('updating close matches for %d examples' % len(idxs))
    for i, b in enumerate(zip(idxs, scores)):
        if b[1]:
            #print(b)
            train_dataset.update_close_example(b[0], True)
        else:
            # TODO: Do we need to update negative examples?
            train_dataset.update_close_example(b[0], False)
            if len(hard_recs)>0 and hard_recs_count < MAX_HARD_RECS:
                #print(b)
                #print(recs[i])
                hard_recs[hard_recs_count]['vcfrec'] = recs[i]
                hard_recs_count += 1
    #print('After update, train_loader has %d close examples' % train_dataset.count_close_examples())
    return hard_recs_count

def get_learning_rate(optimizer):
    """
    Return the learning rate(s) of an optimizer
    """
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr

# Just update bad examples -- so we don't keep visiting them
def update_blacklist(train_dataset, idxs, blacklist):
    #print('blacklist')
    for i, b in enumerate(zip(idxs, blacklist)):
        #print((i,b))
        if b[1]:
            print('adding blacklist')
            print((i,b))
            train_dataset.update_blacklist(b[0], True)

COVERAGE_SCALE_FACTOR = 1./100.
#BINARY_DELTA_LOSS_WEIGHT = 1.0
#AUGMENTED_EXAMPLE_WEIGHT = 0.2
def train(args, model, device, train_loader, optimizer, epoch, train_dataset=None, debug=True,
    trust_starts={}, trust_ends={}, non_trust_train_weight=1.0):
    """
    Train the model
    """
    model.train()
    print('Train Epoch: ' + str(epoch) + " lr: "+ str(get_learning_rate(optimizer)) +" on "+ str(torch.cuda.device_count()) + " GPUs!")
    a = time.perf_counter()
    time_zero = time.time()
    total_loss, total_bin_loss, total_vt_loss, total_delta_loss, total_af_loss, total_cov_loss, total_base_loss = 0,0,0,0,0,0,0

    # Count how many "close matches" we encounter during epoch of training
    # TODO: Save "close match" IDs so we could down-sample them for training (like RL replay)
    total_close_matches = 0
    total_delta_examples, total_vt_delta_examples = 0,0
    total_items = 0

    # Criteria, with (optional) label smoothing
    bin_loss_criteria = SoftBCEWithLogitsLoss(label_smoothing=args.label_smoothing,
        num_classes=2, close_match_window=args.close_match_window,
        pos_weight=torch.Tensor([args.fp_train_weight, 1.]).cuda())
    vt_loss_criteria = SoftBCEWithLogitsLoss(label_smoothing=args.label_smoothing,
        num_classes=3, close_match_window=args.close_match_window,
        pos_weight=torch.Tensor([args.fp_train_weight,1.,1.]).cuda())
    # Focal loss will only be used if gamma > 0. [else same as normal SoftBCE criteria]
    bin_focal_loss_criteria = SoftBCEWithLogitsFocalLoss(label_smoothing=args.label_smoothing,
        num_classes=2, close_match_window=args.close_match_window,
        pos_weight=torch.Tensor([args.fp_train_weight, 1.]).cuda(),
        alpha=args.focal_loss_alpha, gamma=args.focal_loss_gamma)
    vt_focal_loss_criteria = SoftBCEWithLogitsFocalLoss(label_smoothing=args.label_smoothing,
        num_classes=3, close_match_window=args.close_match_window,
        pos_weight=torch.Tensor([args.fp_train_weight,1.,1.]).cuda(),
        alpha=args.focal_loss_alpha, gamma=args.focal_loss_gamma)

    assert args.focal_loss_gamma >= 0., 'Illegal focal loss gamma %s' % args.focal_loss_gamma
    if args.focal_loss_gamma > 0.:
        print('Training with focal loss -- hold your horses. Alpha %.4f; gamma %.4f' % (args.focal_loss_alpha, args.focal_loss_gamma))

    # Track hard examples -- enough details to know location & variant
    if args.save_hard_example_records:
        hard_recs = np.empty(MAX_HARD_RECS, dtype=HARD_RECS_DTYPE)
    else:
        hard_recs = []
    hard_recs_count = 0

    for batch, items in enumerate(train_loader):
        if debug and batch == 0:
            print('Time to load training dataset %.5fs' % (time.time() - time_zero))

        if args.max_train_batches > 0 and batch > args.max_train_batches:
            print('Exit -- train %d batch' % args.max_train_batches)
            break

        # Open up VCF -- examine mutation types (SNP, Insert, Delete)
        is_snp_tensor = items["is_snp"].float()
        reads = items["reads"].long()
        ref = items["ref"].long()
        q_scores = items["q-scores"].long()
        strands = items["strands"].long()
        # Aux *inputs* for multi-allele -- ref and var bases (so same pileup, differnt scores)
        ref_bases = items["var_ref_vector"].long()
        var_bases = items["var_base_vector"].long()
        ref_masks = items["ref_mask"].long()
        var_masks = items["var_mask"].long()
        # Keep IDs of items. Why? So we can track easy examples, and possibly skip them in next epoch
        idxs = items['idx'].long()
        blacklist = items['blacklist']
        # We want .cuda() on targets
        target = items["label"].long().cuda()
        # Binary target -- assume that labels {0: TP, 1:FN, 2:FP}
        target_binary = (target<=1).long().unsqueeze(1)
        # Auxillary targets...
        # Softmax 0,1,2 --> none, homo, heterozygous
        target_var_type = items["var_type"].long().cuda().unsqueeze(1)
        # 0. to 1.
        target_allele_freq = items["allele_freq"].float().unsqueeze(1)
        # Number of reads at target position?
        target_coverage = items["coverage"].float().cuda().unsqueeze(1) * COVERAGE_SCALE_FACTOR
        # Softmax over bases (ATGC) + Insert/delete
        target_var_base_enum = items["var_base_enum"].long().cuda()
        target_var_ref_enum = items["var_ref_enum"].long().cuda()

        # Text description of variant -- useful for debug -- or saving "hard examples" to file
        vcf_records = items['vcfrec']

        # Example weighting [based on trust/non trust, etc]
        # TODO -- lookup trust regions in data pre-processing?
        binary_trust_weight = torch.ones(target_binary.shape, dtype=torch.float, requires_grad=False)
        if trust_starts:
            ids = items["name"]
            for i, n in enumerate(ids):
                name = 'HG001:' + n
                chrom, idx = n.split(':')
                # HACK: Handle "chr1", and just "1"
                # TODO: More general "grab integer after string" -- but regexp is slower
                if chrom.startswith('chr') or chrom.startswith('Chr'):
                    chrom = chrom[3:]
                idx = int(idx.strip())
                trust = is_in_region(chrom=chrom, loc=idx, start_locations=trust_starts, end_locations=trust_ends)
                if not trust:
                    binary_trust_weight[i] = non_trust_train_weight
        binary_trust_weight = binary_trust_weight.cuda(target.get_device())
        binary_trust_weight.requires_grad = False

        # Multiply by SNP/non-SNP weight -- in case we want to punt on Indels
        non_snp_weight = args.non_snp_train_weight
        batch_snp_weight = is_snp_tensor + (1. - is_snp_tensor) * non_snp_weight
        batch_snp_weight = batch_snp_weight.cuda(target.get_device()).unsqueeze(1)
        total_class_weight = batch_snp_weight * binary_trust_weight

        # HACK -- to experiment, just flip coins to decide in which direction to "enhance" reads
        rm_var_reads = 0
        rm_non_var_reads = 0
        # If training two batches -- directional step -- hardcode split rate at 50/50
        directional_augmentation = args.training_use_directional_augmentation and args.delay_augmentation_epochs < epoch
        if directional_augmentation:
            rm_var_reads_rate, rm_non_var_reads_rate = 0.5, 0.5
        else:
            rm_var_reads_rate = args.rm_var_reads_rate
            rm_non_var_reads_rate = args.rm_non_var_reads_rate
        # HACK: Modify one, or the other -- keep adding at exponential rate [allow steps > 1]
        if np.random.random() <= rm_var_reads_rate:
            rm_var_reads += 1
            while True:
                if np.random.random() <= rm_var_reads_rate and rm_var_reads < 10:
                    rm_var_reads += 1
                else:
                    break
        elif np.random.random() <= (rm_non_var_reads_rate) / (1. - rm_var_reads_rate):
            rm_non_var_reads += 1
            while True:
                if np.random.random() <= rm_non_var_reads_rate and rm_non_var_reads < 10:
                    rm_non_var_reads += 1
                else:
                    break

        # Supply vector for binary trust weights as input
        optimizer.zero_grad()

        # Try something new -- compute batch as-is. Then take a directional step,
        # compute that too, and see if probablities move in the expected direction?

        # Original model loss -- no augmentation
        if directional_augmentation or args.delay_augmentation_epochs >= epoch:
            rm_non_var_init, rm_var_init = 0,0
        else:
            rm_non_var_init, rm_var_init = rm_non_var_reads, rm_var_reads
        # Run the model -- either once [if no directional augmetation] or the first time (if directional augment)
        # loss_layers_softmax, soft_bin_outputs, soft_vt_outputs
        bin_out, vt_out, af_out, cov_out, vb_out, vr_out, early_bin_out, early_vt_out, bin_out_w, vt_out_w, bls, vtls, bs_out, vts_out = model(reads, ref,
            q_scores=q_scores, strands=strands, binary_trust_vector=binary_trust_weight,
            af_scores=target_allele_freq, ref_bases=ref_bases, var_bases=var_bases,
            ref_masks=ref_masks, var_masks=var_masks,
            rm_non_var_reads=rm_non_var_init, rm_var_reads=rm_var_init)

        # TODO: Put loss computation in a function. Especially if we call same loss for data augmented examples
        # Try soft version of BCE loss
        binary_loss, bin_close_items = bin_focal_loss_criteria(bin_out, target_binary, weight=total_class_weight)
        # auxillary losses
        # Soft version of BCE loss
        vt_loss, vt_close_items  = vt_focal_loss_criteria(vt_out, target_var_type, weight=total_class_weight)

        # If applicable, calculate "early loss" for Bin and VT
        if len(args.early_loss_layers) > 0:
            soft_bin_loss, bin_close_items = bin_focal_loss_criteria(bs_out, target_binary, weight=total_class_weight, logits=False)
            soft_vt_loss, vt_close_items = vt_focal_loss_criteria(vts_out, target_var_type, weight=total_class_weight, logits=False)

            # Also take tiny loss for all output layers... else could be unstable as no gradient.
            bin_early_loss = [bin_focal_loss_criteria(early_bin_out[i], target_binary, weight=total_class_weight)[0] for i in range(len(args.early_loss_layers))]
            #bin_early_loss = sum(bin_early_loss)
            vt_early_loss = [vt_focal_loss_criteria(early_vt_out[i], target_var_type, weight=total_class_weight)[0] for i in range(len(args.early_loss_layers))]
            final_bin_loss, _ = bin_focal_loss_criteria(bin_out, target_binary, weight=total_class_weight)
            final_vt_loss, _ = vt_focal_loss_criteria(vt_out, target_var_type, weight=total_class_weight)

            # losses from learned mixing
            binary_loss = soft_bin_loss
            vt_loss = soft_vt_loss

            # plus tiny gradient from each layer
            binary_loss += args.layer_loss_weight * (sum(bin_early_loss) + binary_loss)
            vt_loss += args.layer_loss_weight * (sum(vt_early_loss) + vt_loss)

            # For debug, etc
            bin_out = bs_out
            vt_out = vts_out
        else:
            # TODO: Put loss computation in a function. Especially if we call same loss for data augmented examples
            # Try soft version of BCE loss
            binary_loss, bin_close_items = bin_focal_loss_criteria(bin_out, target_binary, weight=total_class_weight)
            # auxillary losses
            # Soft version of BCE loss
            vt_loss, vt_close_items  = vt_focal_loss_criteria(vt_out, target_var_type, weight=total_class_weight)

        # Update counter
        total_close_matches += torch.sum(vt_close_items).detach().cpu().numpy()
        total_items += vt_close_items.shape[0]

        # HACK -- update "is this example close (easy)?" table in the *data loader*
        # This may then be used in the next epoch of training.
        hard_recs_count = update_close_matches(train_dataset, idxs=idxs.cpu().numpy(),
            scores=vt_close_items.detach().cpu().numpy(), recs=vcf_records, hard_recs=hard_recs, hard_recs_count=hard_recs_count)

        # Also update the blacklist, if any
        update_blacklist(train_dataset, idxs=idxs.cpu().numpy(), blacklist=blacklist.cpu().numpy())

        """
        # Depracated -- forced weight intermediate loss criteria not working.
        if False:
            bin_early_loss = [bin_focal_loss_criteria(early_bin_out[i], target_binary, weight=total_class_weight)[0] for i in range(len(args.early_loss_layers))]
            #bin_early_loss = sum(bin_early_loss)
            vt_early_loss = [vt_focal_loss_criteria(early_vt_out[i], target_var_type, weight=total_class_weight)[0] for i in range(len(args.early_loss_layers))]
            #vt_early_loss = sum(vt_early_loss)

            # HACK -- weight the loss -- according to output weight
            num_losses = len(args.early_loss_layers) + 1
            if args.learn_early_loss_weight:
                # NOTE: We get the loss weight parameter, repeated by number of GPUs :-(
                bin_out_w = bin_out_w.view(-1,num_losses)
                vt_out_w = vt_out_w.view(-1,num_losses)
                bin_out_w = F.softmax(torch.mean(bin_out_w, dim=0))
                vt_out_w = F.softmax(torch.mean(vt_out_w, dim=0))
            else:
                bin_out_w = torch.tensor([args.early_loss_weight for i in args.early_loss_layers] + [1.0]).cuda()
                bin_out_w = bin_out_w / torch.sum(bin_out_w)
                vt_out_w = torch.tensor([args.early_loss_weight for i in args.early_loss_layers] + [1.0]).cuda()
                vt_out_w = vt_out_w / torch.sum(vt_out_w)
            bin_early_loss.append(binary_loss)
            vt_early_loss.append(vt_loss)
            bin_early_loss = torch.stack(bin_early_loss)
            vt_early_loss = torch.stack(vt_early_loss)
            weighted_bin_loss = bin_early_loss * bin_out_w
            weighted_vt_loss = vt_early_loss * vt_out_w
            weighted_bin_loss = torch.mean(weighted_bin_loss)
            weighted_vt_loss = torch.mean(weighted_vt_loss)

            binary_loss = weighted_bin_loss
            vt_loss = weighted_vt_loss

            # weighted averge of intermediate & final loss
            # TODO: Adjust this ratio over time -- like 64 layer transformer [early focus on shallower net, then deeper]
            #binary_loss = (binary_loss * 1.0 + bin_early_loss * args.early_loss_weight) / (1.0 + args.early_loss_weight * len(args.early_loss_layers))
            #vt_loss = (vt_loss * 1.0 + vt_early_loss * args.early_loss_weight) / (1.0 + args.early_loss_weight * len(args.early_loss_layers))
            """

        # Don't use AF as loss, if also as input?
        af_loss = F.binary_cross_entropy(af_out, target_allele_freq.cuda(), weight=total_class_weight)
        cov_loss = F.mse_loss(cov_out, target_coverage)
        # NOTE: For predicting bases -- focus on bases that are possible in the softmax
        vb_loss = F.cross_entropy(vb_out, target_var_base_enum, weight=torch.Tensor([0.001,1.,1.,1.,1.,1.,0.001,0.001,1.,0.001]).cuda(target.get_device()))
        vr_loss = F.cross_entropy(vr_out, target_var_ref_enum, weight=torch.Tensor([0.001,1.,1.,1.,1.,1.,0.001,0.001,1.,0.001]).cuda(target.get_device()))

        # Will we do example debug in this step?
        if args.loss_debug_freq > 0 and args.auxillary_loss_weight > 0. and (batch % args.loss_debug_freq == 0):
            debug_step=True
        else:
            debug_step=False

        """
        # Deprecated -- removing directional augmetnation as it's not working
        # Directional augmentation step
        if False:
            bin_out_mod, vt_out_mod, af_out_mod, cov_out_mod, vb_out_mod, vr_out_mod, _, _, bin_out_w, vt_out_w = model(reads, ref,
                q_scores=q_scores, strands=strands, binary_trust_vector=binary_trust_weight,
                af_scores=target_allele_freq, ref_bases=ref_bases, var_bases=var_bases,
                ref_masks=ref_masks, var_masks=var_masks,
                rm_non_var_reads=rm_non_var_reads, rm_var_reads=rm_var_reads)

            # Ignore aux losses -- we only care about real ones (binary & VT) for augmented step
            binary_loss_mod, _ = bin_focal_loss_criteria(bin_out_mod, target_binary, weight=total_class_weight)
            vt_loss_mod, _  = vt_focal_loss_criteria(vt_out_mod, target_var_type, weight=total_class_weight)

            # Compare binary classification -- and compute a ranking loss
            if debug_step:
                print('-------------\nbinary classification -- normal, and with augmentation')
                print('(rm_var_reads, rm_non_var_reads)')
                print((rm_var_reads, rm_non_var_reads))

            delta = F.softmax(bin_out_mod, dim=1) - F.softmax(bin_out, dim=1)
            if rm_var_reads > 0:
                delta *= -1
            clipped_delta = F.relu(delta.narrow(1,0,1))
            if debug_step:
                print(torch.cat((target_binary.float(),
                    F.softmax(bin_out, dim=1), F.softmax(bin_out_mod, dim=1),
                    delta, clipped_delta), dim=1)[0:5,:])
            binary_delta_loss = torch.mean(clipped_delta)

            # Count how many examples are OOL -- grace epsilon though since moodel has tiny bit of stochasticity
            num_delta_examples = torch.nonzero(F.relu(clipped_delta - args.label_smoothing)).size(0)
            total_delta_examples += num_delta_examples
            if debug_step:
                print('Examples > epsilon in wrong order? %s' % num_delta_examples)

            # Same for VT -- three part loss.
            delta = F.softmax(vt_out_mod, dim=1) - F.softmax(vt_out, dim=1)
            if rm_var_reads > 0:
                delta *= -1
            # NV (no variant) loss if P(NV) increases when we remove non-variant reads
            clipped_delta_nv = F.relu(delta.narrow(1,0,1))
            # OV (homozygous var) loss if P(OV) increases when we remove variant-supporting reads
            clipped_delta_ov = F.relu(-1*delta.narrow(1,2,1))
            # No direct loss for HV (heterozygous variant) -- indirect through pieces above
            clipped_delta = torch.cat((clipped_delta_nv, clipped_delta_ov), dim=1)
            clipped_delta = torch.sum(clipped_delta, dim=1)
            if debug_step:
                print(torch.cat((target_var_type.float(), F.softmax(vt_out, dim=1), F.softmax(vt_out_mod, dim=1),
                    clipped_delta_nv, clipped_delta_ov), dim=1)[0:5,:])
            vt_delta_loss = torch.mean(clipped_delta)

            # Count examples wrong in NV or OV dimension
            num_vt_delta_examples = torch.nonzero(F.relu(clipped_delta - args.label_smoothing)).size(0)
            total_vt_delta_examples += num_vt_delta_examples
            if debug_step:
                print('VT Examples > epsilon in wrong order? %s' % num_vt_delta_examples)
                """

        if debug_step:
            print('losses:')
            print([(n, l.item()) for (n,l) in zip(['bin_loss', 'vt', 'af', 'cov', 'vb', 'vr'], [binary_loss, vt_loss, af_loss, cov_loss, vb_loss, vr_loss])])
            # Debug an example
            print('binary prediction')
            print(torch.cat((bin_out, target_binary.float()), dim=1).detach().cpu().numpy()[0:3])
            #print(target_binary.cpu().numpy()[0:3])
            print('variant type')
            print(torch.cat((vt_out, target_var_type.float()), dim=1).detach().cpu().numpy()[0:3])
            #print(target_var_type.cpu().numpy()[0:3])
            print('allele freq')
            print(torch.cat((af_out, target_allele_freq.cuda()), dim=1).detach().cpu().numpy()[0:3])
            #print(target_allele_freq.cpu().numpy()[0:3])
            print('coverage')
            print(torch.cat((cov_out, target_coverage.cuda()), dim=1).detach().cpu().numpy()[0:3])
            if len(args.early_loss_layers) > 0:
                print('learned output weights')
                print(vtls.detach().cpu().numpy()[0:3])
                # What are the rows producing maximum activation across output heads?
                max_out_weight, max_out_indices = torch.max(vtls, dim=0)
                print('max output weights, indices, details')
                print(max_out_weight.detach().cpu().numpy())
                print(max_out_indices.detach().cpu().numpy())
                print(torch.cat((vt_out, target_var_type.float(), af_out, target_allele_freq.cuda()), dim=1)[max_out_indices].detach().cpu().numpy())
            #print(target_coverage.cpu().numpy()[0:3])
            print('%d/%d [%.2f%%] "close matches" within %.2f * %.5f (label smoothing) of true label' % (total_close_matches,
                total_items, (total_close_matches/max(total_items,1) * 100.), args.close_match_window, args.label_smoothing))
            print('%d/%d [%.2f%%] "wrong delta" examples where data augmentation goes in wrong direction' % (total_delta_examples,
                total_items, (total_delta_examples/max(total_items,1) * 100.)))
            print('%d/%d [%.2f%%] "VT wrong delta" examples where data augmentation goes in wrong direction' % (total_vt_delta_examples,
                total_items, (total_vt_delta_examples/max(total_items,1) * 100.)))
            if len(args.early_loss_layers) > 0 and False:
                print('Weight for early loss and final layers -- binary and VT:')
                print(bin_out_w)
                print(vt_out_w)
            # Make sure we get this progress
            print('\n', flush=True)

        # In directional augmentation case, average bin & VT losses -- and add delta loss
        if directional_augmentation:
            loss = (binary_loss + binary_loss_mod * args.augmented_example_weight)/(1.+args.augmented_example_weight) * args.binary_weight
            loss += ((vt_loss + vt_loss_mod*args.augmented_example_weight)/(1.+args.augmented_example_weight) +
                af_loss * args.auxillary_loss_allele_weight + cov_loss + (vb_loss + vr_loss) * args.auxillary_loss_bases_weight) * args.auxillary_loss_weight
            loss += (binary_delta_loss + vt_delta_loss) * args.delta_loss_weight
            total_delta_loss += binary_delta_loss.detach() + vt_delta_loss.detach()
        else:
            loss = (binary_loss) * args.binary_weight
            loss += ((vt_loss) + af_loss * args.auxillary_loss_allele_weight + cov_loss + (vb_loss + vr_loss) * args.auxillary_loss_bases_weight) * args.auxillary_loss_weight

        total_loss += loss.detach()
        total_bin_loss += binary_loss.detach()
        total_vt_loss += vt_loss.detach()
        total_af_loss += af_loss.detach()
        total_cov_loss += cov_loss.detach()
        total_base_loss += vb_loss.detach() + vr_loss.detach()
        loss.backward()
        # Gradient clipping -- useful to avoid blowup for deep networks
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if batch % args.log_interval == 0:
            loss_value = loss.item()
            torch.cuda.synchronize()
            print('  Elapsed ({:.02e}s) [{}/{} ({:.0f}%)]  Loss: {:.6f}  Total: {:.6f}| bin: {:.5f} vt: {:.5f} dlt: {:.5f} af: {:.5f} cov: {:.5f} bases: {:.5f}'.format(
                time.perf_counter()-a, batch * len(ref), len(train_loader.dataset),
                100. * batch / len(train_loader), loss_value, total_loss.item()/(batch+1),
                total_bin_loss.item()/(batch+1), total_vt_loss.item()/(batch+1),
                (total_delta_loss.item()/(batch+1) if directional_augmentation  else 0.), total_af_loss.item()/(batch+1),
                total_cov_loss.item()/(batch+1), total_base_loss.item()/(batch+1)), end='\r')
            a = time.perf_counter()
    # Make sure we don't delete last loss line above ^^ with next print output
    print('\n')
    # If saving hard examples -- snip and write "hard recs" file to disk
    if args.save_hard_example_records:
        filename = args.save_vcf_records_file
        dir = os.path.dirname(filename)
        basefile = os.path.basename(filename)
        temp_filename = next(tempfile._get_candidate_names()) + ('_epoch%s_' % epoch) + 'hard_recs'
        temp_filename = os.path.join(dir, temp_filename)
        print('Saving %d hard records to disk (%d max): %s' % (hard_recs_count, MAX_HARD_RECS, temp_filename))
        np.save(temp_filename, hard_recs[:hard_recs_count])
    print('%d/%d [%.2f%%] "close matches" within %.2f * %.5f (label smoothing) of true label' % (total_close_matches,
                total_items, (total_close_matches/max(total_items,1) * 100.), args.close_match_window, args.label_smoothing))
    print('%d/%d [%.2f%%] "wrong delta" examples where data augmentation goes in wrong direction' % (total_delta_examples,
                total_items, (total_delta_examples/max(total_items,1) * 100.)))
    print('%d/%d [%.2f%%] "VT wrong delta" examples where data augmentation goes in wrong direction' % (total_vt_delta_examples,
                total_items, (total_vt_delta_examples/max(total_items,1) * 100.)))
    if len(args.early_loss_layers) > 0 and False:
        print('Weight for early loss and final layers -- binary and VT:')
        print(bin_out_w)
        print(vt_out_w)
    # Make sure we get this progress
    print('\n', flush=True)

def test(args, model, device, test_loader, debug=False,
    gatk_table={}, trust_starts={}, trust_ends={}, non_trust_train_weight=1.0, epoch=1):
    model.eval()
    a = time.perf_counter()
    test_loss, total_bin_loss, total_vt_loss, total_af_loss, total_cov_loss, total_base_loss = 0,0,0,0,0,0
    correct, correct_binary = 0, 0
    time_zero = time.time()
    # Compute values for AUC -- assume that validation results can fit in memory
    bin_targets = []
    bin_results = []
    trust_targets = []
    trust_results =  []
    gatk_results = []
    gatk_trust_results = []
    if args.trust_snp_only:
        print('On Trust Region -- evaluating SNPs only!')
    # Upon request, initialize and save VCF
    if args.save_vcf_records:
        print('Saving generated VCF records -- step by step during inference')
        assert args.save_vcf_records_file != '', 'Need a valid filename for args.save_vcf_records_file to save records'
        # Initializes header for filename, to dump VCF records in every batch
        vcf_tempfile = initialize_vcf_tempfile(sample_vcf=args.sample_vcf, filename=args.save_vcf_records_file, epoch=epoch)
        print('Initialized VCF file -- will write records through inference %s' % vcf_tempfile)
    with torch.no_grad():
        # Criteria, with (optional) label smoothing
        bin_loss_criteria = SoftBCEWithLogitsLoss(label_smoothing=args.label_smoothing, num_classes=2, pos_weight=torch.Tensor([args.fp_train_weight, 1.]).cuda())
        vt_loss_criteria = SoftBCEWithLogitsLoss(label_smoothing=args.label_smoothing, num_classes=3, pos_weight=torch.Tensor([args.fp_train_weight,1.,1.]).cuda())
        # Focal loss will only be used if gamma > 0. [else same as normal SoftBCE criteria]
        bin_focal_loss_criteria = SoftBCEWithLogitsFocalLoss(label_smoothing=args.label_smoothing,
            num_classes=2, pos_weight=torch.Tensor([args.fp_train_weight, 1.]).cuda(),
            alpha=args.focal_loss_alpha, gamma=args.focal_loss_gamma)
        vt_focal_loss_criteria = SoftBCEWithLogitsFocalLoss(label_smoothing=args.label_smoothing,
            num_classes=3, pos_weight=torch.Tensor([args.fp_train_weight,1.,1.]).cuda(),
            alpha=args.focal_loss_alpha, gamma=args.focal_loss_gamma)

        for batch, items in enumerate(test_loader):
            if debug and batch == 0:
                print('Time to load test dataset %2.fs' % (time.time() - time_zero))

            if args.max_test_batches > 0 and batch > args.max_test_batches:
                print('Exit -- limit test %d batch' % args.max_test_batches)
                break

            # Open up VCF -- examine mutation types (SNP, Insert, Delete)
            vcf_records = items['vcfrec']
            is_snp_tensor = items["is_snp"].float()
            ref = items["ref"].long()
            reads = items["reads"].long()
            q_scores = items["q-scores"].long()
            strands = items["strands"].long()
            # Aux *inputs* for multi-allele -- ref and var bases (so same pileup, differnt scores)
            ref_bases = items["var_ref_vector"].long()
            var_bases = items["var_base_vector"].long()
            ref_masks = items["ref_mask"].long()
            var_masks = items["var_mask"].long()
            target = items["label"].long().cuda()
            # Binary target -- assume that labels {0: TP, 1:FN, 2:FP}
            target_binary = (target <= 1).long().unsqueeze(1)
            # Auxillary targets...
            # Softmax 0,1,2 --> none, homo, heterozygous
            target_var_type = items["var_type"].long().cuda().unsqueeze(1)
            # 0. to 1.
            target_allele_freq = items["allele_freq"].float().unsqueeze(1)
            # Number of reads at target position?
            target_coverage = items["coverage"].float().cuda().unsqueeze(1) * COVERAGE_SCALE_FACTOR
            # Softmax over bases (ATGC) + Insert/delete
            target_var_base_enum = items["var_base_enum"].long().cuda()
            target_var_ref_enum = items["var_ref_enum"].long().cuda()

            # Example weighting [based on trust/non trust, etc]
            # TODO -- lookup trust regions in data pre-processing?
            binary_trust_weight = torch.ones(target_binary.shape, dtype=torch.float, requires_grad=False)
            if trust_starts:
                ids = items["name"]
                for i, n in enumerate(ids):
                    name = 'HG001:' + n
                    chrom, idx = n.split(':')
                    # HACK: Handle "chr1", and just "1"
                    # TODO: More general "grab integer after string" -- but regexp is slower
                    if chrom.startswith('chr') or chrom.startswith('Chr'):
                        chrom = chrom[3:]
                    idx = int(idx.strip())
                    trust = is_in_region(chrom=chrom, loc=idx, start_locations=trust_starts, end_locations=trust_ends)
                    if not trust:
                        binary_trust_weight[i] = non_trust_train_weight
            binary_trust_weight = binary_trust_weight.cuda(target.get_device())
            binary_trust_weight.requires_grad = False

            # Multiply by SNP/non-SNP weight -- in case we want to punt on Indels
            non_snp_weight = args.non_snp_train_weight
            batch_snp_weight = is_snp_tensor + (1. - is_snp_tensor) * non_snp_weight
            batch_snp_weight = batch_snp_weight.cuda(target.get_device()).unsqueeze(1)
            total_class_weight = batch_snp_weight * binary_trust_weight

            # Supply vector for binary trust weights as input
            bin_out, vt_out, af_out, cov_out, vb_out, vr_out, e_bin_out, e_vt_out, bin_out_w, vt_out_w, bls, vtls, bs_out, vts_out = model(reads, ref,
                q_scores=q_scores, strands=strands, binary_trust_vector=binary_trust_weight,
                af_scores=target_allele_freq, ref_bases=ref_bases, var_bases=var_bases,
                ref_masks=ref_masks, var_masks=var_masks)

            # If applicable, calculate "early loss" for Bin and VT
            if len(args.early_loss_layers) > 0:
                soft_bin_loss, _ = bin_focal_loss_criteria(bs_out, target_binary, weight=total_class_weight, logits=False)
                #print(soft_bin_loss)
                soft_vt_loss, _ = vt_focal_loss_criteria(vts_out, target_var_type, weight=total_class_weight, logits=False)
                #print(soft_vt_loss)
                binary_loss = soft_bin_loss
                vt_loss = soft_vt_loss

            else:
                 # Try soft label BCE loss
                binary_loss, _ = bin_focal_loss_criteria(bin_out, target_binary, weight=total_class_weight)

                # auxillary losses
                vt_loss, _ = vt_focal_loss_criteria(vt_out, target_var_type, weight=total_class_weight)

            # Don't use AF as loss -- if also potentially using as input?
            af_loss = F.binary_cross_entropy(af_out, target_allele_freq.cuda(), weight=total_class_weight)
            cov_loss = F.mse_loss(cov_out, target_coverage)
            # NOTE: For predicting bases -- focus on bases that are possible in the softmax
            vb_loss = F.cross_entropy(vb_out, target_var_base_enum, weight=torch.Tensor([0.001,1.,1.,1.,1.,1.,0.001,0.001,1.,0.001]).cuda(target.get_device()))
            vr_loss = F.cross_entropy(vr_out, target_var_ref_enum, weight=torch.Tensor([0.001,1.,1.,1.,1.,1.,0.001,0.001,1.,0.001]).cuda(target.get_device()))
            # Compute losses, don't backprop
            loss = binary_loss * args.binary_weight
            loss += (vt_loss + af_loss * args.auxillary_loss_allele_weight + cov_loss + (vb_loss + vr_loss) * args.auxillary_loss_bases_weight) * args.auxillary_loss_weight
            test_loss += loss.cpu()
            total_bin_loss += binary_loss
            total_vt_loss += vt_loss
            total_af_loss += af_loss
            total_cov_loss += cov_loss
            total_base_loss += vb_loss + vr_loss

            # Count predictions [default threshold]
            bin_threshold = 0.5

            # HACK: Convert binary prediction to [positive] probability [0.,1.]
            # Option: Predict with VT -- take 1 - prob(not mutation)
            if args.use_var_type_threshold:
                if len(args.early_loss_layers) > 0:
                    bin_out = vts_out
                else:
                    bin_out = F.softmax((vt_out), dim=1)
                bin_out = 1. - bin_out[:,0]
            else:
                if len(args.early_loss_layers) > 0:
                    bin_out = bs_out
                else:
                    bin_out = F.softmax((bin_out), dim=1)
                bin_out = 1. - bin_out[:,0]
            vt_out = F.softmax((vt_out), dim=1)
            target_binary = target_binary.squeeze().float()

            pred_binary = (bin_out >= bin_threshold).float()
            # Calculate FP, TP, etc here -- to get precision, recall, F1
            correct_binary += pred_binary.eq(target_binary.view_as(pred_binary)).sum().item()
            bin_targets += list(target_binary.cpu().numpy())
            bin_results += list(bin_out.squeeze().cpu().numpy())

            if batch % args.log_interval == 0:
                loss_value = loss
                torch.cuda.synchronize()
                print('  Elapsed ({:.02e}s) [{}/{} ({:.0f}%)]  Loss: {:.6f}  Total: {:.6f}| bin: {:.5f} vt: {:.5f} af: {:.5f} cov: {:.5f} bases: {:.5f}'.format(
                    time.perf_counter()-a, batch * len(ref), len(test_loader.dataset),
                    100. * batch / len(test_loader), loss_value, test_loss/(batch+1),
                    total_bin_loss/(batch+1), total_vt_loss/(batch+1), total_af_loss/(batch+1),
                    total_cov_loss/(batch+1), total_base_loss/(batch+1)), end='\r')
                a = time.perf_counter()

            # If table given, collect GATK for this batch via lookup table.
            if gatk_table:
                ids = items["name"]
                # NOTE: assume that for the default GATK, just check if position is in the table.
                for n in ids:
                    chrom, idx = n.split(':')
                    # HACK: name may be 'chr1' or '1' -- so strip the 'chr' then write it back as needed
                    if chrom.startswith('chr') or chrom.startswith('Chr'):
                        chrom = chrom[3:]
                    n = 'chr' + chrom + ':' + str(idx)
                    name = 'HG001:' + n
                    gatk_results.append((name in gatk_table.keys()))

            # If trusted region given, build target and results for these separately
            if trust_starts:
                ids = items["name"]
                # 'chr1:121120307'
                for i, n in enumerate(ids):
                    chrom, idx = n.split(':')
                    is_snp = is_snp_tensor[i]
                    # HACK: Handle "chr1", and just "1"
                    # TODO: More general "grab integer after string" -- but regexp is slower
                    if chrom.startswith('chr') or chrom.startswith('Chr'):
                        chrom = chrom[3:]
                    idx = int(idx.strip())
                    trust = is_in_region(chrom=chrom, loc=idx, start_locations=trust_starts, end_locations=trust_ends)
                    if trust and (args.trust_snp_only and is_snp):
                        trust_targets.append(target_binary[i])
                        trust_results.append(bin_out[i])
                        # If GATK in trusted region given, collect predictions
                        if gatk_table:
                            n = 'chr' + chrom + ':' + str(idx)
                            name = 'HG001:' + n
                            gatk_trust_results.append((name in gatk_table.keys()))

            # Upon request, append to VCF records
            if args.save_vcf_records:
                if debug:
                    print('appending VCF record to %s' % vcf_tempfile)
                append_vcf_records(vcf_file=vcf_tempfile, bin_results=bin_out, vt_results=vt_out, vcf_records=vcf_records, debug=debug)

    # Now use results and outputs, to compute binary AUC
    # TODO: Choose optimal threshold, for F1 maximization
    print('Flattening results')
    bin_targets = np.array(bin_targets)
    bin_results = np.array(bin_results)
    trust_targets = np.array(trust_targets)
    trust_results = np.array(trust_results)

    fpr, tpr, thresholds = roc_curve(bin_targets, bin_results, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print('binary auc: %.5f' % roc_auc)

    # Precision/recall curve
    precision, recall, thresholds = precision_recall_curve(bin_targets, bin_results)
    # Loop over thresholds, and choose best F1:
    best_i, best_f1 = -1, -1
    for i,_ in enumerate(thresholds):
        f1 = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        if f1 > best_f1:
            best_i = i
            best_f1 = f1
    best_precision = precision[best_i]
    best_recall = recall[best_i]
    best_threshold = thresholds[best_i]
    print('Best threshold %.5f: F1 %.5f (prec: %.5f; recall: %.5f)' % (best_threshold, best_f1, best_precision, best_recall))

    # Show confusion matrix (for best binary threshold above).
    cm = confusion_matrix(bin_targets, bin_results >= best_threshold)
    print('Positives (mutations) -- %d (%.2f%%)' % (np.sum(bin_targets==1.), np.sum(bin_targets==1.)/len(bin_targets) * 100.))
    print('Binary confusion matrix:')
    print(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)

    # If GATK comparison exists, look up results.
    if gatk_table:
        print('\nNow confusion matrix on GATK (with default threshold)')
        gatk_results = np.array(gatk_results)

        precision_gatk, recall_gatk, _ = precision_recall_curve(bin_targets, gatk_results)
        precision_gatk = precision_gatk[1]
        recall_gatk = recall_gatk[1]
        f1_gatk = 2 * (precision_gatk * recall_gatk) / (precision_gatk + recall_gatk)
        print('\nDefault GATK threshold: F1 %.5f (prec: %.5f; recall: %.5f)' % (f1_gatk, precision_gatk, recall_gatk))

        # Show confusion matrix (for best binary threshold above).
        cm = confusion_matrix(bin_targets, gatk_results)
        print('Positives (mutations) -- %d (%.2f%%)' % (np.sum(bin_targets==1.), np.sum(bin_targets==1.)/len(bin_targets) * 100.))
        print('Binary confusion matrix (GATK):')
        print(cm)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm)

        # Since we can't tune GATK (easily) for AUC, show closest threshold on our classifier, for GATK recall:
        idx = np.abs(recall - recall_gatk).argmin()
        print((precision[idx], recall[idx], thresholds[idx]))

        # Confusoin matrix with recall of GATK:
        best_precision = precision[idx]
        best_recall = recall[idx]
        best_f1 = 2 * (best_precision * best_recall) / (best_precision + best_recall)
        best_threshold = thresholds[idx]
        print('\nClosest GATK threshold %.5f: F1 %.5f (prec: %.5f; recall: %.5f)' % (best_threshold, best_f1, best_precision, best_recall))

        cm = confusion_matrix(bin_targets, bin_results >= best_threshold)
        print('Positives (mutations) -- %d (%.2f%%)' % (np.sum(bin_targets==1.), np.sum(bin_targets==1.)/len(bin_targets) * 100.))
        print('Binary confusion matrix:')
        print(cm)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm)

    # If available, show confusion matrix on trust regions only
    if args.trust_snp_only:
        snp_string = ' SNP only'
    if trust_starts and False:
        fpr, tpr, _ = roc_curve(trust_targets, trust_results, pos_label=1)
        roc_auc = auc(fpr, tpr)
        print('\n[Trust region%s] binary auc: %.5f' % (snp_string, roc_auc))

        # Precision/recall curve
        trust_precision, trust_recall, trust_thresholds = precision_recall_curve(trust_targets, trust_results)
        # Loop over thresholds, and choose best F1:
        best_i, best_f1 = -1, -1
        for i,_ in enumerate(trust_thresholds):
            f1 = 2 * (trust_precision[i] * trust_recall[i]) / (trust_precision[i] + trust_recall[i])
            if f1 > best_f1:
                best_i = i
                best_f1 = f1
        best_precision = trust_precision[best_i]
        best_recall = trust_recall[best_i]
        best_threshold = trust_thresholds[best_i]
        print('[Trust region] Best threshold %.5f: F1 %.5f (prec: %.5f; recall: %.5f)' % (best_threshold, best_f1, best_precision, best_recall))

        # Show confusion matrix (for best binary threshold above).
        cm = confusion_matrix(trust_targets, trust_results >= best_threshold)
        print('[Trust region] Positives (mutations) -- %d (%.2f%%)' % (np.sum(trust_targets==1.), np.sum(trust_targets==1.)/len(trust_targets) * 100.))
        print('[Trust region] Binary confusion matrix:')
        print(cm)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm)

    # Show GATK confusion matrix, on Trust Regions only
    if gatk_table and trust_starts and False:
        print('\n[Trust region] confusion matrix on GATK (default threshold)')
        gatk_trust_results = np.array(gatk_trust_results)

        precision_gatk, recall_gatk, _ = precision_recall_curve(trust_targets, gatk_trust_results)
        precision_gatk = precision_gatk[1]
        recall_gatk = recall_gatk[1]
        f1_gatk = 2 * (precision_gatk * recall_gatk) / (precision_gatk + recall_gatk)
        print('[Trust region] GATK: F1 %.5f (prec: %.5f; recall: %.5f)' % (f1_gatk, precision_gatk, recall_gatk))

        cm = confusion_matrix(trust_targets, gatk_trust_results)
        print('[Trust region] Positives (mutations) -- %d (%.2f%%)' % (np.sum(trust_targets==1.), np.sum(trust_targets==1.)/len(trust_targets) * 100.))
        print('[Trust region] Binary confusion matrix (GATK):')
        print(cm)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm)

        # And finally, find a fit to match GATK precision [on trust regions only] so we can compare error rates
        # Since we can't tune GATK (easily) for AUC, show closest threshold on our classifier, for GATK recall:
        idx = np.abs(trust_recall - recall_gatk).argmin()
        print((trust_precision[idx], trust_recall[idx], trust_thresholds[idx]))

        # Confusion matrix with same recall as GATK -- show our gains (or losses) on precision
        best_precision = trust_precision[idx]
        best_recall = trust_recall[idx]
        best_f1 = 2 * (best_precision * best_recall) / (best_precision + best_recall)
        best_threshold = trust_thresholds[idx]
        print('\n[Trust region] Closest GATK threshold %.5f: F1 %.5f (prec: %.5f; recall: %.5f)' % (best_threshold, best_f1, best_precision, best_recall))

        cm = confusion_matrix(trust_targets, trust_results >= best_threshold)
        print('[Trust region] Positives (mutations) -- %d (%.2f%%)' % (np.sum(trust_targets==1.), np.sum(trust_targets==1.)/len(trust_targets) * 100.))
        print('[Trust region] Binary confusion matrix:')
        print(cm)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(cm)

    # Save VCF records, upon request
    # Actually quite slow. Don't collect VCF records unless we are going to use them...
    if args.save_vcf_records:
        print('Finished streaming VCFs -- should be saved %d records to file %s' % (len(bin_results), vcf_tempfile))

    test_loss /= (batch+1)

    print('\nTest set: Average loss: {:.6f}, Bin-Acc: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct_binary, len(test_loader.dataset), 100. * correct_binary / len(test_loader.dataset)))

    print('', flush=True)
    return test_loss

