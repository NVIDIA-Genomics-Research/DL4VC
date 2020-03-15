# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """Soft version of BCEWithLogitsLoss. Supports label smoothing.
    Args:
        label_smoothing (float):
            The smoothing parameter :math:`epsilon` for label smoothing. For InceptionV3,
            :math:`epsilon`=0.1 [albeit with 1000 classes]. What does DeepVariant use?
        weight (:class:`torch.Tensor`):
            A 1D tensor of size equal to the number of classes. Pass-through to BCELoss.
        num_classes (int):
            The number of classes. Matters for label smoothing.
    """

    def __init__(self, label_smoothing=0, pos_weight=None, num_classes=2, close_match_window=2., **kwargs):
        super(SoftBCEWithLogitsLoss, self).__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self.confidence = 1 - self.label_smoothing
        self.num_classes = num_classes
        # For numerical stability -- not really necessary since no exp() but just in case
        self.epsilon = 1.e-8
        self.close_match_window = close_match_window
        self.register_buffer('pos_weight', pos_weight)
        assert label_smoothing >= 0.0 and label_smoothing <= 1.0

    def forward(self, input, target, weight=None):
        # Expand targets and add smoothing value
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.label_smoothing / (self.num_classes - 1))
        one_hot.scatter_(1, target, self.confidence)

        # Find which examples within constant distance of label_smoothing
        # Sigmoid and clamp -- manually expand for weight calculation
        #input_pred = torch.sigmoid(input)
        input_pred = F.softmax(input, dim=1)
        input_pred = input_pred.clamp(self.epsilon, 1. - self.epsilon)
        distance = torch.sum(torch.abs(input_pred - one_hot), dim=1) / 2.
        close_distance = distance <= (self.label_smoothing * self.close_match_window)

        # Pass example weight, if any (None is fine) and category weight (less to common cases)
        return F.binary_cross_entropy_with_logits(input, one_hot,
                                                  weight,
                                                  pos_weight=self.pos_weight), close_distance

class SoftBCEWithLogitsFocalLoss(nn.BCEWithLogitsLoss):
    """
    Implements focal loss -- for multiclass case (prediction close to 1.0 in any class gets down-weighted)
    Focal Loss for Dense Object Detection (FAIR)
    https://arxiv.org/pdf/1708.02002.pdf
    """
    def __init__(self, label_smoothing=0, pos_weight=None, num_classes=2,
        alpha=1., gamma=0., close_match_window=2., **kwargs):
        super(SoftBCEWithLogitsFocalLoss, self).__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self.confidence = 1 - self.label_smoothing
        self.num_classes = num_classes

        # NOTE -- we do not really use Alpha -- since we supply class weight separately.
        self.alpha = alpha
        self.gamma = gamma
        # For numerical stability -- not really necessary since no exp() but just in case
        self.epsilon = 1.e-8
        # HACK: Return index of examples within {close_match_window * label_smoothing}
        # distance of target label.
        # Why? Report how many examples are "well classified" (and could be downsampled in training)
        # NOTE: A perfect binary classification (1., 0., 0.) is (1.0 * label_smoothing) distance from target.
        self.close_match_window = close_match_window
        self.register_buffer('pos_weight', pos_weight)
        assert label_smoothing >= 0.0 and label_smoothing <= 1.0

    def forward(self, input, target, weight=None, logits=True):
        # Expand targets and add smoothing value
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.label_smoothing / (self.num_classes - 1))
        one_hot.scatter_(1, target, self.confidence)

        # Avoid rolling our own BCE -- numerically unstable -- use PyTorch version
        if logits:
            ce = F.binary_cross_entropy_with_logits(input, one_hot,
                                                    weight, reduction='none')
        else:
            input = input.clamp(0., 1.)
            ce = F.binary_cross_entropy(input, one_hot,
                                        weight, reduction='none')

        # Sigmoid and clamp -- manually expand for weight calculation
        if logits:
            input = F.softmax(input, dim=1)
            input = input.clamp(0., 1.)

        # compute example weight
        # Pt = "correctness probability" from the paper.
        # (p if y==1, (1-p) if y==0) -- binary case [ours is soft version]
        pt = one_hot * input + (1-one_hot) * (1-input)
        weight = (1 - pt) ** self.gamma
        # Multiply by category weight, if present
        weight = weight * self.pos_weight / torch.sum(self.pos_weight)
        # Focal loss is cross entropy, times weight
        focal_loss = self.alpha * weight * ce
        # Sum in n-classes dimension, and mean reduce.
        focal_loss = torch.sum(focal_loss, dim=1)
        # Reduce across the batch -- or don't [if we want to reduce ourselves]
        focal_loss = torch.mean(focal_loss)

        # Find which examples within constant distance of label_smoothing
        distance = torch.sum(torch.abs(input - one_hot), dim=1) / 2.
        close_distance = distance <= (self.label_smoothing * self.close_match_window)
        return focal_loss, close_distance




