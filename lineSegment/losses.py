# import tensorflow as tf


# def weighted_loss(label, logits):
#     logits = tf.sigmoid(logits)
#     positive_wire_mask = tf.cast(label, tf.bool)
#     negative_wire_mask = tf.logical_not(positive_wire_mask)
#     wire_acc = tf.reduce_sum(1. - tf.boolean_mask(logits, positive_wire_mask))
#     no_wire_acc = tf.reduce_sum(tf.boolean_mask(logits, negative_wire_mask))
#     t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
#     loss = 5 * wire_acc + no_wire_acc + t * 100
#     return loss


# def weighted_loss1(label, logits):
#     logit = tf.sigmoid(logits)
#     t1 = tf.reduce_mean(tf.losses.mean_squared_error(label[..., 1], logit[..., 1]))
#     t2 = tf.reduce_mean(tf.losses.mean_squared_error(label[..., 0], logit[..., 0]))
#     loss = t1 * 100 + t2
#     return loss

import torch
import torch.nn.functional as F
import torch.nn as nn

def weighted_loss(logits, label):
    # 使用 sigmoid 激活函数
    logits1 = torch.sigmoid(logits)

    # 创建 mask
    positive_wire_mask = label>0.5
    negative_wire_mask = ~positive_wire_mask  # 取反的 mask
    # print(positive_wire_mask)
    # print(negative_wire_mask)

    # 计算 wire_acc 和 no_wire_acc
    wire_acc = torch.sum(1.0 - logits1[positive_wire_mask])
    no_wire_acc = torch.sum(logits1[negative_wire_mask])

    # softmax 交叉熵损失
    # print(logits.shape, label.shape)
    # t = F.cross_entropy(logits, label.long().squeeze(1), reduction='mean')
    # t = nn.BCEWithLogitsLoss(logits, label)

    # 总损失
    loss1 = 5 * wire_acc + no_wire_acc
    return loss1


def weighted_loss1(label, logits):
    # 使用 sigmoid 激活函数
    logits = torch.sigmoid(logits)

    # 计算均方误差损失
    t1 = F.mse_loss(logits[..., 1], label[..., 1].float(), reduction='mean')
    t2 = F.mse_loss(logits[..., 0], label[..., 0].float(), reduction='mean')

    # 总损失
    loss = t1 * 100 + t2
    return loss
