from __future__ import absolute_import
import math

import chainer
import chainer.functions as F
import numpy as np

from ..links import CLink
from ..links import BinaryLinear
from ..links import BatchNormalization
from ..links import SoftmaxCrossEntropy
from ..utils import binary_util as bu


class BinaryLinearBNSoftmax(chainer.Chain, CLink):
    def __init__(self, out_channels):
        super(BinaryLinearBNSoftmax, self).__init__()
        self.cname = "l_b_linear_bn_softmax"
        with self.init_scope():
            self.bl = BinaryLinear(out_channels)
            self.bn = BatchNormalization(out_channels)
            self.sm = SoftmaxCrossEntropy()

    def __call__(self, h, t=None):
        return self.bn(self.bl(h))

    def generate_c(self, link_idx, inp_shape):
        name = self.cname + str(link_idx)
        text = []

        # BinaryLinear bl
        l = self.bl
        lName = l.name
        lname = name + '_' + lName
        for p in l.params():
            pname = p.name
            if pname == 'W':
                text += [bu.np_to_uint8C(bu.binarize_real(p.data.T),
                                         lname + '_' + pname, 'col_major', pad='1')]
                num_classes = p.data.shape[0]
                fc_size = p.data.shape[1]
            elif pname == 'b':
                text += [bu.np_to_floatC(p.data, lname +
                                         '_' + pname, 'row_major')]

        # BatchNormalization bn
        l = self.bn
        lName = l.name
        lname = name + '_' + lName
        for p in l.params():
            pname = p.name
            if pname == 'gamma':
                text += [bu.np_to_floatC(p.data, lname +
                                         '_' + pname, 'row_major')]
            elif pname == 'beta':
                text += [bu.np_to_floatC(p.data, lname +
                                         '_' + pname, 'row_major')]
        for p in l._persistent:
            pname = p
            persistent = l.__dict__[p]
            if pname == 'avg_mean':
                text += [bu.np_to_floatC(persistent,
                                         lname + '_mean', 'row_major')]
            elif pname == 'avg_var':
                text += [bu.np_to_floatC(np.sqrt(persistent,
                                                 dtype=persistent.dtype), lname + '_std', 'row_major')]

        text = "\n".join(text) + '\n'
        m = 1
        n = fc_size
        k = num_classes

        ftext = "void {name}(uint8_t* input, uint8_t* output){{\n"
        ftext += "  blinear_layer(input, {name}_bl_W, output, {name}_bl_b, {name}_bn_gamma, {name}_bn_beta, {name}_bn_mean, {name}_bn_std, {m}, {n}, {k}); \n}}\n\n"
        ftext = ftext.format(name=name, m=m, n=n, k=k)
        text += ftext

        return text

    def param_mem(self):
        mem = 0.
        for p in self.bl.params():
            if p.name == 'W':
                m, w = p.data.shape
                mem += m * math.ceil(w / 8.)
                #Bias + BN
                mem += 5 * m * 32

        return mem

    def temp_mem(self, inp_shape):
        m = inp_shape[0]
        w = np.prod(inp_shape[1:])
        res_w = math.ceil(w / 8.)
        return m * res_w
