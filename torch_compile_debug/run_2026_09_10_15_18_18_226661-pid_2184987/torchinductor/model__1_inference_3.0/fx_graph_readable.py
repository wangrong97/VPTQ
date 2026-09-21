class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2048, 128]", arg1_1: "f32[128, 128]", arg2_1: "f32[8, 16]", arg3_1: "f32[8, 16, 2]"):
        # File: /tmp/try_compile.py:12 in block_loop, code: idx_out = torch.zeros(n_row, cols, nvec, dtype=torch.long, device=dev)
        full_default: "i64[8, 128, 128]" = torch.ops.aten.full.default([8, 128, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select: "f32[2048]" = torch.ops.aten.select.int(arg0_1, 1, 0)
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_1: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 0)
        select_2: "f32[]" = torch.ops.aten.select.int(select_1, 0, 0);  select_1 = None
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        view: "f32[8, 128, 2]" = torch.ops.aten.view.default(select, [8, 128, 2])
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        baddbmm: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze, view, permute, alpha = -2.0);  unsqueeze = view = permute = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm, -1);  baddbmm = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_1: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin, -1)
        expand: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_1, [8, 128, 2]);  unsqueeze_1 = None
        gather: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand);  expand = None
        view_1: "f32[2048]" = torch.ops.aten.view.default(gather, [-1]);  gather = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_3: "i64[8, 128]" = torch.ops.aten.select.int(full_default, 1, 0)
        copy: "i64[8, 128]" = torch.ops.aten.copy.default(select_3, argmin);  select_3 = argmin = None
        select_scatter: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(full_default, copy, 1, 0);  full_default = copy = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub: "f32[2048]" = torch.ops.aten.sub.Tensor(select, view_1);  select = view_1 = None
        div: "f32[2048]" = torch.ops.aten.div.Tensor(sub, select_2);  sub = select_2 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_5: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 0)
        expand_1: "f32[2048, 128]" = torch.ops.aten.expand.default(arg0_1, [2048, 128])
        mul: "f32[2048, 128]" = torch.ops.aten.mul.Tensor(expand_1, 1);  expand_1 = None
        view_2: "f32[2048, 1]" = torch.ops.aten.view.default(div, [2048, 1]);  div = None
        mul_1: "f32[2048, 128]" = torch.ops.aten.mul.Tensor(view_2, select_5);  view_2 = select_5 = None
        mul_2: "f32[2048, 128]" = torch.ops.aten.mul.Tensor(mul_1, -1.0);  mul_1 = None
        add: "f32[2048, 128]" = torch.ops.aten.add.Tensor(mul, mul_2);  mul = mul_2 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_7: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 1)
        select_8: "f32[]" = torch.ops.aten.select.int(select_7, 0, 1);  select_7 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_9: "f32[2048]" = torch.ops.aten.select.int(add, 1, 1)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_2: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_1: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_10: "f32[2048]" = torch.ops.aten.select.int(add, 1, 1)
        view_4: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_10, [8, 128, 2]);  select_10 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_1: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_2, view_4, permute_1, alpha = -2.0);  unsqueeze_2 = view_4 = permute_1 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_1: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_1, -1);  baddbmm_1 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_3: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_1, -1)
        expand_2: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_3, [8, 128, 2]);  unsqueeze_3 = None
        gather_1: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_2);  expand_2 = None
        view_5: "f32[2048]" = torch.ops.aten.view.default(gather_1, [-1]);  gather_1 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_12: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter, 1, 1)
        copy_1: "i64[8, 128]" = torch.ops.aten.copy.default(select_12, argmin_1);  select_12 = argmin_1 = None
        select_scatter_1: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter, copy_1, 1, 1);  select_scatter = copy_1 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_1: "f32[2048]" = torch.ops.aten.sub.Tensor(select_9, view_5);  select_9 = view_5 = None
        div_1: "f32[2048]" = torch.ops.aten.div.Tensor(sub_1, select_8);  sub_1 = select_8 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_14: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 1)
        slice_2: "f32[127]" = torch.ops.aten.slice.Tensor(select_14, 0, 1, 9223372036854775807);  select_14 = None
        slice_3: "f32[2048, 127]" = torch.ops.aten.slice.Tensor(add, 1, 1, 9223372036854775807)
        expand_3: "f32[2048, 127]" = torch.ops.aten.expand.default(slice_3, [2048, 127]);  slice_3 = None
        mul_3: "f32[2048, 127]" = torch.ops.aten.mul.Tensor(expand_3, 1);  expand_3 = None
        view_6: "f32[2048, 1]" = torch.ops.aten.view.default(div_1, [2048, 1]);  div_1 = None
        mul_4: "f32[2048, 127]" = torch.ops.aten.mul.Tensor(view_6, slice_2);  view_6 = slice_2 = None
        mul_5: "f32[2048, 127]" = torch.ops.aten.mul.Tensor(mul_4, -1.0);  mul_4 = None
        add_1: "f32[2048, 127]" = torch.ops.aten.add.Tensor(mul_3, mul_5);  mul_3 = mul_5 = None
        slice_scatter: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(add, add_1, 1, 1, 9223372036854775807);  add = add_1 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_16: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 2)
        select_17: "f32[]" = torch.ops.aten.select.int(select_16, 0, 2);  select_16 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_18: "f32[2048]" = torch.ops.aten.select.int(slice_scatter, 1, 2)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_4: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_2: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_19: "f32[2048]" = torch.ops.aten.select.int(slice_scatter, 1, 2)
        view_8: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_19, [8, 128, 2]);  select_19 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_2: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_4, view_8, permute_2, alpha = -2.0);  unsqueeze_4 = view_8 = permute_2 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_2: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_2, -1);  baddbmm_2 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_5: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_2, -1)
        expand_4: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_5, [8, 128, 2]);  unsqueeze_5 = None
        gather_2: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_4);  expand_4 = None
        view_9: "f32[2048]" = torch.ops.aten.view.default(gather_2, [-1]);  gather_2 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_21: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_1, 1, 2)
        copy_2: "i64[8, 128]" = torch.ops.aten.copy.default(select_21, argmin_2);  select_21 = argmin_2 = None
        select_scatter_2: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_1, copy_2, 1, 2);  select_scatter_1 = copy_2 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_2: "f32[2048]" = torch.ops.aten.sub.Tensor(select_18, view_9);  select_18 = view_9 = None
        div_2: "f32[2048]" = torch.ops.aten.div.Tensor(sub_2, select_17);  sub_2 = select_17 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_23: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 2)
        slice_6: "f32[126]" = torch.ops.aten.slice.Tensor(select_23, 0, 2, 9223372036854775807);  select_23 = None
        slice_7: "f32[2048, 126]" = torch.ops.aten.slice.Tensor(slice_scatter, 1, 2, 9223372036854775807)
        expand_5: "f32[2048, 126]" = torch.ops.aten.expand.default(slice_7, [2048, 126]);  slice_7 = None
        mul_6: "f32[2048, 126]" = torch.ops.aten.mul.Tensor(expand_5, 1);  expand_5 = None
        view_10: "f32[2048, 1]" = torch.ops.aten.view.default(div_2, [2048, 1]);  div_2 = None
        mul_7: "f32[2048, 126]" = torch.ops.aten.mul.Tensor(view_10, slice_6);  view_10 = slice_6 = None
        mul_8: "f32[2048, 126]" = torch.ops.aten.mul.Tensor(mul_7, -1.0);  mul_7 = None
        add_2: "f32[2048, 126]" = torch.ops.aten.add.Tensor(mul_6, mul_8);  mul_6 = mul_8 = None
        slice_scatter_1: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter, add_2, 1, 2, 9223372036854775807);  slice_scatter = add_2 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_25: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 3)
        select_26: "f32[]" = torch.ops.aten.select.int(select_25, 0, 3);  select_25 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_27: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_1, 1, 3)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_6: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_3: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_28: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_1, 1, 3)
        view_12: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_28, [8, 128, 2]);  select_28 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_3: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_6, view_12, permute_3, alpha = -2.0);  unsqueeze_6 = view_12 = permute_3 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_3: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_3, -1);  baddbmm_3 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_7: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_3, -1)
        expand_6: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_7, [8, 128, 2]);  unsqueeze_7 = None
        gather_3: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_6);  expand_6 = None
        view_13: "f32[2048]" = torch.ops.aten.view.default(gather_3, [-1]);  gather_3 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_30: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_2, 1, 3)
        copy_3: "i64[8, 128]" = torch.ops.aten.copy.default(select_30, argmin_3);  select_30 = argmin_3 = None
        select_scatter_3: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_2, copy_3, 1, 3);  select_scatter_2 = copy_3 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_3: "f32[2048]" = torch.ops.aten.sub.Tensor(select_27, view_13);  select_27 = view_13 = None
        div_3: "f32[2048]" = torch.ops.aten.div.Tensor(sub_3, select_26);  sub_3 = select_26 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_32: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 3)
        slice_10: "f32[125]" = torch.ops.aten.slice.Tensor(select_32, 0, 3, 9223372036854775807);  select_32 = None
        slice_11: "f32[2048, 125]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 1, 3, 9223372036854775807)
        expand_7: "f32[2048, 125]" = torch.ops.aten.expand.default(slice_11, [2048, 125]);  slice_11 = None
        mul_9: "f32[2048, 125]" = torch.ops.aten.mul.Tensor(expand_7, 1);  expand_7 = None
        view_14: "f32[2048, 1]" = torch.ops.aten.view.default(div_3, [2048, 1]);  div_3 = None
        mul_10: "f32[2048, 125]" = torch.ops.aten.mul.Tensor(view_14, slice_10);  view_14 = slice_10 = None
        mul_11: "f32[2048, 125]" = torch.ops.aten.mul.Tensor(mul_10, -1.0);  mul_10 = None
        add_3: "f32[2048, 125]" = torch.ops.aten.add.Tensor(mul_9, mul_11);  mul_9 = mul_11 = None
        slice_scatter_2: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_1, add_3, 1, 3, 9223372036854775807);  slice_scatter_1 = add_3 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_34: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 4)
        select_35: "f32[]" = torch.ops.aten.select.int(select_34, 0, 4);  select_34 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_36: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_2, 1, 4)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_8: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_4: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_37: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_2, 1, 4)
        view_16: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_37, [8, 128, 2]);  select_37 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_4: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_8, view_16, permute_4, alpha = -2.0);  unsqueeze_8 = view_16 = permute_4 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_4: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_4, -1);  baddbmm_4 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_9: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_4, -1)
        expand_8: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_9, [8, 128, 2]);  unsqueeze_9 = None
        gather_4: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_8);  expand_8 = None
        view_17: "f32[2048]" = torch.ops.aten.view.default(gather_4, [-1]);  gather_4 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_39: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_3, 1, 4)
        copy_4: "i64[8, 128]" = torch.ops.aten.copy.default(select_39, argmin_4);  select_39 = argmin_4 = None
        select_scatter_4: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_3, copy_4, 1, 4);  select_scatter_3 = copy_4 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_4: "f32[2048]" = torch.ops.aten.sub.Tensor(select_36, view_17);  select_36 = view_17 = None
        div_4: "f32[2048]" = torch.ops.aten.div.Tensor(sub_4, select_35);  sub_4 = select_35 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_41: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 4)
        slice_14: "f32[124]" = torch.ops.aten.slice.Tensor(select_41, 0, 4, 9223372036854775807);  select_41 = None
        slice_15: "f32[2048, 124]" = torch.ops.aten.slice.Tensor(slice_scatter_2, 1, 4, 9223372036854775807)
        expand_9: "f32[2048, 124]" = torch.ops.aten.expand.default(slice_15, [2048, 124]);  slice_15 = None
        mul_12: "f32[2048, 124]" = torch.ops.aten.mul.Tensor(expand_9, 1);  expand_9 = None
        view_18: "f32[2048, 1]" = torch.ops.aten.view.default(div_4, [2048, 1]);  div_4 = None
        mul_13: "f32[2048, 124]" = torch.ops.aten.mul.Tensor(view_18, slice_14);  view_18 = slice_14 = None
        mul_14: "f32[2048, 124]" = torch.ops.aten.mul.Tensor(mul_13, -1.0);  mul_13 = None
        add_4: "f32[2048, 124]" = torch.ops.aten.add.Tensor(mul_12, mul_14);  mul_12 = mul_14 = None
        slice_scatter_3: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_2, add_4, 1, 4, 9223372036854775807);  slice_scatter_2 = add_4 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_43: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 5)
        select_44: "f32[]" = torch.ops.aten.select.int(select_43, 0, 5);  select_43 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_45: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_3, 1, 5)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_10: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_5: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_46: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_3, 1, 5)
        view_20: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_46, [8, 128, 2]);  select_46 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_5: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_10, view_20, permute_5, alpha = -2.0);  unsqueeze_10 = view_20 = permute_5 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_5: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_5, -1);  baddbmm_5 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_11: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_5, -1)
        expand_10: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_11, [8, 128, 2]);  unsqueeze_11 = None
        gather_5: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_10);  expand_10 = None
        view_21: "f32[2048]" = torch.ops.aten.view.default(gather_5, [-1]);  gather_5 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_48: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_4, 1, 5)
        copy_5: "i64[8, 128]" = torch.ops.aten.copy.default(select_48, argmin_5);  select_48 = argmin_5 = None
        select_scatter_5: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_4, copy_5, 1, 5);  select_scatter_4 = copy_5 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_5: "f32[2048]" = torch.ops.aten.sub.Tensor(select_45, view_21);  select_45 = view_21 = None
        div_5: "f32[2048]" = torch.ops.aten.div.Tensor(sub_5, select_44);  sub_5 = select_44 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_50: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 5)
        slice_18: "f32[123]" = torch.ops.aten.slice.Tensor(select_50, 0, 5, 9223372036854775807);  select_50 = None
        slice_19: "f32[2048, 123]" = torch.ops.aten.slice.Tensor(slice_scatter_3, 1, 5, 9223372036854775807)
        expand_11: "f32[2048, 123]" = torch.ops.aten.expand.default(slice_19, [2048, 123]);  slice_19 = None
        mul_15: "f32[2048, 123]" = torch.ops.aten.mul.Tensor(expand_11, 1);  expand_11 = None
        view_22: "f32[2048, 1]" = torch.ops.aten.view.default(div_5, [2048, 1]);  div_5 = None
        mul_16: "f32[2048, 123]" = torch.ops.aten.mul.Tensor(view_22, slice_18);  view_22 = slice_18 = None
        mul_17: "f32[2048, 123]" = torch.ops.aten.mul.Tensor(mul_16, -1.0);  mul_16 = None
        add_5: "f32[2048, 123]" = torch.ops.aten.add.Tensor(mul_15, mul_17);  mul_15 = mul_17 = None
        slice_scatter_4: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_3, add_5, 1, 5, 9223372036854775807);  slice_scatter_3 = add_5 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_52: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 6)
        select_53: "f32[]" = torch.ops.aten.select.int(select_52, 0, 6);  select_52 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_54: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_4, 1, 6)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_12: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_6: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_55: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_4, 1, 6)
        view_24: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_55, [8, 128, 2]);  select_55 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_6: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_12, view_24, permute_6, alpha = -2.0);  unsqueeze_12 = view_24 = permute_6 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_6: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_6, -1);  baddbmm_6 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_13: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_6, -1)
        expand_12: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_13, [8, 128, 2]);  unsqueeze_13 = None
        gather_6: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_12);  expand_12 = None
        view_25: "f32[2048]" = torch.ops.aten.view.default(gather_6, [-1]);  gather_6 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_57: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_5, 1, 6)
        copy_6: "i64[8, 128]" = torch.ops.aten.copy.default(select_57, argmin_6);  select_57 = argmin_6 = None
        select_scatter_6: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_5, copy_6, 1, 6);  select_scatter_5 = copy_6 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_6: "f32[2048]" = torch.ops.aten.sub.Tensor(select_54, view_25);  select_54 = view_25 = None
        div_6: "f32[2048]" = torch.ops.aten.div.Tensor(sub_6, select_53);  sub_6 = select_53 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_59: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 6)
        slice_22: "f32[122]" = torch.ops.aten.slice.Tensor(select_59, 0, 6, 9223372036854775807);  select_59 = None
        slice_23: "f32[2048, 122]" = torch.ops.aten.slice.Tensor(slice_scatter_4, 1, 6, 9223372036854775807)
        expand_13: "f32[2048, 122]" = torch.ops.aten.expand.default(slice_23, [2048, 122]);  slice_23 = None
        mul_18: "f32[2048, 122]" = torch.ops.aten.mul.Tensor(expand_13, 1);  expand_13 = None
        view_26: "f32[2048, 1]" = torch.ops.aten.view.default(div_6, [2048, 1]);  div_6 = None
        mul_19: "f32[2048, 122]" = torch.ops.aten.mul.Tensor(view_26, slice_22);  view_26 = slice_22 = None
        mul_20: "f32[2048, 122]" = torch.ops.aten.mul.Tensor(mul_19, -1.0);  mul_19 = None
        add_6: "f32[2048, 122]" = torch.ops.aten.add.Tensor(mul_18, mul_20);  mul_18 = mul_20 = None
        slice_scatter_5: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_4, add_6, 1, 6, 9223372036854775807);  slice_scatter_4 = add_6 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_61: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 7)
        select_62: "f32[]" = torch.ops.aten.select.int(select_61, 0, 7);  select_61 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_63: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_5, 1, 7)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_14: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_7: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_64: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_5, 1, 7)
        view_28: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_64, [8, 128, 2]);  select_64 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_7: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_14, view_28, permute_7, alpha = -2.0);  unsqueeze_14 = view_28 = permute_7 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_7: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_7, -1);  baddbmm_7 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_15: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_7, -1)
        expand_14: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_15, [8, 128, 2]);  unsqueeze_15 = None
        gather_7: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_14);  expand_14 = None
        view_29: "f32[2048]" = torch.ops.aten.view.default(gather_7, [-1]);  gather_7 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_66: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_6, 1, 7)
        copy_7: "i64[8, 128]" = torch.ops.aten.copy.default(select_66, argmin_7);  select_66 = argmin_7 = None
        select_scatter_7: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_6, copy_7, 1, 7);  select_scatter_6 = copy_7 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_7: "f32[2048]" = torch.ops.aten.sub.Tensor(select_63, view_29);  select_63 = view_29 = None
        div_7: "f32[2048]" = torch.ops.aten.div.Tensor(sub_7, select_62);  sub_7 = select_62 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_68: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 7)
        slice_26: "f32[121]" = torch.ops.aten.slice.Tensor(select_68, 0, 7, 9223372036854775807);  select_68 = None
        slice_27: "f32[2048, 121]" = torch.ops.aten.slice.Tensor(slice_scatter_5, 1, 7, 9223372036854775807)
        expand_15: "f32[2048, 121]" = torch.ops.aten.expand.default(slice_27, [2048, 121]);  slice_27 = None
        mul_21: "f32[2048, 121]" = torch.ops.aten.mul.Tensor(expand_15, 1);  expand_15 = None
        view_30: "f32[2048, 1]" = torch.ops.aten.view.default(div_7, [2048, 1]);  div_7 = None
        mul_22: "f32[2048, 121]" = torch.ops.aten.mul.Tensor(view_30, slice_26);  view_30 = slice_26 = None
        mul_23: "f32[2048, 121]" = torch.ops.aten.mul.Tensor(mul_22, -1.0);  mul_22 = None
        add_7: "f32[2048, 121]" = torch.ops.aten.add.Tensor(mul_21, mul_23);  mul_21 = mul_23 = None
        slice_scatter_6: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_5, add_7, 1, 7, 9223372036854775807);  slice_scatter_5 = add_7 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_70: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 8)
        select_71: "f32[]" = torch.ops.aten.select.int(select_70, 0, 8);  select_70 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_72: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_6, 1, 8)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_16: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_8: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_73: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_6, 1, 8)
        view_32: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_73, [8, 128, 2]);  select_73 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_8: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_16, view_32, permute_8, alpha = -2.0);  unsqueeze_16 = view_32 = permute_8 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_8: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_8, -1);  baddbmm_8 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_17: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_8, -1)
        expand_16: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_17, [8, 128, 2]);  unsqueeze_17 = None
        gather_8: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_16);  expand_16 = None
        view_33: "f32[2048]" = torch.ops.aten.view.default(gather_8, [-1]);  gather_8 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_75: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_7, 1, 8)
        copy_8: "i64[8, 128]" = torch.ops.aten.copy.default(select_75, argmin_8);  select_75 = argmin_8 = None
        select_scatter_8: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_7, copy_8, 1, 8);  select_scatter_7 = copy_8 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_8: "f32[2048]" = torch.ops.aten.sub.Tensor(select_72, view_33);  select_72 = view_33 = None
        div_8: "f32[2048]" = torch.ops.aten.div.Tensor(sub_8, select_71);  sub_8 = select_71 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_77: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 8)
        slice_30: "f32[120]" = torch.ops.aten.slice.Tensor(select_77, 0, 8, 9223372036854775807);  select_77 = None
        slice_31: "f32[2048, 120]" = torch.ops.aten.slice.Tensor(slice_scatter_6, 1, 8, 9223372036854775807)
        expand_17: "f32[2048, 120]" = torch.ops.aten.expand.default(slice_31, [2048, 120]);  slice_31 = None
        mul_24: "f32[2048, 120]" = torch.ops.aten.mul.Tensor(expand_17, 1);  expand_17 = None
        view_34: "f32[2048, 1]" = torch.ops.aten.view.default(div_8, [2048, 1]);  div_8 = None
        mul_25: "f32[2048, 120]" = torch.ops.aten.mul.Tensor(view_34, slice_30);  view_34 = slice_30 = None
        mul_26: "f32[2048, 120]" = torch.ops.aten.mul.Tensor(mul_25, -1.0);  mul_25 = None
        add_8: "f32[2048, 120]" = torch.ops.aten.add.Tensor(mul_24, mul_26);  mul_24 = mul_26 = None
        slice_scatter_7: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_6, add_8, 1, 8, 9223372036854775807);  slice_scatter_6 = add_8 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_79: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 9)
        select_80: "f32[]" = torch.ops.aten.select.int(select_79, 0, 9);  select_79 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_81: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_7, 1, 9)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_18: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_9: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_82: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_7, 1, 9)
        view_36: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_82, [8, 128, 2]);  select_82 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_9: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_18, view_36, permute_9, alpha = -2.0);  unsqueeze_18 = view_36 = permute_9 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_9: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_9, -1);  baddbmm_9 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_19: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_9, -1)
        expand_18: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_19, [8, 128, 2]);  unsqueeze_19 = None
        gather_9: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_18);  expand_18 = None
        view_37: "f32[2048]" = torch.ops.aten.view.default(gather_9, [-1]);  gather_9 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_84: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_8, 1, 9)
        copy_9: "i64[8, 128]" = torch.ops.aten.copy.default(select_84, argmin_9);  select_84 = argmin_9 = None
        select_scatter_9: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_8, copy_9, 1, 9);  select_scatter_8 = copy_9 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_9: "f32[2048]" = torch.ops.aten.sub.Tensor(select_81, view_37);  select_81 = view_37 = None
        div_9: "f32[2048]" = torch.ops.aten.div.Tensor(sub_9, select_80);  sub_9 = select_80 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_86: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 9)
        slice_34: "f32[119]" = torch.ops.aten.slice.Tensor(select_86, 0, 9, 9223372036854775807);  select_86 = None
        slice_35: "f32[2048, 119]" = torch.ops.aten.slice.Tensor(slice_scatter_7, 1, 9, 9223372036854775807)
        expand_19: "f32[2048, 119]" = torch.ops.aten.expand.default(slice_35, [2048, 119]);  slice_35 = None
        mul_27: "f32[2048, 119]" = torch.ops.aten.mul.Tensor(expand_19, 1);  expand_19 = None
        view_38: "f32[2048, 1]" = torch.ops.aten.view.default(div_9, [2048, 1]);  div_9 = None
        mul_28: "f32[2048, 119]" = torch.ops.aten.mul.Tensor(view_38, slice_34);  view_38 = slice_34 = None
        mul_29: "f32[2048, 119]" = torch.ops.aten.mul.Tensor(mul_28, -1.0);  mul_28 = None
        add_9: "f32[2048, 119]" = torch.ops.aten.add.Tensor(mul_27, mul_29);  mul_27 = mul_29 = None
        slice_scatter_8: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_7, add_9, 1, 9, 9223372036854775807);  slice_scatter_7 = add_9 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_88: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 10)
        select_89: "f32[]" = torch.ops.aten.select.int(select_88, 0, 10);  select_88 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_90: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_8, 1, 10)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_20: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_10: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_91: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_8, 1, 10)
        view_40: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_91, [8, 128, 2]);  select_91 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_10: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_20, view_40, permute_10, alpha = -2.0);  unsqueeze_20 = view_40 = permute_10 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_10: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_10, -1);  baddbmm_10 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_21: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_10, -1)
        expand_20: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_21, [8, 128, 2]);  unsqueeze_21 = None
        gather_10: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_20);  expand_20 = None
        view_41: "f32[2048]" = torch.ops.aten.view.default(gather_10, [-1]);  gather_10 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_93: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_9, 1, 10)
        copy_10: "i64[8, 128]" = torch.ops.aten.copy.default(select_93, argmin_10);  select_93 = argmin_10 = None
        select_scatter_10: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_9, copy_10, 1, 10);  select_scatter_9 = copy_10 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_10: "f32[2048]" = torch.ops.aten.sub.Tensor(select_90, view_41);  select_90 = view_41 = None
        div_10: "f32[2048]" = torch.ops.aten.div.Tensor(sub_10, select_89);  sub_10 = select_89 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_95: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 10)
        slice_38: "f32[118]" = torch.ops.aten.slice.Tensor(select_95, 0, 10, 9223372036854775807);  select_95 = None
        slice_39: "f32[2048, 118]" = torch.ops.aten.slice.Tensor(slice_scatter_8, 1, 10, 9223372036854775807)
        expand_21: "f32[2048, 118]" = torch.ops.aten.expand.default(slice_39, [2048, 118]);  slice_39 = None
        mul_30: "f32[2048, 118]" = torch.ops.aten.mul.Tensor(expand_21, 1);  expand_21 = None
        view_42: "f32[2048, 1]" = torch.ops.aten.view.default(div_10, [2048, 1]);  div_10 = None
        mul_31: "f32[2048, 118]" = torch.ops.aten.mul.Tensor(view_42, slice_38);  view_42 = slice_38 = None
        mul_32: "f32[2048, 118]" = torch.ops.aten.mul.Tensor(mul_31, -1.0);  mul_31 = None
        add_10: "f32[2048, 118]" = torch.ops.aten.add.Tensor(mul_30, mul_32);  mul_30 = mul_32 = None
        slice_scatter_9: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_8, add_10, 1, 10, 9223372036854775807);  slice_scatter_8 = add_10 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_97: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 11)
        select_98: "f32[]" = torch.ops.aten.select.int(select_97, 0, 11);  select_97 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_99: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_9, 1, 11)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_22: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_11: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_100: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_9, 1, 11)
        view_44: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_100, [8, 128, 2]);  select_100 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_11: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_22, view_44, permute_11, alpha = -2.0);  unsqueeze_22 = view_44 = permute_11 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_11: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_11, -1);  baddbmm_11 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_23: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_11, -1)
        expand_22: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_23, [8, 128, 2]);  unsqueeze_23 = None
        gather_11: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_22);  expand_22 = None
        view_45: "f32[2048]" = torch.ops.aten.view.default(gather_11, [-1]);  gather_11 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_102: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_10, 1, 11)
        copy_11: "i64[8, 128]" = torch.ops.aten.copy.default(select_102, argmin_11);  select_102 = argmin_11 = None
        select_scatter_11: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_10, copy_11, 1, 11);  select_scatter_10 = copy_11 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_11: "f32[2048]" = torch.ops.aten.sub.Tensor(select_99, view_45);  select_99 = view_45 = None
        div_11: "f32[2048]" = torch.ops.aten.div.Tensor(sub_11, select_98);  sub_11 = select_98 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_104: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 11)
        slice_42: "f32[117]" = torch.ops.aten.slice.Tensor(select_104, 0, 11, 9223372036854775807);  select_104 = None
        slice_43: "f32[2048, 117]" = torch.ops.aten.slice.Tensor(slice_scatter_9, 1, 11, 9223372036854775807)
        expand_23: "f32[2048, 117]" = torch.ops.aten.expand.default(slice_43, [2048, 117]);  slice_43 = None
        mul_33: "f32[2048, 117]" = torch.ops.aten.mul.Tensor(expand_23, 1);  expand_23 = None
        view_46: "f32[2048, 1]" = torch.ops.aten.view.default(div_11, [2048, 1]);  div_11 = None
        mul_34: "f32[2048, 117]" = torch.ops.aten.mul.Tensor(view_46, slice_42);  view_46 = slice_42 = None
        mul_35: "f32[2048, 117]" = torch.ops.aten.mul.Tensor(mul_34, -1.0);  mul_34 = None
        add_11: "f32[2048, 117]" = torch.ops.aten.add.Tensor(mul_33, mul_35);  mul_33 = mul_35 = None
        slice_scatter_10: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_9, add_11, 1, 11, 9223372036854775807);  slice_scatter_9 = add_11 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_106: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 12)
        select_107: "f32[]" = torch.ops.aten.select.int(select_106, 0, 12);  select_106 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_108: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_10, 1, 12)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_24: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_12: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_109: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_10, 1, 12)
        view_48: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_109, [8, 128, 2]);  select_109 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_12: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_24, view_48, permute_12, alpha = -2.0);  unsqueeze_24 = view_48 = permute_12 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_12: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_12, -1);  baddbmm_12 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_25: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_12, -1)
        expand_24: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_25, [8, 128, 2]);  unsqueeze_25 = None
        gather_12: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_24);  expand_24 = None
        view_49: "f32[2048]" = torch.ops.aten.view.default(gather_12, [-1]);  gather_12 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_111: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_11, 1, 12)
        copy_12: "i64[8, 128]" = torch.ops.aten.copy.default(select_111, argmin_12);  select_111 = argmin_12 = None
        select_scatter_12: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_11, copy_12, 1, 12);  select_scatter_11 = copy_12 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_12: "f32[2048]" = torch.ops.aten.sub.Tensor(select_108, view_49);  select_108 = view_49 = None
        div_12: "f32[2048]" = torch.ops.aten.div.Tensor(sub_12, select_107);  sub_12 = select_107 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_113: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 12)
        slice_46: "f32[116]" = torch.ops.aten.slice.Tensor(select_113, 0, 12, 9223372036854775807);  select_113 = None
        slice_47: "f32[2048, 116]" = torch.ops.aten.slice.Tensor(slice_scatter_10, 1, 12, 9223372036854775807)
        expand_25: "f32[2048, 116]" = torch.ops.aten.expand.default(slice_47, [2048, 116]);  slice_47 = None
        mul_36: "f32[2048, 116]" = torch.ops.aten.mul.Tensor(expand_25, 1);  expand_25 = None
        view_50: "f32[2048, 1]" = torch.ops.aten.view.default(div_12, [2048, 1]);  div_12 = None
        mul_37: "f32[2048, 116]" = torch.ops.aten.mul.Tensor(view_50, slice_46);  view_50 = slice_46 = None
        mul_38: "f32[2048, 116]" = torch.ops.aten.mul.Tensor(mul_37, -1.0);  mul_37 = None
        add_12: "f32[2048, 116]" = torch.ops.aten.add.Tensor(mul_36, mul_38);  mul_36 = mul_38 = None
        slice_scatter_11: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_10, add_12, 1, 12, 9223372036854775807);  slice_scatter_10 = add_12 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_115: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 13)
        select_116: "f32[]" = torch.ops.aten.select.int(select_115, 0, 13);  select_115 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_117: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_11, 1, 13)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_26: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_13: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_118: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_11, 1, 13)
        view_52: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_118, [8, 128, 2]);  select_118 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_13: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_26, view_52, permute_13, alpha = -2.0);  unsqueeze_26 = view_52 = permute_13 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_13: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_13, -1);  baddbmm_13 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_27: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_13, -1)
        expand_26: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_27, [8, 128, 2]);  unsqueeze_27 = None
        gather_13: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_26);  expand_26 = None
        view_53: "f32[2048]" = torch.ops.aten.view.default(gather_13, [-1]);  gather_13 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_120: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_12, 1, 13)
        copy_13: "i64[8, 128]" = torch.ops.aten.copy.default(select_120, argmin_13);  select_120 = argmin_13 = None
        select_scatter_13: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_12, copy_13, 1, 13);  select_scatter_12 = copy_13 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_13: "f32[2048]" = torch.ops.aten.sub.Tensor(select_117, view_53);  select_117 = view_53 = None
        div_13: "f32[2048]" = torch.ops.aten.div.Tensor(sub_13, select_116);  sub_13 = select_116 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_122: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 13)
        slice_50: "f32[115]" = torch.ops.aten.slice.Tensor(select_122, 0, 13, 9223372036854775807);  select_122 = None
        slice_51: "f32[2048, 115]" = torch.ops.aten.slice.Tensor(slice_scatter_11, 1, 13, 9223372036854775807)
        expand_27: "f32[2048, 115]" = torch.ops.aten.expand.default(slice_51, [2048, 115]);  slice_51 = None
        mul_39: "f32[2048, 115]" = torch.ops.aten.mul.Tensor(expand_27, 1);  expand_27 = None
        view_54: "f32[2048, 1]" = torch.ops.aten.view.default(div_13, [2048, 1]);  div_13 = None
        mul_40: "f32[2048, 115]" = torch.ops.aten.mul.Tensor(view_54, slice_50);  view_54 = slice_50 = None
        mul_41: "f32[2048, 115]" = torch.ops.aten.mul.Tensor(mul_40, -1.0);  mul_40 = None
        add_13: "f32[2048, 115]" = torch.ops.aten.add.Tensor(mul_39, mul_41);  mul_39 = mul_41 = None
        slice_scatter_12: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_11, add_13, 1, 13, 9223372036854775807);  slice_scatter_11 = add_13 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_124: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 14)
        select_125: "f32[]" = torch.ops.aten.select.int(select_124, 0, 14);  select_124 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_126: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_12, 1, 14)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_28: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_14: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_127: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_12, 1, 14)
        view_56: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_127, [8, 128, 2]);  select_127 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_14: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_28, view_56, permute_14, alpha = -2.0);  unsqueeze_28 = view_56 = permute_14 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_14: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_14, -1);  baddbmm_14 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_29: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_14, -1)
        expand_28: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_29, [8, 128, 2]);  unsqueeze_29 = None
        gather_14: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_28);  expand_28 = None
        view_57: "f32[2048]" = torch.ops.aten.view.default(gather_14, [-1]);  gather_14 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_129: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_13, 1, 14)
        copy_14: "i64[8, 128]" = torch.ops.aten.copy.default(select_129, argmin_14);  select_129 = argmin_14 = None
        select_scatter_14: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_13, copy_14, 1, 14);  select_scatter_13 = copy_14 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_14: "f32[2048]" = torch.ops.aten.sub.Tensor(select_126, view_57);  select_126 = view_57 = None
        div_14: "f32[2048]" = torch.ops.aten.div.Tensor(sub_14, select_125);  sub_14 = select_125 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_131: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 14)
        slice_54: "f32[114]" = torch.ops.aten.slice.Tensor(select_131, 0, 14, 9223372036854775807);  select_131 = None
        slice_55: "f32[2048, 114]" = torch.ops.aten.slice.Tensor(slice_scatter_12, 1, 14, 9223372036854775807)
        expand_29: "f32[2048, 114]" = torch.ops.aten.expand.default(slice_55, [2048, 114]);  slice_55 = None
        mul_42: "f32[2048, 114]" = torch.ops.aten.mul.Tensor(expand_29, 1);  expand_29 = None
        view_58: "f32[2048, 1]" = torch.ops.aten.view.default(div_14, [2048, 1]);  div_14 = None
        mul_43: "f32[2048, 114]" = torch.ops.aten.mul.Tensor(view_58, slice_54);  view_58 = slice_54 = None
        mul_44: "f32[2048, 114]" = torch.ops.aten.mul.Tensor(mul_43, -1.0);  mul_43 = None
        add_14: "f32[2048, 114]" = torch.ops.aten.add.Tensor(mul_42, mul_44);  mul_42 = mul_44 = None
        slice_scatter_13: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_12, add_14, 1, 14, 9223372036854775807);  slice_scatter_12 = add_14 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_133: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 15)
        select_134: "f32[]" = torch.ops.aten.select.int(select_133, 0, 15);  select_133 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_135: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_13, 1, 15)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_30: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_15: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_136: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_13, 1, 15)
        view_60: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_136, [8, 128, 2]);  select_136 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_15: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_30, view_60, permute_15, alpha = -2.0);  unsqueeze_30 = view_60 = permute_15 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_15: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_15, -1);  baddbmm_15 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_31: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_15, -1)
        expand_30: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_31, [8, 128, 2]);  unsqueeze_31 = None
        gather_15: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_30);  expand_30 = None
        view_61: "f32[2048]" = torch.ops.aten.view.default(gather_15, [-1]);  gather_15 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_138: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_14, 1, 15)
        copy_15: "i64[8, 128]" = torch.ops.aten.copy.default(select_138, argmin_15);  select_138 = argmin_15 = None
        select_scatter_15: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_14, copy_15, 1, 15);  select_scatter_14 = copy_15 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_15: "f32[2048]" = torch.ops.aten.sub.Tensor(select_135, view_61);  select_135 = view_61 = None
        div_15: "f32[2048]" = torch.ops.aten.div.Tensor(sub_15, select_134);  sub_15 = select_134 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_140: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 15)
        slice_58: "f32[113]" = torch.ops.aten.slice.Tensor(select_140, 0, 15, 9223372036854775807);  select_140 = None
        slice_59: "f32[2048, 113]" = torch.ops.aten.slice.Tensor(slice_scatter_13, 1, 15, 9223372036854775807)
        expand_31: "f32[2048, 113]" = torch.ops.aten.expand.default(slice_59, [2048, 113]);  slice_59 = None
        mul_45: "f32[2048, 113]" = torch.ops.aten.mul.Tensor(expand_31, 1);  expand_31 = None
        view_62: "f32[2048, 1]" = torch.ops.aten.view.default(div_15, [2048, 1]);  div_15 = None
        mul_46: "f32[2048, 113]" = torch.ops.aten.mul.Tensor(view_62, slice_58);  view_62 = slice_58 = None
        mul_47: "f32[2048, 113]" = torch.ops.aten.mul.Tensor(mul_46, -1.0);  mul_46 = None
        add_15: "f32[2048, 113]" = torch.ops.aten.add.Tensor(mul_45, mul_47);  mul_45 = mul_47 = None
        slice_scatter_14: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_13, add_15, 1, 15, 9223372036854775807);  slice_scatter_13 = add_15 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_142: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 16)
        select_143: "f32[]" = torch.ops.aten.select.int(select_142, 0, 16);  select_142 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_144: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_14, 1, 16)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_32: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_16: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_145: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_14, 1, 16)
        view_64: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_145, [8, 128, 2]);  select_145 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_16: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_32, view_64, permute_16, alpha = -2.0);  unsqueeze_32 = view_64 = permute_16 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_16: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_16, -1);  baddbmm_16 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_33: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_16, -1)
        expand_32: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_33, [8, 128, 2]);  unsqueeze_33 = None
        gather_16: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_32);  expand_32 = None
        view_65: "f32[2048]" = torch.ops.aten.view.default(gather_16, [-1]);  gather_16 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_147: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_15, 1, 16)
        copy_16: "i64[8, 128]" = torch.ops.aten.copy.default(select_147, argmin_16);  select_147 = argmin_16 = None
        select_scatter_16: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_15, copy_16, 1, 16);  select_scatter_15 = copy_16 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_16: "f32[2048]" = torch.ops.aten.sub.Tensor(select_144, view_65);  select_144 = view_65 = None
        div_16: "f32[2048]" = torch.ops.aten.div.Tensor(sub_16, select_143);  sub_16 = select_143 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_149: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 16)
        slice_62: "f32[112]" = torch.ops.aten.slice.Tensor(select_149, 0, 16, 9223372036854775807);  select_149 = None
        slice_63: "f32[2048, 112]" = torch.ops.aten.slice.Tensor(slice_scatter_14, 1, 16, 9223372036854775807)
        expand_33: "f32[2048, 112]" = torch.ops.aten.expand.default(slice_63, [2048, 112]);  slice_63 = None
        mul_48: "f32[2048, 112]" = torch.ops.aten.mul.Tensor(expand_33, 1);  expand_33 = None
        view_66: "f32[2048, 1]" = torch.ops.aten.view.default(div_16, [2048, 1]);  div_16 = None
        mul_49: "f32[2048, 112]" = torch.ops.aten.mul.Tensor(view_66, slice_62);  view_66 = slice_62 = None
        mul_50: "f32[2048, 112]" = torch.ops.aten.mul.Tensor(mul_49, -1.0);  mul_49 = None
        add_16: "f32[2048, 112]" = torch.ops.aten.add.Tensor(mul_48, mul_50);  mul_48 = mul_50 = None
        slice_scatter_15: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_14, add_16, 1, 16, 9223372036854775807);  slice_scatter_14 = add_16 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_151: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 17)
        select_152: "f32[]" = torch.ops.aten.select.int(select_151, 0, 17);  select_151 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_153: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_15, 1, 17)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_34: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_17: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_154: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_15, 1, 17)
        view_68: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_154, [8, 128, 2]);  select_154 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_17: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_34, view_68, permute_17, alpha = -2.0);  unsqueeze_34 = view_68 = permute_17 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_17: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_17, -1);  baddbmm_17 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_35: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_17, -1)
        expand_34: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_35, [8, 128, 2]);  unsqueeze_35 = None
        gather_17: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_34);  expand_34 = None
        view_69: "f32[2048]" = torch.ops.aten.view.default(gather_17, [-1]);  gather_17 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_156: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_16, 1, 17)
        copy_17: "i64[8, 128]" = torch.ops.aten.copy.default(select_156, argmin_17);  select_156 = argmin_17 = None
        select_scatter_17: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_16, copy_17, 1, 17);  select_scatter_16 = copy_17 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_17: "f32[2048]" = torch.ops.aten.sub.Tensor(select_153, view_69);  select_153 = view_69 = None
        div_17: "f32[2048]" = torch.ops.aten.div.Tensor(sub_17, select_152);  sub_17 = select_152 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_158: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 17)
        slice_66: "f32[111]" = torch.ops.aten.slice.Tensor(select_158, 0, 17, 9223372036854775807);  select_158 = None
        slice_67: "f32[2048, 111]" = torch.ops.aten.slice.Tensor(slice_scatter_15, 1, 17, 9223372036854775807)
        expand_35: "f32[2048, 111]" = torch.ops.aten.expand.default(slice_67, [2048, 111]);  slice_67 = None
        mul_51: "f32[2048, 111]" = torch.ops.aten.mul.Tensor(expand_35, 1);  expand_35 = None
        view_70: "f32[2048, 1]" = torch.ops.aten.view.default(div_17, [2048, 1]);  div_17 = None
        mul_52: "f32[2048, 111]" = torch.ops.aten.mul.Tensor(view_70, slice_66);  view_70 = slice_66 = None
        mul_53: "f32[2048, 111]" = torch.ops.aten.mul.Tensor(mul_52, -1.0);  mul_52 = None
        add_17: "f32[2048, 111]" = torch.ops.aten.add.Tensor(mul_51, mul_53);  mul_51 = mul_53 = None
        slice_scatter_16: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_15, add_17, 1, 17, 9223372036854775807);  slice_scatter_15 = add_17 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_160: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 18)
        select_161: "f32[]" = torch.ops.aten.select.int(select_160, 0, 18);  select_160 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_162: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_16, 1, 18)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_36: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_18: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_163: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_16, 1, 18)
        view_72: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_163, [8, 128, 2]);  select_163 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_18: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_36, view_72, permute_18, alpha = -2.0);  unsqueeze_36 = view_72 = permute_18 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_18: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_18, -1);  baddbmm_18 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_37: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_18, -1)
        expand_36: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_37, [8, 128, 2]);  unsqueeze_37 = None
        gather_18: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_36);  expand_36 = None
        view_73: "f32[2048]" = torch.ops.aten.view.default(gather_18, [-1]);  gather_18 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_165: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_17, 1, 18)
        copy_18: "i64[8, 128]" = torch.ops.aten.copy.default(select_165, argmin_18);  select_165 = argmin_18 = None
        select_scatter_18: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_17, copy_18, 1, 18);  select_scatter_17 = copy_18 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_18: "f32[2048]" = torch.ops.aten.sub.Tensor(select_162, view_73);  select_162 = view_73 = None
        div_18: "f32[2048]" = torch.ops.aten.div.Tensor(sub_18, select_161);  sub_18 = select_161 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_167: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 18)
        slice_70: "f32[110]" = torch.ops.aten.slice.Tensor(select_167, 0, 18, 9223372036854775807);  select_167 = None
        slice_71: "f32[2048, 110]" = torch.ops.aten.slice.Tensor(slice_scatter_16, 1, 18, 9223372036854775807)
        expand_37: "f32[2048, 110]" = torch.ops.aten.expand.default(slice_71, [2048, 110]);  slice_71 = None
        mul_54: "f32[2048, 110]" = torch.ops.aten.mul.Tensor(expand_37, 1);  expand_37 = None
        view_74: "f32[2048, 1]" = torch.ops.aten.view.default(div_18, [2048, 1]);  div_18 = None
        mul_55: "f32[2048, 110]" = torch.ops.aten.mul.Tensor(view_74, slice_70);  view_74 = slice_70 = None
        mul_56: "f32[2048, 110]" = torch.ops.aten.mul.Tensor(mul_55, -1.0);  mul_55 = None
        add_18: "f32[2048, 110]" = torch.ops.aten.add.Tensor(mul_54, mul_56);  mul_54 = mul_56 = None
        slice_scatter_17: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_16, add_18, 1, 18, 9223372036854775807);  slice_scatter_16 = add_18 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_169: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 19)
        select_170: "f32[]" = torch.ops.aten.select.int(select_169, 0, 19);  select_169 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_171: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_17, 1, 19)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_38: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_19: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_172: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_17, 1, 19)
        view_76: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_172, [8, 128, 2]);  select_172 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_19: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_38, view_76, permute_19, alpha = -2.0);  unsqueeze_38 = view_76 = permute_19 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_19: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_19, -1);  baddbmm_19 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_39: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_19, -1)
        expand_38: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_39, [8, 128, 2]);  unsqueeze_39 = None
        gather_19: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_38);  expand_38 = None
        view_77: "f32[2048]" = torch.ops.aten.view.default(gather_19, [-1]);  gather_19 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_174: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_18, 1, 19)
        copy_19: "i64[8, 128]" = torch.ops.aten.copy.default(select_174, argmin_19);  select_174 = argmin_19 = None
        select_scatter_19: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_18, copy_19, 1, 19);  select_scatter_18 = copy_19 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_19: "f32[2048]" = torch.ops.aten.sub.Tensor(select_171, view_77);  select_171 = view_77 = None
        div_19: "f32[2048]" = torch.ops.aten.div.Tensor(sub_19, select_170);  sub_19 = select_170 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_176: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 19)
        slice_74: "f32[109]" = torch.ops.aten.slice.Tensor(select_176, 0, 19, 9223372036854775807);  select_176 = None
        slice_75: "f32[2048, 109]" = torch.ops.aten.slice.Tensor(slice_scatter_17, 1, 19, 9223372036854775807)
        expand_39: "f32[2048, 109]" = torch.ops.aten.expand.default(slice_75, [2048, 109]);  slice_75 = None
        mul_57: "f32[2048, 109]" = torch.ops.aten.mul.Tensor(expand_39, 1);  expand_39 = None
        view_78: "f32[2048, 1]" = torch.ops.aten.view.default(div_19, [2048, 1]);  div_19 = None
        mul_58: "f32[2048, 109]" = torch.ops.aten.mul.Tensor(view_78, slice_74);  view_78 = slice_74 = None
        mul_59: "f32[2048, 109]" = torch.ops.aten.mul.Tensor(mul_58, -1.0);  mul_58 = None
        add_19: "f32[2048, 109]" = torch.ops.aten.add.Tensor(mul_57, mul_59);  mul_57 = mul_59 = None
        slice_scatter_18: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_17, add_19, 1, 19, 9223372036854775807);  slice_scatter_17 = add_19 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_178: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 20)
        select_179: "f32[]" = torch.ops.aten.select.int(select_178, 0, 20);  select_178 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_180: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_18, 1, 20)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_40: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_20: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_181: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_18, 1, 20)
        view_80: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_181, [8, 128, 2]);  select_181 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_20: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_40, view_80, permute_20, alpha = -2.0);  unsqueeze_40 = view_80 = permute_20 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_20: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_20, -1);  baddbmm_20 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_41: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_20, -1)
        expand_40: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_41, [8, 128, 2]);  unsqueeze_41 = None
        gather_20: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_40);  expand_40 = None
        view_81: "f32[2048]" = torch.ops.aten.view.default(gather_20, [-1]);  gather_20 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_183: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_19, 1, 20)
        copy_20: "i64[8, 128]" = torch.ops.aten.copy.default(select_183, argmin_20);  select_183 = argmin_20 = None
        select_scatter_20: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_19, copy_20, 1, 20);  select_scatter_19 = copy_20 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_20: "f32[2048]" = torch.ops.aten.sub.Tensor(select_180, view_81);  select_180 = view_81 = None
        div_20: "f32[2048]" = torch.ops.aten.div.Tensor(sub_20, select_179);  sub_20 = select_179 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_185: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 20)
        slice_78: "f32[108]" = torch.ops.aten.slice.Tensor(select_185, 0, 20, 9223372036854775807);  select_185 = None
        slice_79: "f32[2048, 108]" = torch.ops.aten.slice.Tensor(slice_scatter_18, 1, 20, 9223372036854775807)
        expand_41: "f32[2048, 108]" = torch.ops.aten.expand.default(slice_79, [2048, 108]);  slice_79 = None
        mul_60: "f32[2048, 108]" = torch.ops.aten.mul.Tensor(expand_41, 1);  expand_41 = None
        view_82: "f32[2048, 1]" = torch.ops.aten.view.default(div_20, [2048, 1]);  div_20 = None
        mul_61: "f32[2048, 108]" = torch.ops.aten.mul.Tensor(view_82, slice_78);  view_82 = slice_78 = None
        mul_62: "f32[2048, 108]" = torch.ops.aten.mul.Tensor(mul_61, -1.0);  mul_61 = None
        add_20: "f32[2048, 108]" = torch.ops.aten.add.Tensor(mul_60, mul_62);  mul_60 = mul_62 = None
        slice_scatter_19: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_18, add_20, 1, 20, 9223372036854775807);  slice_scatter_18 = add_20 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_187: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 21)
        select_188: "f32[]" = torch.ops.aten.select.int(select_187, 0, 21);  select_187 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_189: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_19, 1, 21)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_42: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_21: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_190: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_19, 1, 21)
        view_84: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_190, [8, 128, 2]);  select_190 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_21: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_42, view_84, permute_21, alpha = -2.0);  unsqueeze_42 = view_84 = permute_21 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_21: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_21, -1);  baddbmm_21 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_43: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_21, -1)
        expand_42: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_43, [8, 128, 2]);  unsqueeze_43 = None
        gather_21: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_42);  expand_42 = None
        view_85: "f32[2048]" = torch.ops.aten.view.default(gather_21, [-1]);  gather_21 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_192: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_20, 1, 21)
        copy_21: "i64[8, 128]" = torch.ops.aten.copy.default(select_192, argmin_21);  select_192 = argmin_21 = None
        select_scatter_21: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_20, copy_21, 1, 21);  select_scatter_20 = copy_21 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_21: "f32[2048]" = torch.ops.aten.sub.Tensor(select_189, view_85);  select_189 = view_85 = None
        div_21: "f32[2048]" = torch.ops.aten.div.Tensor(sub_21, select_188);  sub_21 = select_188 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_194: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 21)
        slice_82: "f32[107]" = torch.ops.aten.slice.Tensor(select_194, 0, 21, 9223372036854775807);  select_194 = None
        slice_83: "f32[2048, 107]" = torch.ops.aten.slice.Tensor(slice_scatter_19, 1, 21, 9223372036854775807)
        expand_43: "f32[2048, 107]" = torch.ops.aten.expand.default(slice_83, [2048, 107]);  slice_83 = None
        mul_63: "f32[2048, 107]" = torch.ops.aten.mul.Tensor(expand_43, 1);  expand_43 = None
        view_86: "f32[2048, 1]" = torch.ops.aten.view.default(div_21, [2048, 1]);  div_21 = None
        mul_64: "f32[2048, 107]" = torch.ops.aten.mul.Tensor(view_86, slice_82);  view_86 = slice_82 = None
        mul_65: "f32[2048, 107]" = torch.ops.aten.mul.Tensor(mul_64, -1.0);  mul_64 = None
        add_21: "f32[2048, 107]" = torch.ops.aten.add.Tensor(mul_63, mul_65);  mul_63 = mul_65 = None
        slice_scatter_20: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_19, add_21, 1, 21, 9223372036854775807);  slice_scatter_19 = add_21 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_196: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 22)
        select_197: "f32[]" = torch.ops.aten.select.int(select_196, 0, 22);  select_196 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_198: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_20, 1, 22)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_44: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_22: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_199: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_20, 1, 22)
        view_88: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_199, [8, 128, 2]);  select_199 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_22: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_44, view_88, permute_22, alpha = -2.0);  unsqueeze_44 = view_88 = permute_22 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_22: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_22, -1);  baddbmm_22 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_45: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_22, -1)
        expand_44: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_45, [8, 128, 2]);  unsqueeze_45 = None
        gather_22: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_44);  expand_44 = None
        view_89: "f32[2048]" = torch.ops.aten.view.default(gather_22, [-1]);  gather_22 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_201: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_21, 1, 22)
        copy_22: "i64[8, 128]" = torch.ops.aten.copy.default(select_201, argmin_22);  select_201 = argmin_22 = None
        select_scatter_22: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_21, copy_22, 1, 22);  select_scatter_21 = copy_22 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_22: "f32[2048]" = torch.ops.aten.sub.Tensor(select_198, view_89);  select_198 = view_89 = None
        div_22: "f32[2048]" = torch.ops.aten.div.Tensor(sub_22, select_197);  sub_22 = select_197 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_203: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 22)
        slice_86: "f32[106]" = torch.ops.aten.slice.Tensor(select_203, 0, 22, 9223372036854775807);  select_203 = None
        slice_87: "f32[2048, 106]" = torch.ops.aten.slice.Tensor(slice_scatter_20, 1, 22, 9223372036854775807)
        expand_45: "f32[2048, 106]" = torch.ops.aten.expand.default(slice_87, [2048, 106]);  slice_87 = None
        mul_66: "f32[2048, 106]" = torch.ops.aten.mul.Tensor(expand_45, 1);  expand_45 = None
        view_90: "f32[2048, 1]" = torch.ops.aten.view.default(div_22, [2048, 1]);  div_22 = None
        mul_67: "f32[2048, 106]" = torch.ops.aten.mul.Tensor(view_90, slice_86);  view_90 = slice_86 = None
        mul_68: "f32[2048, 106]" = torch.ops.aten.mul.Tensor(mul_67, -1.0);  mul_67 = None
        add_22: "f32[2048, 106]" = torch.ops.aten.add.Tensor(mul_66, mul_68);  mul_66 = mul_68 = None
        slice_scatter_21: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_20, add_22, 1, 22, 9223372036854775807);  slice_scatter_20 = add_22 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_205: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 23)
        select_206: "f32[]" = torch.ops.aten.select.int(select_205, 0, 23);  select_205 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_207: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_21, 1, 23)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_46: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_23: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_208: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_21, 1, 23)
        view_92: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_208, [8, 128, 2]);  select_208 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_23: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_46, view_92, permute_23, alpha = -2.0);  unsqueeze_46 = view_92 = permute_23 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_23: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_23, -1);  baddbmm_23 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_47: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_23, -1)
        expand_46: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_47, [8, 128, 2]);  unsqueeze_47 = None
        gather_23: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_46);  expand_46 = None
        view_93: "f32[2048]" = torch.ops.aten.view.default(gather_23, [-1]);  gather_23 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_210: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_22, 1, 23)
        copy_23: "i64[8, 128]" = torch.ops.aten.copy.default(select_210, argmin_23);  select_210 = argmin_23 = None
        select_scatter_23: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_22, copy_23, 1, 23);  select_scatter_22 = copy_23 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_23: "f32[2048]" = torch.ops.aten.sub.Tensor(select_207, view_93);  select_207 = view_93 = None
        div_23: "f32[2048]" = torch.ops.aten.div.Tensor(sub_23, select_206);  sub_23 = select_206 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_212: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 23)
        slice_90: "f32[105]" = torch.ops.aten.slice.Tensor(select_212, 0, 23, 9223372036854775807);  select_212 = None
        slice_91: "f32[2048, 105]" = torch.ops.aten.slice.Tensor(slice_scatter_21, 1, 23, 9223372036854775807)
        expand_47: "f32[2048, 105]" = torch.ops.aten.expand.default(slice_91, [2048, 105]);  slice_91 = None
        mul_69: "f32[2048, 105]" = torch.ops.aten.mul.Tensor(expand_47, 1);  expand_47 = None
        view_94: "f32[2048, 1]" = torch.ops.aten.view.default(div_23, [2048, 1]);  div_23 = None
        mul_70: "f32[2048, 105]" = torch.ops.aten.mul.Tensor(view_94, slice_90);  view_94 = slice_90 = None
        mul_71: "f32[2048, 105]" = torch.ops.aten.mul.Tensor(mul_70, -1.0);  mul_70 = None
        add_23: "f32[2048, 105]" = torch.ops.aten.add.Tensor(mul_69, mul_71);  mul_69 = mul_71 = None
        slice_scatter_22: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_21, add_23, 1, 23, 9223372036854775807);  slice_scatter_21 = add_23 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_214: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 24)
        select_215: "f32[]" = torch.ops.aten.select.int(select_214, 0, 24);  select_214 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_216: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_22, 1, 24)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_48: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_24: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_217: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_22, 1, 24)
        view_96: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_217, [8, 128, 2]);  select_217 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_24: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_48, view_96, permute_24, alpha = -2.0);  unsqueeze_48 = view_96 = permute_24 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_24: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_24, -1);  baddbmm_24 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_49: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_24, -1)
        expand_48: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_49, [8, 128, 2]);  unsqueeze_49 = None
        gather_24: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_48);  expand_48 = None
        view_97: "f32[2048]" = torch.ops.aten.view.default(gather_24, [-1]);  gather_24 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_219: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_23, 1, 24)
        copy_24: "i64[8, 128]" = torch.ops.aten.copy.default(select_219, argmin_24);  select_219 = argmin_24 = None
        select_scatter_24: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_23, copy_24, 1, 24);  select_scatter_23 = copy_24 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_24: "f32[2048]" = torch.ops.aten.sub.Tensor(select_216, view_97);  select_216 = view_97 = None
        div_24: "f32[2048]" = torch.ops.aten.div.Tensor(sub_24, select_215);  sub_24 = select_215 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_221: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 24)
        slice_94: "f32[104]" = torch.ops.aten.slice.Tensor(select_221, 0, 24, 9223372036854775807);  select_221 = None
        slice_95: "f32[2048, 104]" = torch.ops.aten.slice.Tensor(slice_scatter_22, 1, 24, 9223372036854775807)
        expand_49: "f32[2048, 104]" = torch.ops.aten.expand.default(slice_95, [2048, 104]);  slice_95 = None
        mul_72: "f32[2048, 104]" = torch.ops.aten.mul.Tensor(expand_49, 1);  expand_49 = None
        view_98: "f32[2048, 1]" = torch.ops.aten.view.default(div_24, [2048, 1]);  div_24 = None
        mul_73: "f32[2048, 104]" = torch.ops.aten.mul.Tensor(view_98, slice_94);  view_98 = slice_94 = None
        mul_74: "f32[2048, 104]" = torch.ops.aten.mul.Tensor(mul_73, -1.0);  mul_73 = None
        add_24: "f32[2048, 104]" = torch.ops.aten.add.Tensor(mul_72, mul_74);  mul_72 = mul_74 = None
        slice_scatter_23: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_22, add_24, 1, 24, 9223372036854775807);  slice_scatter_22 = add_24 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_223: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 25)
        select_224: "f32[]" = torch.ops.aten.select.int(select_223, 0, 25);  select_223 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_225: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_23, 1, 25)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_50: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_25: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_226: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_23, 1, 25)
        view_100: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_226, [8, 128, 2]);  select_226 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_25: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_50, view_100, permute_25, alpha = -2.0);  unsqueeze_50 = view_100 = permute_25 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_25: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_25, -1);  baddbmm_25 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_51: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_25, -1)
        expand_50: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_51, [8, 128, 2]);  unsqueeze_51 = None
        gather_25: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_50);  expand_50 = None
        view_101: "f32[2048]" = torch.ops.aten.view.default(gather_25, [-1]);  gather_25 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_228: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_24, 1, 25)
        copy_25: "i64[8, 128]" = torch.ops.aten.copy.default(select_228, argmin_25);  select_228 = argmin_25 = None
        select_scatter_25: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_24, copy_25, 1, 25);  select_scatter_24 = copy_25 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_25: "f32[2048]" = torch.ops.aten.sub.Tensor(select_225, view_101);  select_225 = view_101 = None
        div_25: "f32[2048]" = torch.ops.aten.div.Tensor(sub_25, select_224);  sub_25 = select_224 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_230: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 25)
        slice_98: "f32[103]" = torch.ops.aten.slice.Tensor(select_230, 0, 25, 9223372036854775807);  select_230 = None
        slice_99: "f32[2048, 103]" = torch.ops.aten.slice.Tensor(slice_scatter_23, 1, 25, 9223372036854775807)
        expand_51: "f32[2048, 103]" = torch.ops.aten.expand.default(slice_99, [2048, 103]);  slice_99 = None
        mul_75: "f32[2048, 103]" = torch.ops.aten.mul.Tensor(expand_51, 1);  expand_51 = None
        view_102: "f32[2048, 1]" = torch.ops.aten.view.default(div_25, [2048, 1]);  div_25 = None
        mul_76: "f32[2048, 103]" = torch.ops.aten.mul.Tensor(view_102, slice_98);  view_102 = slice_98 = None
        mul_77: "f32[2048, 103]" = torch.ops.aten.mul.Tensor(mul_76, -1.0);  mul_76 = None
        add_25: "f32[2048, 103]" = torch.ops.aten.add.Tensor(mul_75, mul_77);  mul_75 = mul_77 = None
        slice_scatter_24: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_23, add_25, 1, 25, 9223372036854775807);  slice_scatter_23 = add_25 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_232: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 26)
        select_233: "f32[]" = torch.ops.aten.select.int(select_232, 0, 26);  select_232 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_234: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_24, 1, 26)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_52: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_26: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_235: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_24, 1, 26)
        view_104: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_235, [8, 128, 2]);  select_235 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_26: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_52, view_104, permute_26, alpha = -2.0);  unsqueeze_52 = view_104 = permute_26 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_26: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_26, -1);  baddbmm_26 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_53: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_26, -1)
        expand_52: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_53, [8, 128, 2]);  unsqueeze_53 = None
        gather_26: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_52);  expand_52 = None
        view_105: "f32[2048]" = torch.ops.aten.view.default(gather_26, [-1]);  gather_26 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_237: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_25, 1, 26)
        copy_26: "i64[8, 128]" = torch.ops.aten.copy.default(select_237, argmin_26);  select_237 = argmin_26 = None
        select_scatter_26: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_25, copy_26, 1, 26);  select_scatter_25 = copy_26 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_26: "f32[2048]" = torch.ops.aten.sub.Tensor(select_234, view_105);  select_234 = view_105 = None
        div_26: "f32[2048]" = torch.ops.aten.div.Tensor(sub_26, select_233);  sub_26 = select_233 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_239: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 26)
        slice_102: "f32[102]" = torch.ops.aten.slice.Tensor(select_239, 0, 26, 9223372036854775807);  select_239 = None
        slice_103: "f32[2048, 102]" = torch.ops.aten.slice.Tensor(slice_scatter_24, 1, 26, 9223372036854775807)
        expand_53: "f32[2048, 102]" = torch.ops.aten.expand.default(slice_103, [2048, 102]);  slice_103 = None
        mul_78: "f32[2048, 102]" = torch.ops.aten.mul.Tensor(expand_53, 1);  expand_53 = None
        view_106: "f32[2048, 1]" = torch.ops.aten.view.default(div_26, [2048, 1]);  div_26 = None
        mul_79: "f32[2048, 102]" = torch.ops.aten.mul.Tensor(view_106, slice_102);  view_106 = slice_102 = None
        mul_80: "f32[2048, 102]" = torch.ops.aten.mul.Tensor(mul_79, -1.0);  mul_79 = None
        add_26: "f32[2048, 102]" = torch.ops.aten.add.Tensor(mul_78, mul_80);  mul_78 = mul_80 = None
        slice_scatter_25: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_24, add_26, 1, 26, 9223372036854775807);  slice_scatter_24 = add_26 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_241: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 27)
        select_242: "f32[]" = torch.ops.aten.select.int(select_241, 0, 27);  select_241 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_243: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_25, 1, 27)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_54: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_27: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_244: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_25, 1, 27)
        view_108: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_244, [8, 128, 2]);  select_244 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_27: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_54, view_108, permute_27, alpha = -2.0);  unsqueeze_54 = view_108 = permute_27 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_27: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_27, -1);  baddbmm_27 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_55: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_27, -1)
        expand_54: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_55, [8, 128, 2]);  unsqueeze_55 = None
        gather_27: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_54);  expand_54 = None
        view_109: "f32[2048]" = torch.ops.aten.view.default(gather_27, [-1]);  gather_27 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_246: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_26, 1, 27)
        copy_27: "i64[8, 128]" = torch.ops.aten.copy.default(select_246, argmin_27);  select_246 = argmin_27 = None
        select_scatter_27: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_26, copy_27, 1, 27);  select_scatter_26 = copy_27 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_27: "f32[2048]" = torch.ops.aten.sub.Tensor(select_243, view_109);  select_243 = view_109 = None
        div_27: "f32[2048]" = torch.ops.aten.div.Tensor(sub_27, select_242);  sub_27 = select_242 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_248: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 27)
        slice_106: "f32[101]" = torch.ops.aten.slice.Tensor(select_248, 0, 27, 9223372036854775807);  select_248 = None
        slice_107: "f32[2048, 101]" = torch.ops.aten.slice.Tensor(slice_scatter_25, 1, 27, 9223372036854775807)
        expand_55: "f32[2048, 101]" = torch.ops.aten.expand.default(slice_107, [2048, 101]);  slice_107 = None
        mul_81: "f32[2048, 101]" = torch.ops.aten.mul.Tensor(expand_55, 1);  expand_55 = None
        view_110: "f32[2048, 1]" = torch.ops.aten.view.default(div_27, [2048, 1]);  div_27 = None
        mul_82: "f32[2048, 101]" = torch.ops.aten.mul.Tensor(view_110, slice_106);  view_110 = slice_106 = None
        mul_83: "f32[2048, 101]" = torch.ops.aten.mul.Tensor(mul_82, -1.0);  mul_82 = None
        add_27: "f32[2048, 101]" = torch.ops.aten.add.Tensor(mul_81, mul_83);  mul_81 = mul_83 = None
        slice_scatter_26: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_25, add_27, 1, 27, 9223372036854775807);  slice_scatter_25 = add_27 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_250: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 28)
        select_251: "f32[]" = torch.ops.aten.select.int(select_250, 0, 28);  select_250 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_252: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_26, 1, 28)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_56: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_28: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_253: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_26, 1, 28)
        view_112: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_253, [8, 128, 2]);  select_253 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_28: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_56, view_112, permute_28, alpha = -2.0);  unsqueeze_56 = view_112 = permute_28 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_28: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_28, -1);  baddbmm_28 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_57: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_28, -1)
        expand_56: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_57, [8, 128, 2]);  unsqueeze_57 = None
        gather_28: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_56);  expand_56 = None
        view_113: "f32[2048]" = torch.ops.aten.view.default(gather_28, [-1]);  gather_28 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_255: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_27, 1, 28)
        copy_28: "i64[8, 128]" = torch.ops.aten.copy.default(select_255, argmin_28);  select_255 = argmin_28 = None
        select_scatter_28: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_27, copy_28, 1, 28);  select_scatter_27 = copy_28 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_28: "f32[2048]" = torch.ops.aten.sub.Tensor(select_252, view_113);  select_252 = view_113 = None
        div_28: "f32[2048]" = torch.ops.aten.div.Tensor(sub_28, select_251);  sub_28 = select_251 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_257: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 28)
        slice_110: "f32[100]" = torch.ops.aten.slice.Tensor(select_257, 0, 28, 9223372036854775807);  select_257 = None
        slice_111: "f32[2048, 100]" = torch.ops.aten.slice.Tensor(slice_scatter_26, 1, 28, 9223372036854775807)
        expand_57: "f32[2048, 100]" = torch.ops.aten.expand.default(slice_111, [2048, 100]);  slice_111 = None
        mul_84: "f32[2048, 100]" = torch.ops.aten.mul.Tensor(expand_57, 1);  expand_57 = None
        view_114: "f32[2048, 1]" = torch.ops.aten.view.default(div_28, [2048, 1]);  div_28 = None
        mul_85: "f32[2048, 100]" = torch.ops.aten.mul.Tensor(view_114, slice_110);  view_114 = slice_110 = None
        mul_86: "f32[2048, 100]" = torch.ops.aten.mul.Tensor(mul_85, -1.0);  mul_85 = None
        add_28: "f32[2048, 100]" = torch.ops.aten.add.Tensor(mul_84, mul_86);  mul_84 = mul_86 = None
        slice_scatter_27: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_26, add_28, 1, 28, 9223372036854775807);  slice_scatter_26 = add_28 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_259: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 29)
        select_260: "f32[]" = torch.ops.aten.select.int(select_259, 0, 29);  select_259 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_261: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_27, 1, 29)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_58: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_29: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_262: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_27, 1, 29)
        view_116: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_262, [8, 128, 2]);  select_262 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_29: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_58, view_116, permute_29, alpha = -2.0);  unsqueeze_58 = view_116 = permute_29 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_29: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_29, -1);  baddbmm_29 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_59: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_29, -1)
        expand_58: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_59, [8, 128, 2]);  unsqueeze_59 = None
        gather_29: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_58);  expand_58 = None
        view_117: "f32[2048]" = torch.ops.aten.view.default(gather_29, [-1]);  gather_29 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_264: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_28, 1, 29)
        copy_29: "i64[8, 128]" = torch.ops.aten.copy.default(select_264, argmin_29);  select_264 = argmin_29 = None
        select_scatter_29: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_28, copy_29, 1, 29);  select_scatter_28 = copy_29 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_29: "f32[2048]" = torch.ops.aten.sub.Tensor(select_261, view_117);  select_261 = view_117 = None
        div_29: "f32[2048]" = torch.ops.aten.div.Tensor(sub_29, select_260);  sub_29 = select_260 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_266: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 29)
        slice_114: "f32[99]" = torch.ops.aten.slice.Tensor(select_266, 0, 29, 9223372036854775807);  select_266 = None
        slice_115: "f32[2048, 99]" = torch.ops.aten.slice.Tensor(slice_scatter_27, 1, 29, 9223372036854775807)
        expand_59: "f32[2048, 99]" = torch.ops.aten.expand.default(slice_115, [2048, 99]);  slice_115 = None
        mul_87: "f32[2048, 99]" = torch.ops.aten.mul.Tensor(expand_59, 1);  expand_59 = None
        view_118: "f32[2048, 1]" = torch.ops.aten.view.default(div_29, [2048, 1]);  div_29 = None
        mul_88: "f32[2048, 99]" = torch.ops.aten.mul.Tensor(view_118, slice_114);  view_118 = slice_114 = None
        mul_89: "f32[2048, 99]" = torch.ops.aten.mul.Tensor(mul_88, -1.0);  mul_88 = None
        add_29: "f32[2048, 99]" = torch.ops.aten.add.Tensor(mul_87, mul_89);  mul_87 = mul_89 = None
        slice_scatter_28: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_27, add_29, 1, 29, 9223372036854775807);  slice_scatter_27 = add_29 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_268: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 30)
        select_269: "f32[]" = torch.ops.aten.select.int(select_268, 0, 30);  select_268 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_270: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_28, 1, 30)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_60: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_30: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_271: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_28, 1, 30)
        view_120: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_271, [8, 128, 2]);  select_271 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_30: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_60, view_120, permute_30, alpha = -2.0);  unsqueeze_60 = view_120 = permute_30 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_30: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_30, -1);  baddbmm_30 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_61: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_30, -1)
        expand_60: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_61, [8, 128, 2]);  unsqueeze_61 = None
        gather_30: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_60);  expand_60 = None
        view_121: "f32[2048]" = torch.ops.aten.view.default(gather_30, [-1]);  gather_30 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_273: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_29, 1, 30)
        copy_30: "i64[8, 128]" = torch.ops.aten.copy.default(select_273, argmin_30);  select_273 = argmin_30 = None
        select_scatter_30: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_29, copy_30, 1, 30);  select_scatter_29 = copy_30 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_30: "f32[2048]" = torch.ops.aten.sub.Tensor(select_270, view_121);  select_270 = view_121 = None
        div_30: "f32[2048]" = torch.ops.aten.div.Tensor(sub_30, select_269);  sub_30 = select_269 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_275: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 30)
        slice_118: "f32[98]" = torch.ops.aten.slice.Tensor(select_275, 0, 30, 9223372036854775807);  select_275 = None
        slice_119: "f32[2048, 98]" = torch.ops.aten.slice.Tensor(slice_scatter_28, 1, 30, 9223372036854775807)
        expand_61: "f32[2048, 98]" = torch.ops.aten.expand.default(slice_119, [2048, 98]);  slice_119 = None
        mul_90: "f32[2048, 98]" = torch.ops.aten.mul.Tensor(expand_61, 1);  expand_61 = None
        view_122: "f32[2048, 1]" = torch.ops.aten.view.default(div_30, [2048, 1]);  div_30 = None
        mul_91: "f32[2048, 98]" = torch.ops.aten.mul.Tensor(view_122, slice_118);  view_122 = slice_118 = None
        mul_92: "f32[2048, 98]" = torch.ops.aten.mul.Tensor(mul_91, -1.0);  mul_91 = None
        add_30: "f32[2048, 98]" = torch.ops.aten.add.Tensor(mul_90, mul_92);  mul_90 = mul_92 = None
        slice_scatter_29: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_28, add_30, 1, 30, 9223372036854775807);  slice_scatter_28 = add_30 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_277: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 31)
        select_278: "f32[]" = torch.ops.aten.select.int(select_277, 0, 31);  select_277 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_279: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_29, 1, 31)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_62: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_31: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_280: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_29, 1, 31)
        view_124: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_280, [8, 128, 2]);  select_280 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_31: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_62, view_124, permute_31, alpha = -2.0);  unsqueeze_62 = view_124 = permute_31 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_31: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_31, -1);  baddbmm_31 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_63: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_31, -1)
        expand_62: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_63, [8, 128, 2]);  unsqueeze_63 = None
        gather_31: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_62);  expand_62 = None
        view_125: "f32[2048]" = torch.ops.aten.view.default(gather_31, [-1]);  gather_31 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_282: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_30, 1, 31)
        copy_31: "i64[8, 128]" = torch.ops.aten.copy.default(select_282, argmin_31);  select_282 = argmin_31 = None
        select_scatter_31: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_30, copy_31, 1, 31);  select_scatter_30 = copy_31 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_31: "f32[2048]" = torch.ops.aten.sub.Tensor(select_279, view_125);  select_279 = view_125 = None
        div_31: "f32[2048]" = torch.ops.aten.div.Tensor(sub_31, select_278);  sub_31 = select_278 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_284: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 31)
        slice_122: "f32[97]" = torch.ops.aten.slice.Tensor(select_284, 0, 31, 9223372036854775807);  select_284 = None
        slice_123: "f32[2048, 97]" = torch.ops.aten.slice.Tensor(slice_scatter_29, 1, 31, 9223372036854775807)
        expand_63: "f32[2048, 97]" = torch.ops.aten.expand.default(slice_123, [2048, 97]);  slice_123 = None
        mul_93: "f32[2048, 97]" = torch.ops.aten.mul.Tensor(expand_63, 1);  expand_63 = None
        view_126: "f32[2048, 1]" = torch.ops.aten.view.default(div_31, [2048, 1]);  div_31 = None
        mul_94: "f32[2048, 97]" = torch.ops.aten.mul.Tensor(view_126, slice_122);  view_126 = slice_122 = None
        mul_95: "f32[2048, 97]" = torch.ops.aten.mul.Tensor(mul_94, -1.0);  mul_94 = None
        add_31: "f32[2048, 97]" = torch.ops.aten.add.Tensor(mul_93, mul_95);  mul_93 = mul_95 = None
        slice_scatter_30: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_29, add_31, 1, 31, 9223372036854775807);  slice_scatter_29 = add_31 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_286: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 32)
        select_287: "f32[]" = torch.ops.aten.select.int(select_286, 0, 32);  select_286 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_288: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_30, 1, 32)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_64: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_32: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_289: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_30, 1, 32)
        view_128: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_289, [8, 128, 2]);  select_289 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_32: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_64, view_128, permute_32, alpha = -2.0);  unsqueeze_64 = view_128 = permute_32 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_32: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_32, -1);  baddbmm_32 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_65: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_32, -1)
        expand_64: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_65, [8, 128, 2]);  unsqueeze_65 = None
        gather_32: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_64);  expand_64 = None
        view_129: "f32[2048]" = torch.ops.aten.view.default(gather_32, [-1]);  gather_32 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_291: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_31, 1, 32)
        copy_32: "i64[8, 128]" = torch.ops.aten.copy.default(select_291, argmin_32);  select_291 = argmin_32 = None
        select_scatter_32: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_31, copy_32, 1, 32);  select_scatter_31 = copy_32 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_32: "f32[2048]" = torch.ops.aten.sub.Tensor(select_288, view_129);  select_288 = view_129 = None
        div_32: "f32[2048]" = torch.ops.aten.div.Tensor(sub_32, select_287);  sub_32 = select_287 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_293: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 32)
        slice_126: "f32[96]" = torch.ops.aten.slice.Tensor(select_293, 0, 32, 9223372036854775807);  select_293 = None
        slice_127: "f32[2048, 96]" = torch.ops.aten.slice.Tensor(slice_scatter_30, 1, 32, 9223372036854775807)
        expand_65: "f32[2048, 96]" = torch.ops.aten.expand.default(slice_127, [2048, 96]);  slice_127 = None
        mul_96: "f32[2048, 96]" = torch.ops.aten.mul.Tensor(expand_65, 1);  expand_65 = None
        view_130: "f32[2048, 1]" = torch.ops.aten.view.default(div_32, [2048, 1]);  div_32 = None
        mul_97: "f32[2048, 96]" = torch.ops.aten.mul.Tensor(view_130, slice_126);  view_130 = slice_126 = None
        mul_98: "f32[2048, 96]" = torch.ops.aten.mul.Tensor(mul_97, -1.0);  mul_97 = None
        add_32: "f32[2048, 96]" = torch.ops.aten.add.Tensor(mul_96, mul_98);  mul_96 = mul_98 = None
        slice_scatter_31: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_30, add_32, 1, 32, 9223372036854775807);  slice_scatter_30 = add_32 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_295: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 33)
        select_296: "f32[]" = torch.ops.aten.select.int(select_295, 0, 33);  select_295 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_297: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_31, 1, 33)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_66: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_33: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_298: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_31, 1, 33)
        view_132: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_298, [8, 128, 2]);  select_298 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_33: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_66, view_132, permute_33, alpha = -2.0);  unsqueeze_66 = view_132 = permute_33 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_33: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_33, -1);  baddbmm_33 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_67: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_33, -1)
        expand_66: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_67, [8, 128, 2]);  unsqueeze_67 = None
        gather_33: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_66);  expand_66 = None
        view_133: "f32[2048]" = torch.ops.aten.view.default(gather_33, [-1]);  gather_33 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_300: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_32, 1, 33)
        copy_33: "i64[8, 128]" = torch.ops.aten.copy.default(select_300, argmin_33);  select_300 = argmin_33 = None
        select_scatter_33: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_32, copy_33, 1, 33);  select_scatter_32 = copy_33 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_33: "f32[2048]" = torch.ops.aten.sub.Tensor(select_297, view_133);  select_297 = view_133 = None
        div_33: "f32[2048]" = torch.ops.aten.div.Tensor(sub_33, select_296);  sub_33 = select_296 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_302: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 33)
        slice_130: "f32[95]" = torch.ops.aten.slice.Tensor(select_302, 0, 33, 9223372036854775807);  select_302 = None
        slice_131: "f32[2048, 95]" = torch.ops.aten.slice.Tensor(slice_scatter_31, 1, 33, 9223372036854775807)
        expand_67: "f32[2048, 95]" = torch.ops.aten.expand.default(slice_131, [2048, 95]);  slice_131 = None
        mul_99: "f32[2048, 95]" = torch.ops.aten.mul.Tensor(expand_67, 1);  expand_67 = None
        view_134: "f32[2048, 1]" = torch.ops.aten.view.default(div_33, [2048, 1]);  div_33 = None
        mul_100: "f32[2048, 95]" = torch.ops.aten.mul.Tensor(view_134, slice_130);  view_134 = slice_130 = None
        mul_101: "f32[2048, 95]" = torch.ops.aten.mul.Tensor(mul_100, -1.0);  mul_100 = None
        add_33: "f32[2048, 95]" = torch.ops.aten.add.Tensor(mul_99, mul_101);  mul_99 = mul_101 = None
        slice_scatter_32: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_31, add_33, 1, 33, 9223372036854775807);  slice_scatter_31 = add_33 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_304: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 34)
        select_305: "f32[]" = torch.ops.aten.select.int(select_304, 0, 34);  select_304 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_306: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_32, 1, 34)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_68: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_34: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_307: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_32, 1, 34)
        view_136: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_307, [8, 128, 2]);  select_307 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_34: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_68, view_136, permute_34, alpha = -2.0);  unsqueeze_68 = view_136 = permute_34 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_34: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_34, -1);  baddbmm_34 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_69: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_34, -1)
        expand_68: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_69, [8, 128, 2]);  unsqueeze_69 = None
        gather_34: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_68);  expand_68 = None
        view_137: "f32[2048]" = torch.ops.aten.view.default(gather_34, [-1]);  gather_34 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_309: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_33, 1, 34)
        copy_34: "i64[8, 128]" = torch.ops.aten.copy.default(select_309, argmin_34);  select_309 = argmin_34 = None
        select_scatter_34: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_33, copy_34, 1, 34);  select_scatter_33 = copy_34 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_34: "f32[2048]" = torch.ops.aten.sub.Tensor(select_306, view_137);  select_306 = view_137 = None
        div_34: "f32[2048]" = torch.ops.aten.div.Tensor(sub_34, select_305);  sub_34 = select_305 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_311: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 34)
        slice_134: "f32[94]" = torch.ops.aten.slice.Tensor(select_311, 0, 34, 9223372036854775807);  select_311 = None
        slice_135: "f32[2048, 94]" = torch.ops.aten.slice.Tensor(slice_scatter_32, 1, 34, 9223372036854775807)
        expand_69: "f32[2048, 94]" = torch.ops.aten.expand.default(slice_135, [2048, 94]);  slice_135 = None
        mul_102: "f32[2048, 94]" = torch.ops.aten.mul.Tensor(expand_69, 1);  expand_69 = None
        view_138: "f32[2048, 1]" = torch.ops.aten.view.default(div_34, [2048, 1]);  div_34 = None
        mul_103: "f32[2048, 94]" = torch.ops.aten.mul.Tensor(view_138, slice_134);  view_138 = slice_134 = None
        mul_104: "f32[2048, 94]" = torch.ops.aten.mul.Tensor(mul_103, -1.0);  mul_103 = None
        add_34: "f32[2048, 94]" = torch.ops.aten.add.Tensor(mul_102, mul_104);  mul_102 = mul_104 = None
        slice_scatter_33: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_32, add_34, 1, 34, 9223372036854775807);  slice_scatter_32 = add_34 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_313: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 35)
        select_314: "f32[]" = torch.ops.aten.select.int(select_313, 0, 35);  select_313 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_315: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_33, 1, 35)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_70: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_35: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_316: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_33, 1, 35)
        view_140: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_316, [8, 128, 2]);  select_316 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_35: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_70, view_140, permute_35, alpha = -2.0);  unsqueeze_70 = view_140 = permute_35 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_35: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_35, -1);  baddbmm_35 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_71: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_35, -1)
        expand_70: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_71, [8, 128, 2]);  unsqueeze_71 = None
        gather_35: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_70);  expand_70 = None
        view_141: "f32[2048]" = torch.ops.aten.view.default(gather_35, [-1]);  gather_35 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_318: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_34, 1, 35)
        copy_35: "i64[8, 128]" = torch.ops.aten.copy.default(select_318, argmin_35);  select_318 = argmin_35 = None
        select_scatter_35: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_34, copy_35, 1, 35);  select_scatter_34 = copy_35 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_35: "f32[2048]" = torch.ops.aten.sub.Tensor(select_315, view_141);  select_315 = view_141 = None
        div_35: "f32[2048]" = torch.ops.aten.div.Tensor(sub_35, select_314);  sub_35 = select_314 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_320: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 35)
        slice_138: "f32[93]" = torch.ops.aten.slice.Tensor(select_320, 0, 35, 9223372036854775807);  select_320 = None
        slice_139: "f32[2048, 93]" = torch.ops.aten.slice.Tensor(slice_scatter_33, 1, 35, 9223372036854775807)
        expand_71: "f32[2048, 93]" = torch.ops.aten.expand.default(slice_139, [2048, 93]);  slice_139 = None
        mul_105: "f32[2048, 93]" = torch.ops.aten.mul.Tensor(expand_71, 1);  expand_71 = None
        view_142: "f32[2048, 1]" = torch.ops.aten.view.default(div_35, [2048, 1]);  div_35 = None
        mul_106: "f32[2048, 93]" = torch.ops.aten.mul.Tensor(view_142, slice_138);  view_142 = slice_138 = None
        mul_107: "f32[2048, 93]" = torch.ops.aten.mul.Tensor(mul_106, -1.0);  mul_106 = None
        add_35: "f32[2048, 93]" = torch.ops.aten.add.Tensor(mul_105, mul_107);  mul_105 = mul_107 = None
        slice_scatter_34: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_33, add_35, 1, 35, 9223372036854775807);  slice_scatter_33 = add_35 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_322: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 36)
        select_323: "f32[]" = torch.ops.aten.select.int(select_322, 0, 36);  select_322 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_324: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_34, 1, 36)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_72: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_36: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_325: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_34, 1, 36)
        view_144: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_325, [8, 128, 2]);  select_325 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_36: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_72, view_144, permute_36, alpha = -2.0);  unsqueeze_72 = view_144 = permute_36 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_36: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_36, -1);  baddbmm_36 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_73: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_36, -1)
        expand_72: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_73, [8, 128, 2]);  unsqueeze_73 = None
        gather_36: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_72);  expand_72 = None
        view_145: "f32[2048]" = torch.ops.aten.view.default(gather_36, [-1]);  gather_36 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_327: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_35, 1, 36)
        copy_36: "i64[8, 128]" = torch.ops.aten.copy.default(select_327, argmin_36);  select_327 = argmin_36 = None
        select_scatter_36: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_35, copy_36, 1, 36);  select_scatter_35 = copy_36 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_36: "f32[2048]" = torch.ops.aten.sub.Tensor(select_324, view_145);  select_324 = view_145 = None
        div_36: "f32[2048]" = torch.ops.aten.div.Tensor(sub_36, select_323);  sub_36 = select_323 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_329: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 36)
        slice_142: "f32[92]" = torch.ops.aten.slice.Tensor(select_329, 0, 36, 9223372036854775807);  select_329 = None
        slice_143: "f32[2048, 92]" = torch.ops.aten.slice.Tensor(slice_scatter_34, 1, 36, 9223372036854775807)
        expand_73: "f32[2048, 92]" = torch.ops.aten.expand.default(slice_143, [2048, 92]);  slice_143 = None
        mul_108: "f32[2048, 92]" = torch.ops.aten.mul.Tensor(expand_73, 1);  expand_73 = None
        view_146: "f32[2048, 1]" = torch.ops.aten.view.default(div_36, [2048, 1]);  div_36 = None
        mul_109: "f32[2048, 92]" = torch.ops.aten.mul.Tensor(view_146, slice_142);  view_146 = slice_142 = None
        mul_110: "f32[2048, 92]" = torch.ops.aten.mul.Tensor(mul_109, -1.0);  mul_109 = None
        add_36: "f32[2048, 92]" = torch.ops.aten.add.Tensor(mul_108, mul_110);  mul_108 = mul_110 = None
        slice_scatter_35: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_34, add_36, 1, 36, 9223372036854775807);  slice_scatter_34 = add_36 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_331: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 37)
        select_332: "f32[]" = torch.ops.aten.select.int(select_331, 0, 37);  select_331 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_333: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_35, 1, 37)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_74: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_37: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_334: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_35, 1, 37)
        view_148: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_334, [8, 128, 2]);  select_334 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_37: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_74, view_148, permute_37, alpha = -2.0);  unsqueeze_74 = view_148 = permute_37 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_37: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_37, -1);  baddbmm_37 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_75: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_37, -1)
        expand_74: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_75, [8, 128, 2]);  unsqueeze_75 = None
        gather_37: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_74);  expand_74 = None
        view_149: "f32[2048]" = torch.ops.aten.view.default(gather_37, [-1]);  gather_37 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_336: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_36, 1, 37)
        copy_37: "i64[8, 128]" = torch.ops.aten.copy.default(select_336, argmin_37);  select_336 = argmin_37 = None
        select_scatter_37: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_36, copy_37, 1, 37);  select_scatter_36 = copy_37 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_37: "f32[2048]" = torch.ops.aten.sub.Tensor(select_333, view_149);  select_333 = view_149 = None
        div_37: "f32[2048]" = torch.ops.aten.div.Tensor(sub_37, select_332);  sub_37 = select_332 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_338: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 37)
        slice_146: "f32[91]" = torch.ops.aten.slice.Tensor(select_338, 0, 37, 9223372036854775807);  select_338 = None
        slice_147: "f32[2048, 91]" = torch.ops.aten.slice.Tensor(slice_scatter_35, 1, 37, 9223372036854775807)
        expand_75: "f32[2048, 91]" = torch.ops.aten.expand.default(slice_147, [2048, 91]);  slice_147 = None
        mul_111: "f32[2048, 91]" = torch.ops.aten.mul.Tensor(expand_75, 1);  expand_75 = None
        view_150: "f32[2048, 1]" = torch.ops.aten.view.default(div_37, [2048, 1]);  div_37 = None
        mul_112: "f32[2048, 91]" = torch.ops.aten.mul.Tensor(view_150, slice_146);  view_150 = slice_146 = None
        mul_113: "f32[2048, 91]" = torch.ops.aten.mul.Tensor(mul_112, -1.0);  mul_112 = None
        add_37: "f32[2048, 91]" = torch.ops.aten.add.Tensor(mul_111, mul_113);  mul_111 = mul_113 = None
        slice_scatter_36: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_35, add_37, 1, 37, 9223372036854775807);  slice_scatter_35 = add_37 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_340: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 38)
        select_341: "f32[]" = torch.ops.aten.select.int(select_340, 0, 38);  select_340 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_342: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_36, 1, 38)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_76: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_38: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_343: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_36, 1, 38)
        view_152: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_343, [8, 128, 2]);  select_343 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_38: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_76, view_152, permute_38, alpha = -2.0);  unsqueeze_76 = view_152 = permute_38 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_38: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_38, -1);  baddbmm_38 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_77: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_38, -1)
        expand_76: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_77, [8, 128, 2]);  unsqueeze_77 = None
        gather_38: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_76);  expand_76 = None
        view_153: "f32[2048]" = torch.ops.aten.view.default(gather_38, [-1]);  gather_38 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_345: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_37, 1, 38)
        copy_38: "i64[8, 128]" = torch.ops.aten.copy.default(select_345, argmin_38);  select_345 = argmin_38 = None
        select_scatter_38: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_37, copy_38, 1, 38);  select_scatter_37 = copy_38 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_38: "f32[2048]" = torch.ops.aten.sub.Tensor(select_342, view_153);  select_342 = view_153 = None
        div_38: "f32[2048]" = torch.ops.aten.div.Tensor(sub_38, select_341);  sub_38 = select_341 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_347: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 38)
        slice_150: "f32[90]" = torch.ops.aten.slice.Tensor(select_347, 0, 38, 9223372036854775807);  select_347 = None
        slice_151: "f32[2048, 90]" = torch.ops.aten.slice.Tensor(slice_scatter_36, 1, 38, 9223372036854775807)
        expand_77: "f32[2048, 90]" = torch.ops.aten.expand.default(slice_151, [2048, 90]);  slice_151 = None
        mul_114: "f32[2048, 90]" = torch.ops.aten.mul.Tensor(expand_77, 1);  expand_77 = None
        view_154: "f32[2048, 1]" = torch.ops.aten.view.default(div_38, [2048, 1]);  div_38 = None
        mul_115: "f32[2048, 90]" = torch.ops.aten.mul.Tensor(view_154, slice_150);  view_154 = slice_150 = None
        mul_116: "f32[2048, 90]" = torch.ops.aten.mul.Tensor(mul_115, -1.0);  mul_115 = None
        add_38: "f32[2048, 90]" = torch.ops.aten.add.Tensor(mul_114, mul_116);  mul_114 = mul_116 = None
        slice_scatter_37: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_36, add_38, 1, 38, 9223372036854775807);  slice_scatter_36 = add_38 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_349: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 39)
        select_350: "f32[]" = torch.ops.aten.select.int(select_349, 0, 39);  select_349 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_351: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_37, 1, 39)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_78: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_39: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_352: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_37, 1, 39)
        view_156: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_352, [8, 128, 2]);  select_352 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_39: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_78, view_156, permute_39, alpha = -2.0);  unsqueeze_78 = view_156 = permute_39 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_39: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_39, -1);  baddbmm_39 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_79: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_39, -1)
        expand_78: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_79, [8, 128, 2]);  unsqueeze_79 = None
        gather_39: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_78);  expand_78 = None
        view_157: "f32[2048]" = torch.ops.aten.view.default(gather_39, [-1]);  gather_39 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_354: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_38, 1, 39)
        copy_39: "i64[8, 128]" = torch.ops.aten.copy.default(select_354, argmin_39);  select_354 = argmin_39 = None
        select_scatter_39: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_38, copy_39, 1, 39);  select_scatter_38 = copy_39 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_39: "f32[2048]" = torch.ops.aten.sub.Tensor(select_351, view_157);  select_351 = view_157 = None
        div_39: "f32[2048]" = torch.ops.aten.div.Tensor(sub_39, select_350);  sub_39 = select_350 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_356: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 39)
        slice_154: "f32[89]" = torch.ops.aten.slice.Tensor(select_356, 0, 39, 9223372036854775807);  select_356 = None
        slice_155: "f32[2048, 89]" = torch.ops.aten.slice.Tensor(slice_scatter_37, 1, 39, 9223372036854775807)
        expand_79: "f32[2048, 89]" = torch.ops.aten.expand.default(slice_155, [2048, 89]);  slice_155 = None
        mul_117: "f32[2048, 89]" = torch.ops.aten.mul.Tensor(expand_79, 1);  expand_79 = None
        view_158: "f32[2048, 1]" = torch.ops.aten.view.default(div_39, [2048, 1]);  div_39 = None
        mul_118: "f32[2048, 89]" = torch.ops.aten.mul.Tensor(view_158, slice_154);  view_158 = slice_154 = None
        mul_119: "f32[2048, 89]" = torch.ops.aten.mul.Tensor(mul_118, -1.0);  mul_118 = None
        add_39: "f32[2048, 89]" = torch.ops.aten.add.Tensor(mul_117, mul_119);  mul_117 = mul_119 = None
        slice_scatter_38: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_37, add_39, 1, 39, 9223372036854775807);  slice_scatter_37 = add_39 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_358: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 40)
        select_359: "f32[]" = torch.ops.aten.select.int(select_358, 0, 40);  select_358 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_360: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_38, 1, 40)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_80: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_40: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_361: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_38, 1, 40)
        view_160: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_361, [8, 128, 2]);  select_361 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_40: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_80, view_160, permute_40, alpha = -2.0);  unsqueeze_80 = view_160 = permute_40 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_40: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_40, -1);  baddbmm_40 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_81: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_40, -1)
        expand_80: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_81, [8, 128, 2]);  unsqueeze_81 = None
        gather_40: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_80);  expand_80 = None
        view_161: "f32[2048]" = torch.ops.aten.view.default(gather_40, [-1]);  gather_40 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_363: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_39, 1, 40)
        copy_40: "i64[8, 128]" = torch.ops.aten.copy.default(select_363, argmin_40);  select_363 = argmin_40 = None
        select_scatter_40: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_39, copy_40, 1, 40);  select_scatter_39 = copy_40 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_40: "f32[2048]" = torch.ops.aten.sub.Tensor(select_360, view_161);  select_360 = view_161 = None
        div_40: "f32[2048]" = torch.ops.aten.div.Tensor(sub_40, select_359);  sub_40 = select_359 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_365: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 40)
        slice_158: "f32[88]" = torch.ops.aten.slice.Tensor(select_365, 0, 40, 9223372036854775807);  select_365 = None
        slice_159: "f32[2048, 88]" = torch.ops.aten.slice.Tensor(slice_scatter_38, 1, 40, 9223372036854775807)
        expand_81: "f32[2048, 88]" = torch.ops.aten.expand.default(slice_159, [2048, 88]);  slice_159 = None
        mul_120: "f32[2048, 88]" = torch.ops.aten.mul.Tensor(expand_81, 1);  expand_81 = None
        view_162: "f32[2048, 1]" = torch.ops.aten.view.default(div_40, [2048, 1]);  div_40 = None
        mul_121: "f32[2048, 88]" = torch.ops.aten.mul.Tensor(view_162, slice_158);  view_162 = slice_158 = None
        mul_122: "f32[2048, 88]" = torch.ops.aten.mul.Tensor(mul_121, -1.0);  mul_121 = None
        add_40: "f32[2048, 88]" = torch.ops.aten.add.Tensor(mul_120, mul_122);  mul_120 = mul_122 = None
        slice_scatter_39: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_38, add_40, 1, 40, 9223372036854775807);  slice_scatter_38 = add_40 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_367: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 41)
        select_368: "f32[]" = torch.ops.aten.select.int(select_367, 0, 41);  select_367 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_369: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_39, 1, 41)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_82: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_41: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_370: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_39, 1, 41)
        view_164: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_370, [8, 128, 2]);  select_370 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_41: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_82, view_164, permute_41, alpha = -2.0);  unsqueeze_82 = view_164 = permute_41 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_41: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_41, -1);  baddbmm_41 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_83: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_41, -1)
        expand_82: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_83, [8, 128, 2]);  unsqueeze_83 = None
        gather_41: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_82);  expand_82 = None
        view_165: "f32[2048]" = torch.ops.aten.view.default(gather_41, [-1]);  gather_41 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_372: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_40, 1, 41)
        copy_41: "i64[8, 128]" = torch.ops.aten.copy.default(select_372, argmin_41);  select_372 = argmin_41 = None
        select_scatter_41: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_40, copy_41, 1, 41);  select_scatter_40 = copy_41 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_41: "f32[2048]" = torch.ops.aten.sub.Tensor(select_369, view_165);  select_369 = view_165 = None
        div_41: "f32[2048]" = torch.ops.aten.div.Tensor(sub_41, select_368);  sub_41 = select_368 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_374: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 41)
        slice_162: "f32[87]" = torch.ops.aten.slice.Tensor(select_374, 0, 41, 9223372036854775807);  select_374 = None
        slice_163: "f32[2048, 87]" = torch.ops.aten.slice.Tensor(slice_scatter_39, 1, 41, 9223372036854775807)
        expand_83: "f32[2048, 87]" = torch.ops.aten.expand.default(slice_163, [2048, 87]);  slice_163 = None
        mul_123: "f32[2048, 87]" = torch.ops.aten.mul.Tensor(expand_83, 1);  expand_83 = None
        view_166: "f32[2048, 1]" = torch.ops.aten.view.default(div_41, [2048, 1]);  div_41 = None
        mul_124: "f32[2048, 87]" = torch.ops.aten.mul.Tensor(view_166, slice_162);  view_166 = slice_162 = None
        mul_125: "f32[2048, 87]" = torch.ops.aten.mul.Tensor(mul_124, -1.0);  mul_124 = None
        add_41: "f32[2048, 87]" = torch.ops.aten.add.Tensor(mul_123, mul_125);  mul_123 = mul_125 = None
        slice_scatter_40: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_39, add_41, 1, 41, 9223372036854775807);  slice_scatter_39 = add_41 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_376: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 42)
        select_377: "f32[]" = torch.ops.aten.select.int(select_376, 0, 42);  select_376 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_378: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_40, 1, 42)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_84: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_42: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_379: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_40, 1, 42)
        view_168: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_379, [8, 128, 2]);  select_379 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_42: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_84, view_168, permute_42, alpha = -2.0);  unsqueeze_84 = view_168 = permute_42 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_42: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_42, -1);  baddbmm_42 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_85: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_42, -1)
        expand_84: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_85, [8, 128, 2]);  unsqueeze_85 = None
        gather_42: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_84);  expand_84 = None
        view_169: "f32[2048]" = torch.ops.aten.view.default(gather_42, [-1]);  gather_42 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_381: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_41, 1, 42)
        copy_42: "i64[8, 128]" = torch.ops.aten.copy.default(select_381, argmin_42);  select_381 = argmin_42 = None
        select_scatter_42: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_41, copy_42, 1, 42);  select_scatter_41 = copy_42 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_42: "f32[2048]" = torch.ops.aten.sub.Tensor(select_378, view_169);  select_378 = view_169 = None
        div_42: "f32[2048]" = torch.ops.aten.div.Tensor(sub_42, select_377);  sub_42 = select_377 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_383: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 42)
        slice_166: "f32[86]" = torch.ops.aten.slice.Tensor(select_383, 0, 42, 9223372036854775807);  select_383 = None
        slice_167: "f32[2048, 86]" = torch.ops.aten.slice.Tensor(slice_scatter_40, 1, 42, 9223372036854775807)
        expand_85: "f32[2048, 86]" = torch.ops.aten.expand.default(slice_167, [2048, 86]);  slice_167 = None
        mul_126: "f32[2048, 86]" = torch.ops.aten.mul.Tensor(expand_85, 1);  expand_85 = None
        view_170: "f32[2048, 1]" = torch.ops.aten.view.default(div_42, [2048, 1]);  div_42 = None
        mul_127: "f32[2048, 86]" = torch.ops.aten.mul.Tensor(view_170, slice_166);  view_170 = slice_166 = None
        mul_128: "f32[2048, 86]" = torch.ops.aten.mul.Tensor(mul_127, -1.0);  mul_127 = None
        add_42: "f32[2048, 86]" = torch.ops.aten.add.Tensor(mul_126, mul_128);  mul_126 = mul_128 = None
        slice_scatter_41: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_40, add_42, 1, 42, 9223372036854775807);  slice_scatter_40 = add_42 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_385: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 43)
        select_386: "f32[]" = torch.ops.aten.select.int(select_385, 0, 43);  select_385 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_387: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_41, 1, 43)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_86: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_43: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_388: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_41, 1, 43)
        view_172: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_388, [8, 128, 2]);  select_388 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_43: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_86, view_172, permute_43, alpha = -2.0);  unsqueeze_86 = view_172 = permute_43 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_43: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_43, -1);  baddbmm_43 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_87: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_43, -1)
        expand_86: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_87, [8, 128, 2]);  unsqueeze_87 = None
        gather_43: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_86);  expand_86 = None
        view_173: "f32[2048]" = torch.ops.aten.view.default(gather_43, [-1]);  gather_43 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_390: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_42, 1, 43)
        copy_43: "i64[8, 128]" = torch.ops.aten.copy.default(select_390, argmin_43);  select_390 = argmin_43 = None
        select_scatter_43: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_42, copy_43, 1, 43);  select_scatter_42 = copy_43 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_43: "f32[2048]" = torch.ops.aten.sub.Tensor(select_387, view_173);  select_387 = view_173 = None
        div_43: "f32[2048]" = torch.ops.aten.div.Tensor(sub_43, select_386);  sub_43 = select_386 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_392: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 43)
        slice_170: "f32[85]" = torch.ops.aten.slice.Tensor(select_392, 0, 43, 9223372036854775807);  select_392 = None
        slice_171: "f32[2048, 85]" = torch.ops.aten.slice.Tensor(slice_scatter_41, 1, 43, 9223372036854775807)
        expand_87: "f32[2048, 85]" = torch.ops.aten.expand.default(slice_171, [2048, 85]);  slice_171 = None
        mul_129: "f32[2048, 85]" = torch.ops.aten.mul.Tensor(expand_87, 1);  expand_87 = None
        view_174: "f32[2048, 1]" = torch.ops.aten.view.default(div_43, [2048, 1]);  div_43 = None
        mul_130: "f32[2048, 85]" = torch.ops.aten.mul.Tensor(view_174, slice_170);  view_174 = slice_170 = None
        mul_131: "f32[2048, 85]" = torch.ops.aten.mul.Tensor(mul_130, -1.0);  mul_130 = None
        add_43: "f32[2048, 85]" = torch.ops.aten.add.Tensor(mul_129, mul_131);  mul_129 = mul_131 = None
        slice_scatter_42: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_41, add_43, 1, 43, 9223372036854775807);  slice_scatter_41 = add_43 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_394: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 44)
        select_395: "f32[]" = torch.ops.aten.select.int(select_394, 0, 44);  select_394 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_396: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_42, 1, 44)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_88: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_44: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_397: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_42, 1, 44)
        view_176: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_397, [8, 128, 2]);  select_397 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_44: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_88, view_176, permute_44, alpha = -2.0);  unsqueeze_88 = view_176 = permute_44 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_44: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_44, -1);  baddbmm_44 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_89: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_44, -1)
        expand_88: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_89, [8, 128, 2]);  unsqueeze_89 = None
        gather_44: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_88);  expand_88 = None
        view_177: "f32[2048]" = torch.ops.aten.view.default(gather_44, [-1]);  gather_44 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_399: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_43, 1, 44)
        copy_44: "i64[8, 128]" = torch.ops.aten.copy.default(select_399, argmin_44);  select_399 = argmin_44 = None
        select_scatter_44: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_43, copy_44, 1, 44);  select_scatter_43 = copy_44 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_44: "f32[2048]" = torch.ops.aten.sub.Tensor(select_396, view_177);  select_396 = view_177 = None
        div_44: "f32[2048]" = torch.ops.aten.div.Tensor(sub_44, select_395);  sub_44 = select_395 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_401: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 44)
        slice_174: "f32[84]" = torch.ops.aten.slice.Tensor(select_401, 0, 44, 9223372036854775807);  select_401 = None
        slice_175: "f32[2048, 84]" = torch.ops.aten.slice.Tensor(slice_scatter_42, 1, 44, 9223372036854775807)
        expand_89: "f32[2048, 84]" = torch.ops.aten.expand.default(slice_175, [2048, 84]);  slice_175 = None
        mul_132: "f32[2048, 84]" = torch.ops.aten.mul.Tensor(expand_89, 1);  expand_89 = None
        view_178: "f32[2048, 1]" = torch.ops.aten.view.default(div_44, [2048, 1]);  div_44 = None
        mul_133: "f32[2048, 84]" = torch.ops.aten.mul.Tensor(view_178, slice_174);  view_178 = slice_174 = None
        mul_134: "f32[2048, 84]" = torch.ops.aten.mul.Tensor(mul_133, -1.0);  mul_133 = None
        add_44: "f32[2048, 84]" = torch.ops.aten.add.Tensor(mul_132, mul_134);  mul_132 = mul_134 = None
        slice_scatter_43: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_42, add_44, 1, 44, 9223372036854775807);  slice_scatter_42 = add_44 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_403: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 45)
        select_404: "f32[]" = torch.ops.aten.select.int(select_403, 0, 45);  select_403 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_405: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_43, 1, 45)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_90: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_45: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_406: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_43, 1, 45)
        view_180: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_406, [8, 128, 2]);  select_406 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_45: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_90, view_180, permute_45, alpha = -2.0);  unsqueeze_90 = view_180 = permute_45 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_45: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_45, -1);  baddbmm_45 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_91: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_45, -1)
        expand_90: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_91, [8, 128, 2]);  unsqueeze_91 = None
        gather_45: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_90);  expand_90 = None
        view_181: "f32[2048]" = torch.ops.aten.view.default(gather_45, [-1]);  gather_45 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_408: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_44, 1, 45)
        copy_45: "i64[8, 128]" = torch.ops.aten.copy.default(select_408, argmin_45);  select_408 = argmin_45 = None
        select_scatter_45: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_44, copy_45, 1, 45);  select_scatter_44 = copy_45 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_45: "f32[2048]" = torch.ops.aten.sub.Tensor(select_405, view_181);  select_405 = view_181 = None
        div_45: "f32[2048]" = torch.ops.aten.div.Tensor(sub_45, select_404);  sub_45 = select_404 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_410: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 45)
        slice_178: "f32[83]" = torch.ops.aten.slice.Tensor(select_410, 0, 45, 9223372036854775807);  select_410 = None
        slice_179: "f32[2048, 83]" = torch.ops.aten.slice.Tensor(slice_scatter_43, 1, 45, 9223372036854775807)
        expand_91: "f32[2048, 83]" = torch.ops.aten.expand.default(slice_179, [2048, 83]);  slice_179 = None
        mul_135: "f32[2048, 83]" = torch.ops.aten.mul.Tensor(expand_91, 1);  expand_91 = None
        view_182: "f32[2048, 1]" = torch.ops.aten.view.default(div_45, [2048, 1]);  div_45 = None
        mul_136: "f32[2048, 83]" = torch.ops.aten.mul.Tensor(view_182, slice_178);  view_182 = slice_178 = None
        mul_137: "f32[2048, 83]" = torch.ops.aten.mul.Tensor(mul_136, -1.0);  mul_136 = None
        add_45: "f32[2048, 83]" = torch.ops.aten.add.Tensor(mul_135, mul_137);  mul_135 = mul_137 = None
        slice_scatter_44: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_43, add_45, 1, 45, 9223372036854775807);  slice_scatter_43 = add_45 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_412: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 46)
        select_413: "f32[]" = torch.ops.aten.select.int(select_412, 0, 46);  select_412 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_414: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_44, 1, 46)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_92: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_46: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_415: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_44, 1, 46)
        view_184: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_415, [8, 128, 2]);  select_415 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_46: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_92, view_184, permute_46, alpha = -2.0);  unsqueeze_92 = view_184 = permute_46 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_46: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_46, -1);  baddbmm_46 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_93: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_46, -1)
        expand_92: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_93, [8, 128, 2]);  unsqueeze_93 = None
        gather_46: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_92);  expand_92 = None
        view_185: "f32[2048]" = torch.ops.aten.view.default(gather_46, [-1]);  gather_46 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_417: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_45, 1, 46)
        copy_46: "i64[8, 128]" = torch.ops.aten.copy.default(select_417, argmin_46);  select_417 = argmin_46 = None
        select_scatter_46: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_45, copy_46, 1, 46);  select_scatter_45 = copy_46 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_46: "f32[2048]" = torch.ops.aten.sub.Tensor(select_414, view_185);  select_414 = view_185 = None
        div_46: "f32[2048]" = torch.ops.aten.div.Tensor(sub_46, select_413);  sub_46 = select_413 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_419: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 46)
        slice_182: "f32[82]" = torch.ops.aten.slice.Tensor(select_419, 0, 46, 9223372036854775807);  select_419 = None
        slice_183: "f32[2048, 82]" = torch.ops.aten.slice.Tensor(slice_scatter_44, 1, 46, 9223372036854775807)
        expand_93: "f32[2048, 82]" = torch.ops.aten.expand.default(slice_183, [2048, 82]);  slice_183 = None
        mul_138: "f32[2048, 82]" = torch.ops.aten.mul.Tensor(expand_93, 1);  expand_93 = None
        view_186: "f32[2048, 1]" = torch.ops.aten.view.default(div_46, [2048, 1]);  div_46 = None
        mul_139: "f32[2048, 82]" = torch.ops.aten.mul.Tensor(view_186, slice_182);  view_186 = slice_182 = None
        mul_140: "f32[2048, 82]" = torch.ops.aten.mul.Tensor(mul_139, -1.0);  mul_139 = None
        add_46: "f32[2048, 82]" = torch.ops.aten.add.Tensor(mul_138, mul_140);  mul_138 = mul_140 = None
        slice_scatter_45: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_44, add_46, 1, 46, 9223372036854775807);  slice_scatter_44 = add_46 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_421: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 47)
        select_422: "f32[]" = torch.ops.aten.select.int(select_421, 0, 47);  select_421 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_423: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_45, 1, 47)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_94: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_47: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_424: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_45, 1, 47)
        view_188: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_424, [8, 128, 2]);  select_424 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_47: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_94, view_188, permute_47, alpha = -2.0);  unsqueeze_94 = view_188 = permute_47 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_47: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_47, -1);  baddbmm_47 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_95: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_47, -1)
        expand_94: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_95, [8, 128, 2]);  unsqueeze_95 = None
        gather_47: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_94);  expand_94 = None
        view_189: "f32[2048]" = torch.ops.aten.view.default(gather_47, [-1]);  gather_47 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_426: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_46, 1, 47)
        copy_47: "i64[8, 128]" = torch.ops.aten.copy.default(select_426, argmin_47);  select_426 = argmin_47 = None
        select_scatter_47: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_46, copy_47, 1, 47);  select_scatter_46 = copy_47 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_47: "f32[2048]" = torch.ops.aten.sub.Tensor(select_423, view_189);  select_423 = view_189 = None
        div_47: "f32[2048]" = torch.ops.aten.div.Tensor(sub_47, select_422);  sub_47 = select_422 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_428: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 47)
        slice_186: "f32[81]" = torch.ops.aten.slice.Tensor(select_428, 0, 47, 9223372036854775807);  select_428 = None
        slice_187: "f32[2048, 81]" = torch.ops.aten.slice.Tensor(slice_scatter_45, 1, 47, 9223372036854775807)
        expand_95: "f32[2048, 81]" = torch.ops.aten.expand.default(slice_187, [2048, 81]);  slice_187 = None
        mul_141: "f32[2048, 81]" = torch.ops.aten.mul.Tensor(expand_95, 1);  expand_95 = None
        view_190: "f32[2048, 1]" = torch.ops.aten.view.default(div_47, [2048, 1]);  div_47 = None
        mul_142: "f32[2048, 81]" = torch.ops.aten.mul.Tensor(view_190, slice_186);  view_190 = slice_186 = None
        mul_143: "f32[2048, 81]" = torch.ops.aten.mul.Tensor(mul_142, -1.0);  mul_142 = None
        add_47: "f32[2048, 81]" = torch.ops.aten.add.Tensor(mul_141, mul_143);  mul_141 = mul_143 = None
        slice_scatter_46: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_45, add_47, 1, 47, 9223372036854775807);  slice_scatter_45 = add_47 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_430: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 48)
        select_431: "f32[]" = torch.ops.aten.select.int(select_430, 0, 48);  select_430 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_432: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_46, 1, 48)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_96: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_48: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_433: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_46, 1, 48)
        view_192: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_433, [8, 128, 2]);  select_433 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_48: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_96, view_192, permute_48, alpha = -2.0);  unsqueeze_96 = view_192 = permute_48 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_48: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_48, -1);  baddbmm_48 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_97: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_48, -1)
        expand_96: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_97, [8, 128, 2]);  unsqueeze_97 = None
        gather_48: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_96);  expand_96 = None
        view_193: "f32[2048]" = torch.ops.aten.view.default(gather_48, [-1]);  gather_48 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_435: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_47, 1, 48)
        copy_48: "i64[8, 128]" = torch.ops.aten.copy.default(select_435, argmin_48);  select_435 = argmin_48 = None
        select_scatter_48: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_47, copy_48, 1, 48);  select_scatter_47 = copy_48 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_48: "f32[2048]" = torch.ops.aten.sub.Tensor(select_432, view_193);  select_432 = view_193 = None
        div_48: "f32[2048]" = torch.ops.aten.div.Tensor(sub_48, select_431);  sub_48 = select_431 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_437: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 48)
        slice_190: "f32[80]" = torch.ops.aten.slice.Tensor(select_437, 0, 48, 9223372036854775807);  select_437 = None
        slice_191: "f32[2048, 80]" = torch.ops.aten.slice.Tensor(slice_scatter_46, 1, 48, 9223372036854775807)
        expand_97: "f32[2048, 80]" = torch.ops.aten.expand.default(slice_191, [2048, 80]);  slice_191 = None
        mul_144: "f32[2048, 80]" = torch.ops.aten.mul.Tensor(expand_97, 1);  expand_97 = None
        view_194: "f32[2048, 1]" = torch.ops.aten.view.default(div_48, [2048, 1]);  div_48 = None
        mul_145: "f32[2048, 80]" = torch.ops.aten.mul.Tensor(view_194, slice_190);  view_194 = slice_190 = None
        mul_146: "f32[2048, 80]" = torch.ops.aten.mul.Tensor(mul_145, -1.0);  mul_145 = None
        add_48: "f32[2048, 80]" = torch.ops.aten.add.Tensor(mul_144, mul_146);  mul_144 = mul_146 = None
        slice_scatter_47: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_46, add_48, 1, 48, 9223372036854775807);  slice_scatter_46 = add_48 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_439: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 49)
        select_440: "f32[]" = torch.ops.aten.select.int(select_439, 0, 49);  select_439 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_441: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_47, 1, 49)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_98: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_49: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_442: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_47, 1, 49)
        view_196: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_442, [8, 128, 2]);  select_442 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_49: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_98, view_196, permute_49, alpha = -2.0);  unsqueeze_98 = view_196 = permute_49 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_49: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_49, -1);  baddbmm_49 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_99: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_49, -1)
        expand_98: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_99, [8, 128, 2]);  unsqueeze_99 = None
        gather_49: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_98);  expand_98 = None
        view_197: "f32[2048]" = torch.ops.aten.view.default(gather_49, [-1]);  gather_49 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_444: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_48, 1, 49)
        copy_49: "i64[8, 128]" = torch.ops.aten.copy.default(select_444, argmin_49);  select_444 = argmin_49 = None
        select_scatter_49: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_48, copy_49, 1, 49);  select_scatter_48 = copy_49 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_49: "f32[2048]" = torch.ops.aten.sub.Tensor(select_441, view_197);  select_441 = view_197 = None
        div_49: "f32[2048]" = torch.ops.aten.div.Tensor(sub_49, select_440);  sub_49 = select_440 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_446: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 49)
        slice_194: "f32[79]" = torch.ops.aten.slice.Tensor(select_446, 0, 49, 9223372036854775807);  select_446 = None
        slice_195: "f32[2048, 79]" = torch.ops.aten.slice.Tensor(slice_scatter_47, 1, 49, 9223372036854775807)
        expand_99: "f32[2048, 79]" = torch.ops.aten.expand.default(slice_195, [2048, 79]);  slice_195 = None
        mul_147: "f32[2048, 79]" = torch.ops.aten.mul.Tensor(expand_99, 1);  expand_99 = None
        view_198: "f32[2048, 1]" = torch.ops.aten.view.default(div_49, [2048, 1]);  div_49 = None
        mul_148: "f32[2048, 79]" = torch.ops.aten.mul.Tensor(view_198, slice_194);  view_198 = slice_194 = None
        mul_149: "f32[2048, 79]" = torch.ops.aten.mul.Tensor(mul_148, -1.0);  mul_148 = None
        add_49: "f32[2048, 79]" = torch.ops.aten.add.Tensor(mul_147, mul_149);  mul_147 = mul_149 = None
        slice_scatter_48: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_47, add_49, 1, 49, 9223372036854775807);  slice_scatter_47 = add_49 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_448: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 50)
        select_449: "f32[]" = torch.ops.aten.select.int(select_448, 0, 50);  select_448 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_450: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_48, 1, 50)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_100: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_50: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_451: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_48, 1, 50)
        view_200: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_451, [8, 128, 2]);  select_451 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_50: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_100, view_200, permute_50, alpha = -2.0);  unsqueeze_100 = view_200 = permute_50 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_50: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_50, -1);  baddbmm_50 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_101: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_50, -1)
        expand_100: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_101, [8, 128, 2]);  unsqueeze_101 = None
        gather_50: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_100);  expand_100 = None
        view_201: "f32[2048]" = torch.ops.aten.view.default(gather_50, [-1]);  gather_50 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_453: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_49, 1, 50)
        copy_50: "i64[8, 128]" = torch.ops.aten.copy.default(select_453, argmin_50);  select_453 = argmin_50 = None
        select_scatter_50: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_49, copy_50, 1, 50);  select_scatter_49 = copy_50 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_50: "f32[2048]" = torch.ops.aten.sub.Tensor(select_450, view_201);  select_450 = view_201 = None
        div_50: "f32[2048]" = torch.ops.aten.div.Tensor(sub_50, select_449);  sub_50 = select_449 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_455: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 50)
        slice_198: "f32[78]" = torch.ops.aten.slice.Tensor(select_455, 0, 50, 9223372036854775807);  select_455 = None
        slice_199: "f32[2048, 78]" = torch.ops.aten.slice.Tensor(slice_scatter_48, 1, 50, 9223372036854775807)
        expand_101: "f32[2048, 78]" = torch.ops.aten.expand.default(slice_199, [2048, 78]);  slice_199 = None
        mul_150: "f32[2048, 78]" = torch.ops.aten.mul.Tensor(expand_101, 1);  expand_101 = None
        view_202: "f32[2048, 1]" = torch.ops.aten.view.default(div_50, [2048, 1]);  div_50 = None
        mul_151: "f32[2048, 78]" = torch.ops.aten.mul.Tensor(view_202, slice_198);  view_202 = slice_198 = None
        mul_152: "f32[2048, 78]" = torch.ops.aten.mul.Tensor(mul_151, -1.0);  mul_151 = None
        add_50: "f32[2048, 78]" = torch.ops.aten.add.Tensor(mul_150, mul_152);  mul_150 = mul_152 = None
        slice_scatter_49: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_48, add_50, 1, 50, 9223372036854775807);  slice_scatter_48 = add_50 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_457: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 51)
        select_458: "f32[]" = torch.ops.aten.select.int(select_457, 0, 51);  select_457 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_459: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_49, 1, 51)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_102: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_51: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_460: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_49, 1, 51)
        view_204: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_460, [8, 128, 2]);  select_460 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_51: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_102, view_204, permute_51, alpha = -2.0);  unsqueeze_102 = view_204 = permute_51 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_51: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_51, -1);  baddbmm_51 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_103: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_51, -1)
        expand_102: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_103, [8, 128, 2]);  unsqueeze_103 = None
        gather_51: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_102);  expand_102 = None
        view_205: "f32[2048]" = torch.ops.aten.view.default(gather_51, [-1]);  gather_51 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_462: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_50, 1, 51)
        copy_51: "i64[8, 128]" = torch.ops.aten.copy.default(select_462, argmin_51);  select_462 = argmin_51 = None
        select_scatter_51: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_50, copy_51, 1, 51);  select_scatter_50 = copy_51 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_51: "f32[2048]" = torch.ops.aten.sub.Tensor(select_459, view_205);  select_459 = view_205 = None
        div_51: "f32[2048]" = torch.ops.aten.div.Tensor(sub_51, select_458);  sub_51 = select_458 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_464: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 51)
        slice_202: "f32[77]" = torch.ops.aten.slice.Tensor(select_464, 0, 51, 9223372036854775807);  select_464 = None
        slice_203: "f32[2048, 77]" = torch.ops.aten.slice.Tensor(slice_scatter_49, 1, 51, 9223372036854775807)
        expand_103: "f32[2048, 77]" = torch.ops.aten.expand.default(slice_203, [2048, 77]);  slice_203 = None
        mul_153: "f32[2048, 77]" = torch.ops.aten.mul.Tensor(expand_103, 1);  expand_103 = None
        view_206: "f32[2048, 1]" = torch.ops.aten.view.default(div_51, [2048, 1]);  div_51 = None
        mul_154: "f32[2048, 77]" = torch.ops.aten.mul.Tensor(view_206, slice_202);  view_206 = slice_202 = None
        mul_155: "f32[2048, 77]" = torch.ops.aten.mul.Tensor(mul_154, -1.0);  mul_154 = None
        add_51: "f32[2048, 77]" = torch.ops.aten.add.Tensor(mul_153, mul_155);  mul_153 = mul_155 = None
        slice_scatter_50: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_49, add_51, 1, 51, 9223372036854775807);  slice_scatter_49 = add_51 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_466: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 52)
        select_467: "f32[]" = torch.ops.aten.select.int(select_466, 0, 52);  select_466 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_468: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_50, 1, 52)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_104: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_52: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_469: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_50, 1, 52)
        view_208: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_469, [8, 128, 2]);  select_469 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_52: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_104, view_208, permute_52, alpha = -2.0);  unsqueeze_104 = view_208 = permute_52 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_52: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_52, -1);  baddbmm_52 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_105: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_52, -1)
        expand_104: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_105, [8, 128, 2]);  unsqueeze_105 = None
        gather_52: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_104);  expand_104 = None
        view_209: "f32[2048]" = torch.ops.aten.view.default(gather_52, [-1]);  gather_52 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_471: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_51, 1, 52)
        copy_52: "i64[8, 128]" = torch.ops.aten.copy.default(select_471, argmin_52);  select_471 = argmin_52 = None
        select_scatter_52: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_51, copy_52, 1, 52);  select_scatter_51 = copy_52 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_52: "f32[2048]" = torch.ops.aten.sub.Tensor(select_468, view_209);  select_468 = view_209 = None
        div_52: "f32[2048]" = torch.ops.aten.div.Tensor(sub_52, select_467);  sub_52 = select_467 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_473: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 52)
        slice_206: "f32[76]" = torch.ops.aten.slice.Tensor(select_473, 0, 52, 9223372036854775807);  select_473 = None
        slice_207: "f32[2048, 76]" = torch.ops.aten.slice.Tensor(slice_scatter_50, 1, 52, 9223372036854775807)
        expand_105: "f32[2048, 76]" = torch.ops.aten.expand.default(slice_207, [2048, 76]);  slice_207 = None
        mul_156: "f32[2048, 76]" = torch.ops.aten.mul.Tensor(expand_105, 1);  expand_105 = None
        view_210: "f32[2048, 1]" = torch.ops.aten.view.default(div_52, [2048, 1]);  div_52 = None
        mul_157: "f32[2048, 76]" = torch.ops.aten.mul.Tensor(view_210, slice_206);  view_210 = slice_206 = None
        mul_158: "f32[2048, 76]" = torch.ops.aten.mul.Tensor(mul_157, -1.0);  mul_157 = None
        add_52: "f32[2048, 76]" = torch.ops.aten.add.Tensor(mul_156, mul_158);  mul_156 = mul_158 = None
        slice_scatter_51: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_50, add_52, 1, 52, 9223372036854775807);  slice_scatter_50 = add_52 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_475: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 53)
        select_476: "f32[]" = torch.ops.aten.select.int(select_475, 0, 53);  select_475 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_477: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_51, 1, 53)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_106: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_53: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_478: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_51, 1, 53)
        view_212: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_478, [8, 128, 2]);  select_478 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_53: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_106, view_212, permute_53, alpha = -2.0);  unsqueeze_106 = view_212 = permute_53 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_53: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_53, -1);  baddbmm_53 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_107: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_53, -1)
        expand_106: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_107, [8, 128, 2]);  unsqueeze_107 = None
        gather_53: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_106);  expand_106 = None
        view_213: "f32[2048]" = torch.ops.aten.view.default(gather_53, [-1]);  gather_53 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_480: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_52, 1, 53)
        copy_53: "i64[8, 128]" = torch.ops.aten.copy.default(select_480, argmin_53);  select_480 = argmin_53 = None
        select_scatter_53: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_52, copy_53, 1, 53);  select_scatter_52 = copy_53 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_53: "f32[2048]" = torch.ops.aten.sub.Tensor(select_477, view_213);  select_477 = view_213 = None
        div_53: "f32[2048]" = torch.ops.aten.div.Tensor(sub_53, select_476);  sub_53 = select_476 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_482: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 53)
        slice_210: "f32[75]" = torch.ops.aten.slice.Tensor(select_482, 0, 53, 9223372036854775807);  select_482 = None
        slice_211: "f32[2048, 75]" = torch.ops.aten.slice.Tensor(slice_scatter_51, 1, 53, 9223372036854775807)
        expand_107: "f32[2048, 75]" = torch.ops.aten.expand.default(slice_211, [2048, 75]);  slice_211 = None
        mul_159: "f32[2048, 75]" = torch.ops.aten.mul.Tensor(expand_107, 1);  expand_107 = None
        view_214: "f32[2048, 1]" = torch.ops.aten.view.default(div_53, [2048, 1]);  div_53 = None
        mul_160: "f32[2048, 75]" = torch.ops.aten.mul.Tensor(view_214, slice_210);  view_214 = slice_210 = None
        mul_161: "f32[2048, 75]" = torch.ops.aten.mul.Tensor(mul_160, -1.0);  mul_160 = None
        add_53: "f32[2048, 75]" = torch.ops.aten.add.Tensor(mul_159, mul_161);  mul_159 = mul_161 = None
        slice_scatter_52: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_51, add_53, 1, 53, 9223372036854775807);  slice_scatter_51 = add_53 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_484: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 54)
        select_485: "f32[]" = torch.ops.aten.select.int(select_484, 0, 54);  select_484 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_486: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_52, 1, 54)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_108: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_54: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_487: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_52, 1, 54)
        view_216: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_487, [8, 128, 2]);  select_487 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_54: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_108, view_216, permute_54, alpha = -2.0);  unsqueeze_108 = view_216 = permute_54 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_54: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_54, -1);  baddbmm_54 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_109: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_54, -1)
        expand_108: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_109, [8, 128, 2]);  unsqueeze_109 = None
        gather_54: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_108);  expand_108 = None
        view_217: "f32[2048]" = torch.ops.aten.view.default(gather_54, [-1]);  gather_54 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_489: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_53, 1, 54)
        copy_54: "i64[8, 128]" = torch.ops.aten.copy.default(select_489, argmin_54);  select_489 = argmin_54 = None
        select_scatter_54: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_53, copy_54, 1, 54);  select_scatter_53 = copy_54 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_54: "f32[2048]" = torch.ops.aten.sub.Tensor(select_486, view_217);  select_486 = view_217 = None
        div_54: "f32[2048]" = torch.ops.aten.div.Tensor(sub_54, select_485);  sub_54 = select_485 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_491: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 54)
        slice_214: "f32[74]" = torch.ops.aten.slice.Tensor(select_491, 0, 54, 9223372036854775807);  select_491 = None
        slice_215: "f32[2048, 74]" = torch.ops.aten.slice.Tensor(slice_scatter_52, 1, 54, 9223372036854775807)
        expand_109: "f32[2048, 74]" = torch.ops.aten.expand.default(slice_215, [2048, 74]);  slice_215 = None
        mul_162: "f32[2048, 74]" = torch.ops.aten.mul.Tensor(expand_109, 1);  expand_109 = None
        view_218: "f32[2048, 1]" = torch.ops.aten.view.default(div_54, [2048, 1]);  div_54 = None
        mul_163: "f32[2048, 74]" = torch.ops.aten.mul.Tensor(view_218, slice_214);  view_218 = slice_214 = None
        mul_164: "f32[2048, 74]" = torch.ops.aten.mul.Tensor(mul_163, -1.0);  mul_163 = None
        add_54: "f32[2048, 74]" = torch.ops.aten.add.Tensor(mul_162, mul_164);  mul_162 = mul_164 = None
        slice_scatter_53: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_52, add_54, 1, 54, 9223372036854775807);  slice_scatter_52 = add_54 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_493: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 55)
        select_494: "f32[]" = torch.ops.aten.select.int(select_493, 0, 55);  select_493 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_495: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_53, 1, 55)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_110: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_55: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_496: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_53, 1, 55)
        view_220: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_496, [8, 128, 2]);  select_496 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_55: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_110, view_220, permute_55, alpha = -2.0);  unsqueeze_110 = view_220 = permute_55 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_55: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_55, -1);  baddbmm_55 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_111: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_55, -1)
        expand_110: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_111, [8, 128, 2]);  unsqueeze_111 = None
        gather_55: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_110);  expand_110 = None
        view_221: "f32[2048]" = torch.ops.aten.view.default(gather_55, [-1]);  gather_55 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_498: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_54, 1, 55)
        copy_55: "i64[8, 128]" = torch.ops.aten.copy.default(select_498, argmin_55);  select_498 = argmin_55 = None
        select_scatter_55: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_54, copy_55, 1, 55);  select_scatter_54 = copy_55 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_55: "f32[2048]" = torch.ops.aten.sub.Tensor(select_495, view_221);  select_495 = view_221 = None
        div_55: "f32[2048]" = torch.ops.aten.div.Tensor(sub_55, select_494);  sub_55 = select_494 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_500: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 55)
        slice_218: "f32[73]" = torch.ops.aten.slice.Tensor(select_500, 0, 55, 9223372036854775807);  select_500 = None
        slice_219: "f32[2048, 73]" = torch.ops.aten.slice.Tensor(slice_scatter_53, 1, 55, 9223372036854775807)
        expand_111: "f32[2048, 73]" = torch.ops.aten.expand.default(slice_219, [2048, 73]);  slice_219 = None
        mul_165: "f32[2048, 73]" = torch.ops.aten.mul.Tensor(expand_111, 1);  expand_111 = None
        view_222: "f32[2048, 1]" = torch.ops.aten.view.default(div_55, [2048, 1]);  div_55 = None
        mul_166: "f32[2048, 73]" = torch.ops.aten.mul.Tensor(view_222, slice_218);  view_222 = slice_218 = None
        mul_167: "f32[2048, 73]" = torch.ops.aten.mul.Tensor(mul_166, -1.0);  mul_166 = None
        add_55: "f32[2048, 73]" = torch.ops.aten.add.Tensor(mul_165, mul_167);  mul_165 = mul_167 = None
        slice_scatter_54: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_53, add_55, 1, 55, 9223372036854775807);  slice_scatter_53 = add_55 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_502: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 56)
        select_503: "f32[]" = torch.ops.aten.select.int(select_502, 0, 56);  select_502 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_504: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_54, 1, 56)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_112: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_56: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_505: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_54, 1, 56)
        view_224: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_505, [8, 128, 2]);  select_505 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_56: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_112, view_224, permute_56, alpha = -2.0);  unsqueeze_112 = view_224 = permute_56 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_56: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_56, -1);  baddbmm_56 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_113: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_56, -1)
        expand_112: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_113, [8, 128, 2]);  unsqueeze_113 = None
        gather_56: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_112);  expand_112 = None
        view_225: "f32[2048]" = torch.ops.aten.view.default(gather_56, [-1]);  gather_56 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_507: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_55, 1, 56)
        copy_56: "i64[8, 128]" = torch.ops.aten.copy.default(select_507, argmin_56);  select_507 = argmin_56 = None
        select_scatter_56: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_55, copy_56, 1, 56);  select_scatter_55 = copy_56 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_56: "f32[2048]" = torch.ops.aten.sub.Tensor(select_504, view_225);  select_504 = view_225 = None
        div_56: "f32[2048]" = torch.ops.aten.div.Tensor(sub_56, select_503);  sub_56 = select_503 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_509: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 56)
        slice_222: "f32[72]" = torch.ops.aten.slice.Tensor(select_509, 0, 56, 9223372036854775807);  select_509 = None
        slice_223: "f32[2048, 72]" = torch.ops.aten.slice.Tensor(slice_scatter_54, 1, 56, 9223372036854775807)
        expand_113: "f32[2048, 72]" = torch.ops.aten.expand.default(slice_223, [2048, 72]);  slice_223 = None
        mul_168: "f32[2048, 72]" = torch.ops.aten.mul.Tensor(expand_113, 1);  expand_113 = None
        view_226: "f32[2048, 1]" = torch.ops.aten.view.default(div_56, [2048, 1]);  div_56 = None
        mul_169: "f32[2048, 72]" = torch.ops.aten.mul.Tensor(view_226, slice_222);  view_226 = slice_222 = None
        mul_170: "f32[2048, 72]" = torch.ops.aten.mul.Tensor(mul_169, -1.0);  mul_169 = None
        add_56: "f32[2048, 72]" = torch.ops.aten.add.Tensor(mul_168, mul_170);  mul_168 = mul_170 = None
        slice_scatter_55: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_54, add_56, 1, 56, 9223372036854775807);  slice_scatter_54 = add_56 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_511: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 57)
        select_512: "f32[]" = torch.ops.aten.select.int(select_511, 0, 57);  select_511 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_513: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_55, 1, 57)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_114: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_57: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_514: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_55, 1, 57)
        view_228: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_514, [8, 128, 2]);  select_514 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_57: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_114, view_228, permute_57, alpha = -2.0);  unsqueeze_114 = view_228 = permute_57 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_57: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_57, -1);  baddbmm_57 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_115: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_57, -1)
        expand_114: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_115, [8, 128, 2]);  unsqueeze_115 = None
        gather_57: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_114);  expand_114 = None
        view_229: "f32[2048]" = torch.ops.aten.view.default(gather_57, [-1]);  gather_57 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_516: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_56, 1, 57)
        copy_57: "i64[8, 128]" = torch.ops.aten.copy.default(select_516, argmin_57);  select_516 = argmin_57 = None
        select_scatter_57: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_56, copy_57, 1, 57);  select_scatter_56 = copy_57 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_57: "f32[2048]" = torch.ops.aten.sub.Tensor(select_513, view_229);  select_513 = view_229 = None
        div_57: "f32[2048]" = torch.ops.aten.div.Tensor(sub_57, select_512);  sub_57 = select_512 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_518: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 57)
        slice_226: "f32[71]" = torch.ops.aten.slice.Tensor(select_518, 0, 57, 9223372036854775807);  select_518 = None
        slice_227: "f32[2048, 71]" = torch.ops.aten.slice.Tensor(slice_scatter_55, 1, 57, 9223372036854775807)
        expand_115: "f32[2048, 71]" = torch.ops.aten.expand.default(slice_227, [2048, 71]);  slice_227 = None
        mul_171: "f32[2048, 71]" = torch.ops.aten.mul.Tensor(expand_115, 1);  expand_115 = None
        view_230: "f32[2048, 1]" = torch.ops.aten.view.default(div_57, [2048, 1]);  div_57 = None
        mul_172: "f32[2048, 71]" = torch.ops.aten.mul.Tensor(view_230, slice_226);  view_230 = slice_226 = None
        mul_173: "f32[2048, 71]" = torch.ops.aten.mul.Tensor(mul_172, -1.0);  mul_172 = None
        add_57: "f32[2048, 71]" = torch.ops.aten.add.Tensor(mul_171, mul_173);  mul_171 = mul_173 = None
        slice_scatter_56: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_55, add_57, 1, 57, 9223372036854775807);  slice_scatter_55 = add_57 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_520: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 58)
        select_521: "f32[]" = torch.ops.aten.select.int(select_520, 0, 58);  select_520 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_522: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_56, 1, 58)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_116: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_58: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_523: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_56, 1, 58)
        view_232: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_523, [8, 128, 2]);  select_523 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_58: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_116, view_232, permute_58, alpha = -2.0);  unsqueeze_116 = view_232 = permute_58 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_58: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_58, -1);  baddbmm_58 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_117: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_58, -1)
        expand_116: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_117, [8, 128, 2]);  unsqueeze_117 = None
        gather_58: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_116);  expand_116 = None
        view_233: "f32[2048]" = torch.ops.aten.view.default(gather_58, [-1]);  gather_58 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_525: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_57, 1, 58)
        copy_58: "i64[8, 128]" = torch.ops.aten.copy.default(select_525, argmin_58);  select_525 = argmin_58 = None
        select_scatter_58: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_57, copy_58, 1, 58);  select_scatter_57 = copy_58 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_58: "f32[2048]" = torch.ops.aten.sub.Tensor(select_522, view_233);  select_522 = view_233 = None
        div_58: "f32[2048]" = torch.ops.aten.div.Tensor(sub_58, select_521);  sub_58 = select_521 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_527: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 58)
        slice_230: "f32[70]" = torch.ops.aten.slice.Tensor(select_527, 0, 58, 9223372036854775807);  select_527 = None
        slice_231: "f32[2048, 70]" = torch.ops.aten.slice.Tensor(slice_scatter_56, 1, 58, 9223372036854775807)
        expand_117: "f32[2048, 70]" = torch.ops.aten.expand.default(slice_231, [2048, 70]);  slice_231 = None
        mul_174: "f32[2048, 70]" = torch.ops.aten.mul.Tensor(expand_117, 1);  expand_117 = None
        view_234: "f32[2048, 1]" = torch.ops.aten.view.default(div_58, [2048, 1]);  div_58 = None
        mul_175: "f32[2048, 70]" = torch.ops.aten.mul.Tensor(view_234, slice_230);  view_234 = slice_230 = None
        mul_176: "f32[2048, 70]" = torch.ops.aten.mul.Tensor(mul_175, -1.0);  mul_175 = None
        add_58: "f32[2048, 70]" = torch.ops.aten.add.Tensor(mul_174, mul_176);  mul_174 = mul_176 = None
        slice_scatter_57: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_56, add_58, 1, 58, 9223372036854775807);  slice_scatter_56 = add_58 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_529: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 59)
        select_530: "f32[]" = torch.ops.aten.select.int(select_529, 0, 59);  select_529 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_531: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_57, 1, 59)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_118: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_59: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_532: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_57, 1, 59)
        view_236: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_532, [8, 128, 2]);  select_532 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_59: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_118, view_236, permute_59, alpha = -2.0);  unsqueeze_118 = view_236 = permute_59 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_59: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_59, -1);  baddbmm_59 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_119: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_59, -1)
        expand_118: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_119, [8, 128, 2]);  unsqueeze_119 = None
        gather_59: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_118);  expand_118 = None
        view_237: "f32[2048]" = torch.ops.aten.view.default(gather_59, [-1]);  gather_59 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_534: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_58, 1, 59)
        copy_59: "i64[8, 128]" = torch.ops.aten.copy.default(select_534, argmin_59);  select_534 = argmin_59 = None
        select_scatter_59: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_58, copy_59, 1, 59);  select_scatter_58 = copy_59 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_59: "f32[2048]" = torch.ops.aten.sub.Tensor(select_531, view_237);  select_531 = view_237 = None
        div_59: "f32[2048]" = torch.ops.aten.div.Tensor(sub_59, select_530);  sub_59 = select_530 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_536: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 59)
        slice_234: "f32[69]" = torch.ops.aten.slice.Tensor(select_536, 0, 59, 9223372036854775807);  select_536 = None
        slice_235: "f32[2048, 69]" = torch.ops.aten.slice.Tensor(slice_scatter_57, 1, 59, 9223372036854775807)
        expand_119: "f32[2048, 69]" = torch.ops.aten.expand.default(slice_235, [2048, 69]);  slice_235 = None
        mul_177: "f32[2048, 69]" = torch.ops.aten.mul.Tensor(expand_119, 1);  expand_119 = None
        view_238: "f32[2048, 1]" = torch.ops.aten.view.default(div_59, [2048, 1]);  div_59 = None
        mul_178: "f32[2048, 69]" = torch.ops.aten.mul.Tensor(view_238, slice_234);  view_238 = slice_234 = None
        mul_179: "f32[2048, 69]" = torch.ops.aten.mul.Tensor(mul_178, -1.0);  mul_178 = None
        add_59: "f32[2048, 69]" = torch.ops.aten.add.Tensor(mul_177, mul_179);  mul_177 = mul_179 = None
        slice_scatter_58: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_57, add_59, 1, 59, 9223372036854775807);  slice_scatter_57 = add_59 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_538: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 60)
        select_539: "f32[]" = torch.ops.aten.select.int(select_538, 0, 60);  select_538 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_540: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_58, 1, 60)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_120: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_60: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_541: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_58, 1, 60)
        view_240: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_541, [8, 128, 2]);  select_541 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_60: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_120, view_240, permute_60, alpha = -2.0);  unsqueeze_120 = view_240 = permute_60 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_60: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_60, -1);  baddbmm_60 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_121: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_60, -1)
        expand_120: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_121, [8, 128, 2]);  unsqueeze_121 = None
        gather_60: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_120);  expand_120 = None
        view_241: "f32[2048]" = torch.ops.aten.view.default(gather_60, [-1]);  gather_60 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_543: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_59, 1, 60)
        copy_60: "i64[8, 128]" = torch.ops.aten.copy.default(select_543, argmin_60);  select_543 = argmin_60 = None
        select_scatter_60: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_59, copy_60, 1, 60);  select_scatter_59 = copy_60 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_60: "f32[2048]" = torch.ops.aten.sub.Tensor(select_540, view_241);  select_540 = view_241 = None
        div_60: "f32[2048]" = torch.ops.aten.div.Tensor(sub_60, select_539);  sub_60 = select_539 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_545: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 60)
        slice_238: "f32[68]" = torch.ops.aten.slice.Tensor(select_545, 0, 60, 9223372036854775807);  select_545 = None
        slice_239: "f32[2048, 68]" = torch.ops.aten.slice.Tensor(slice_scatter_58, 1, 60, 9223372036854775807)
        expand_121: "f32[2048, 68]" = torch.ops.aten.expand.default(slice_239, [2048, 68]);  slice_239 = None
        mul_180: "f32[2048, 68]" = torch.ops.aten.mul.Tensor(expand_121, 1);  expand_121 = None
        view_242: "f32[2048, 1]" = torch.ops.aten.view.default(div_60, [2048, 1]);  div_60 = None
        mul_181: "f32[2048, 68]" = torch.ops.aten.mul.Tensor(view_242, slice_238);  view_242 = slice_238 = None
        mul_182: "f32[2048, 68]" = torch.ops.aten.mul.Tensor(mul_181, -1.0);  mul_181 = None
        add_60: "f32[2048, 68]" = torch.ops.aten.add.Tensor(mul_180, mul_182);  mul_180 = mul_182 = None
        slice_scatter_59: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_58, add_60, 1, 60, 9223372036854775807);  slice_scatter_58 = add_60 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_547: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 61)
        select_548: "f32[]" = torch.ops.aten.select.int(select_547, 0, 61);  select_547 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_549: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_59, 1, 61)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_122: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_61: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_550: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_59, 1, 61)
        view_244: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_550, [8, 128, 2]);  select_550 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_61: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_122, view_244, permute_61, alpha = -2.0);  unsqueeze_122 = view_244 = permute_61 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_61: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_61, -1);  baddbmm_61 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_123: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_61, -1)
        expand_122: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_123, [8, 128, 2]);  unsqueeze_123 = None
        gather_61: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_122);  expand_122 = None
        view_245: "f32[2048]" = torch.ops.aten.view.default(gather_61, [-1]);  gather_61 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_552: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_60, 1, 61)
        copy_61: "i64[8, 128]" = torch.ops.aten.copy.default(select_552, argmin_61);  select_552 = argmin_61 = None
        select_scatter_61: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_60, copy_61, 1, 61);  select_scatter_60 = copy_61 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_61: "f32[2048]" = torch.ops.aten.sub.Tensor(select_549, view_245);  select_549 = view_245 = None
        div_61: "f32[2048]" = torch.ops.aten.div.Tensor(sub_61, select_548);  sub_61 = select_548 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_554: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 61)
        slice_242: "f32[67]" = torch.ops.aten.slice.Tensor(select_554, 0, 61, 9223372036854775807);  select_554 = None
        slice_243: "f32[2048, 67]" = torch.ops.aten.slice.Tensor(slice_scatter_59, 1, 61, 9223372036854775807)
        expand_123: "f32[2048, 67]" = torch.ops.aten.expand.default(slice_243, [2048, 67]);  slice_243 = None
        mul_183: "f32[2048, 67]" = torch.ops.aten.mul.Tensor(expand_123, 1);  expand_123 = None
        view_246: "f32[2048, 1]" = torch.ops.aten.view.default(div_61, [2048, 1]);  div_61 = None
        mul_184: "f32[2048, 67]" = torch.ops.aten.mul.Tensor(view_246, slice_242);  view_246 = slice_242 = None
        mul_185: "f32[2048, 67]" = torch.ops.aten.mul.Tensor(mul_184, -1.0);  mul_184 = None
        add_61: "f32[2048, 67]" = torch.ops.aten.add.Tensor(mul_183, mul_185);  mul_183 = mul_185 = None
        slice_scatter_60: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_59, add_61, 1, 61, 9223372036854775807);  slice_scatter_59 = add_61 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_556: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 62)
        select_557: "f32[]" = torch.ops.aten.select.int(select_556, 0, 62);  select_556 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_558: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_60, 1, 62)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_124: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_62: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_559: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_60, 1, 62)
        view_248: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_559, [8, 128, 2]);  select_559 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_62: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_124, view_248, permute_62, alpha = -2.0);  unsqueeze_124 = view_248 = permute_62 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_62: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_62, -1);  baddbmm_62 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_125: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_62, -1)
        expand_124: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_125, [8, 128, 2]);  unsqueeze_125 = None
        gather_62: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_124);  expand_124 = None
        view_249: "f32[2048]" = torch.ops.aten.view.default(gather_62, [-1]);  gather_62 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_561: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_61, 1, 62)
        copy_62: "i64[8, 128]" = torch.ops.aten.copy.default(select_561, argmin_62);  select_561 = argmin_62 = None
        select_scatter_62: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_61, copy_62, 1, 62);  select_scatter_61 = copy_62 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_62: "f32[2048]" = torch.ops.aten.sub.Tensor(select_558, view_249);  select_558 = view_249 = None
        div_62: "f32[2048]" = torch.ops.aten.div.Tensor(sub_62, select_557);  sub_62 = select_557 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_563: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 62)
        slice_246: "f32[66]" = torch.ops.aten.slice.Tensor(select_563, 0, 62, 9223372036854775807);  select_563 = None
        slice_247: "f32[2048, 66]" = torch.ops.aten.slice.Tensor(slice_scatter_60, 1, 62, 9223372036854775807)
        expand_125: "f32[2048, 66]" = torch.ops.aten.expand.default(slice_247, [2048, 66]);  slice_247 = None
        mul_186: "f32[2048, 66]" = torch.ops.aten.mul.Tensor(expand_125, 1);  expand_125 = None
        view_250: "f32[2048, 1]" = torch.ops.aten.view.default(div_62, [2048, 1]);  div_62 = None
        mul_187: "f32[2048, 66]" = torch.ops.aten.mul.Tensor(view_250, slice_246);  view_250 = slice_246 = None
        mul_188: "f32[2048, 66]" = torch.ops.aten.mul.Tensor(mul_187, -1.0);  mul_187 = None
        add_62: "f32[2048, 66]" = torch.ops.aten.add.Tensor(mul_186, mul_188);  mul_186 = mul_188 = None
        slice_scatter_61: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_60, add_62, 1, 62, 9223372036854775807);  slice_scatter_60 = add_62 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_565: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 63)
        select_566: "f32[]" = torch.ops.aten.select.int(select_565, 0, 63);  select_565 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_567: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_61, 1, 63)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_126: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_63: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_568: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_61, 1, 63)
        view_252: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_568, [8, 128, 2]);  select_568 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_63: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_126, view_252, permute_63, alpha = -2.0);  unsqueeze_126 = view_252 = permute_63 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_63: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_63, -1);  baddbmm_63 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_127: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_63, -1)
        expand_126: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_127, [8, 128, 2]);  unsqueeze_127 = None
        gather_63: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_126);  expand_126 = None
        view_253: "f32[2048]" = torch.ops.aten.view.default(gather_63, [-1]);  gather_63 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_570: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_62, 1, 63)
        copy_63: "i64[8, 128]" = torch.ops.aten.copy.default(select_570, argmin_63);  select_570 = argmin_63 = None
        select_scatter_63: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_62, copy_63, 1, 63);  select_scatter_62 = copy_63 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_63: "f32[2048]" = torch.ops.aten.sub.Tensor(select_567, view_253);  select_567 = view_253 = None
        div_63: "f32[2048]" = torch.ops.aten.div.Tensor(sub_63, select_566);  sub_63 = select_566 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_572: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 63)
        slice_250: "f32[65]" = torch.ops.aten.slice.Tensor(select_572, 0, 63, 9223372036854775807);  select_572 = None
        slice_251: "f32[2048, 65]" = torch.ops.aten.slice.Tensor(slice_scatter_61, 1, 63, 9223372036854775807)
        expand_127: "f32[2048, 65]" = torch.ops.aten.expand.default(slice_251, [2048, 65]);  slice_251 = None
        mul_189: "f32[2048, 65]" = torch.ops.aten.mul.Tensor(expand_127, 1);  expand_127 = None
        view_254: "f32[2048, 1]" = torch.ops.aten.view.default(div_63, [2048, 1]);  div_63 = None
        mul_190: "f32[2048, 65]" = torch.ops.aten.mul.Tensor(view_254, slice_250);  view_254 = slice_250 = None
        mul_191: "f32[2048, 65]" = torch.ops.aten.mul.Tensor(mul_190, -1.0);  mul_190 = None
        add_63: "f32[2048, 65]" = torch.ops.aten.add.Tensor(mul_189, mul_191);  mul_189 = mul_191 = None
        slice_scatter_62: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_61, add_63, 1, 63, 9223372036854775807);  slice_scatter_61 = add_63 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_574: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 64)
        select_575: "f32[]" = torch.ops.aten.select.int(select_574, 0, 64);  select_574 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_576: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_62, 1, 64)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_128: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_64: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_577: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_62, 1, 64)
        view_256: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_577, [8, 128, 2]);  select_577 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_64: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_128, view_256, permute_64, alpha = -2.0);  unsqueeze_128 = view_256 = permute_64 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_64: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_64, -1);  baddbmm_64 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_129: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_64, -1)
        expand_128: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_129, [8, 128, 2]);  unsqueeze_129 = None
        gather_64: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_128);  expand_128 = None
        view_257: "f32[2048]" = torch.ops.aten.view.default(gather_64, [-1]);  gather_64 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_579: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_63, 1, 64)
        copy_64: "i64[8, 128]" = torch.ops.aten.copy.default(select_579, argmin_64);  select_579 = argmin_64 = None
        select_scatter_64: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_63, copy_64, 1, 64);  select_scatter_63 = copy_64 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_64: "f32[2048]" = torch.ops.aten.sub.Tensor(select_576, view_257);  select_576 = view_257 = None
        div_64: "f32[2048]" = torch.ops.aten.div.Tensor(sub_64, select_575);  sub_64 = select_575 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_581: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 64)
        slice_254: "f32[64]" = torch.ops.aten.slice.Tensor(select_581, 0, 64, 9223372036854775807);  select_581 = None
        slice_255: "f32[2048, 64]" = torch.ops.aten.slice.Tensor(slice_scatter_62, 1, 64, 9223372036854775807)
        expand_129: "f32[2048, 64]" = torch.ops.aten.expand.default(slice_255, [2048, 64]);  slice_255 = None
        mul_192: "f32[2048, 64]" = torch.ops.aten.mul.Tensor(expand_129, 1);  expand_129 = None
        view_258: "f32[2048, 1]" = torch.ops.aten.view.default(div_64, [2048, 1]);  div_64 = None
        mul_193: "f32[2048, 64]" = torch.ops.aten.mul.Tensor(view_258, slice_254);  view_258 = slice_254 = None
        mul_194: "f32[2048, 64]" = torch.ops.aten.mul.Tensor(mul_193, -1.0);  mul_193 = None
        add_64: "f32[2048, 64]" = torch.ops.aten.add.Tensor(mul_192, mul_194);  mul_192 = mul_194 = None
        slice_scatter_63: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_62, add_64, 1, 64, 9223372036854775807);  slice_scatter_62 = add_64 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_583: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 65)
        select_584: "f32[]" = torch.ops.aten.select.int(select_583, 0, 65);  select_583 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_585: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_63, 1, 65)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_130: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_65: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_586: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_63, 1, 65)
        view_260: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_586, [8, 128, 2]);  select_586 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_65: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_130, view_260, permute_65, alpha = -2.0);  unsqueeze_130 = view_260 = permute_65 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_65: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_65, -1);  baddbmm_65 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_131: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_65, -1)
        expand_130: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_131, [8, 128, 2]);  unsqueeze_131 = None
        gather_65: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_130);  expand_130 = None
        view_261: "f32[2048]" = torch.ops.aten.view.default(gather_65, [-1]);  gather_65 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_588: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_64, 1, 65)
        copy_65: "i64[8, 128]" = torch.ops.aten.copy.default(select_588, argmin_65);  select_588 = argmin_65 = None
        select_scatter_65: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_64, copy_65, 1, 65);  select_scatter_64 = copy_65 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_65: "f32[2048]" = torch.ops.aten.sub.Tensor(select_585, view_261);  select_585 = view_261 = None
        div_65: "f32[2048]" = torch.ops.aten.div.Tensor(sub_65, select_584);  sub_65 = select_584 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_590: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 65)
        slice_258: "f32[63]" = torch.ops.aten.slice.Tensor(select_590, 0, 65, 9223372036854775807);  select_590 = None
        slice_259: "f32[2048, 63]" = torch.ops.aten.slice.Tensor(slice_scatter_63, 1, 65, 9223372036854775807)
        expand_131: "f32[2048, 63]" = torch.ops.aten.expand.default(slice_259, [2048, 63]);  slice_259 = None
        mul_195: "f32[2048, 63]" = torch.ops.aten.mul.Tensor(expand_131, 1);  expand_131 = None
        view_262: "f32[2048, 1]" = torch.ops.aten.view.default(div_65, [2048, 1]);  div_65 = None
        mul_196: "f32[2048, 63]" = torch.ops.aten.mul.Tensor(view_262, slice_258);  view_262 = slice_258 = None
        mul_197: "f32[2048, 63]" = torch.ops.aten.mul.Tensor(mul_196, -1.0);  mul_196 = None
        add_65: "f32[2048, 63]" = torch.ops.aten.add.Tensor(mul_195, mul_197);  mul_195 = mul_197 = None
        slice_scatter_64: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_63, add_65, 1, 65, 9223372036854775807);  slice_scatter_63 = add_65 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_592: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 66)
        select_593: "f32[]" = torch.ops.aten.select.int(select_592, 0, 66);  select_592 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_594: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_64, 1, 66)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_132: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_66: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_595: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_64, 1, 66)
        view_264: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_595, [8, 128, 2]);  select_595 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_66: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_132, view_264, permute_66, alpha = -2.0);  unsqueeze_132 = view_264 = permute_66 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_66: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_66, -1);  baddbmm_66 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_133: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_66, -1)
        expand_132: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_133, [8, 128, 2]);  unsqueeze_133 = None
        gather_66: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_132);  expand_132 = None
        view_265: "f32[2048]" = torch.ops.aten.view.default(gather_66, [-1]);  gather_66 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_597: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_65, 1, 66)
        copy_66: "i64[8, 128]" = torch.ops.aten.copy.default(select_597, argmin_66);  select_597 = argmin_66 = None
        select_scatter_66: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_65, copy_66, 1, 66);  select_scatter_65 = copy_66 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_66: "f32[2048]" = torch.ops.aten.sub.Tensor(select_594, view_265);  select_594 = view_265 = None
        div_66: "f32[2048]" = torch.ops.aten.div.Tensor(sub_66, select_593);  sub_66 = select_593 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_599: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 66)
        slice_262: "f32[62]" = torch.ops.aten.slice.Tensor(select_599, 0, 66, 9223372036854775807);  select_599 = None
        slice_263: "f32[2048, 62]" = torch.ops.aten.slice.Tensor(slice_scatter_64, 1, 66, 9223372036854775807)
        expand_133: "f32[2048, 62]" = torch.ops.aten.expand.default(slice_263, [2048, 62]);  slice_263 = None
        mul_198: "f32[2048, 62]" = torch.ops.aten.mul.Tensor(expand_133, 1);  expand_133 = None
        view_266: "f32[2048, 1]" = torch.ops.aten.view.default(div_66, [2048, 1]);  div_66 = None
        mul_199: "f32[2048, 62]" = torch.ops.aten.mul.Tensor(view_266, slice_262);  view_266 = slice_262 = None
        mul_200: "f32[2048, 62]" = torch.ops.aten.mul.Tensor(mul_199, -1.0);  mul_199 = None
        add_66: "f32[2048, 62]" = torch.ops.aten.add.Tensor(mul_198, mul_200);  mul_198 = mul_200 = None
        slice_scatter_65: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_64, add_66, 1, 66, 9223372036854775807);  slice_scatter_64 = add_66 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_601: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 67)
        select_602: "f32[]" = torch.ops.aten.select.int(select_601, 0, 67);  select_601 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_603: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_65, 1, 67)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_134: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_67: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_604: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_65, 1, 67)
        view_268: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_604, [8, 128, 2]);  select_604 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_67: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_134, view_268, permute_67, alpha = -2.0);  unsqueeze_134 = view_268 = permute_67 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_67: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_67, -1);  baddbmm_67 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_135: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_67, -1)
        expand_134: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_135, [8, 128, 2]);  unsqueeze_135 = None
        gather_67: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_134);  expand_134 = None
        view_269: "f32[2048]" = torch.ops.aten.view.default(gather_67, [-1]);  gather_67 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_606: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_66, 1, 67)
        copy_67: "i64[8, 128]" = torch.ops.aten.copy.default(select_606, argmin_67);  select_606 = argmin_67 = None
        select_scatter_67: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_66, copy_67, 1, 67);  select_scatter_66 = copy_67 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_67: "f32[2048]" = torch.ops.aten.sub.Tensor(select_603, view_269);  select_603 = view_269 = None
        div_67: "f32[2048]" = torch.ops.aten.div.Tensor(sub_67, select_602);  sub_67 = select_602 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_608: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 67)
        slice_266: "f32[61]" = torch.ops.aten.slice.Tensor(select_608, 0, 67, 9223372036854775807);  select_608 = None
        slice_267: "f32[2048, 61]" = torch.ops.aten.slice.Tensor(slice_scatter_65, 1, 67, 9223372036854775807)
        expand_135: "f32[2048, 61]" = torch.ops.aten.expand.default(slice_267, [2048, 61]);  slice_267 = None
        mul_201: "f32[2048, 61]" = torch.ops.aten.mul.Tensor(expand_135, 1);  expand_135 = None
        view_270: "f32[2048, 1]" = torch.ops.aten.view.default(div_67, [2048, 1]);  div_67 = None
        mul_202: "f32[2048, 61]" = torch.ops.aten.mul.Tensor(view_270, slice_266);  view_270 = slice_266 = None
        mul_203: "f32[2048, 61]" = torch.ops.aten.mul.Tensor(mul_202, -1.0);  mul_202 = None
        add_67: "f32[2048, 61]" = torch.ops.aten.add.Tensor(mul_201, mul_203);  mul_201 = mul_203 = None
        slice_scatter_66: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_65, add_67, 1, 67, 9223372036854775807);  slice_scatter_65 = add_67 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_610: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 68)
        select_611: "f32[]" = torch.ops.aten.select.int(select_610, 0, 68);  select_610 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_612: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_66, 1, 68)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_136: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_68: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_613: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_66, 1, 68)
        view_272: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_613, [8, 128, 2]);  select_613 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_68: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_136, view_272, permute_68, alpha = -2.0);  unsqueeze_136 = view_272 = permute_68 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_68: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_68, -1);  baddbmm_68 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_137: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_68, -1)
        expand_136: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_137, [8, 128, 2]);  unsqueeze_137 = None
        gather_68: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_136);  expand_136 = None
        view_273: "f32[2048]" = torch.ops.aten.view.default(gather_68, [-1]);  gather_68 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_615: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_67, 1, 68)
        copy_68: "i64[8, 128]" = torch.ops.aten.copy.default(select_615, argmin_68);  select_615 = argmin_68 = None
        select_scatter_68: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_67, copy_68, 1, 68);  select_scatter_67 = copy_68 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_68: "f32[2048]" = torch.ops.aten.sub.Tensor(select_612, view_273);  select_612 = view_273 = None
        div_68: "f32[2048]" = torch.ops.aten.div.Tensor(sub_68, select_611);  sub_68 = select_611 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_617: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 68)
        slice_270: "f32[60]" = torch.ops.aten.slice.Tensor(select_617, 0, 68, 9223372036854775807);  select_617 = None
        slice_271: "f32[2048, 60]" = torch.ops.aten.slice.Tensor(slice_scatter_66, 1, 68, 9223372036854775807)
        expand_137: "f32[2048, 60]" = torch.ops.aten.expand.default(slice_271, [2048, 60]);  slice_271 = None
        mul_204: "f32[2048, 60]" = torch.ops.aten.mul.Tensor(expand_137, 1);  expand_137 = None
        view_274: "f32[2048, 1]" = torch.ops.aten.view.default(div_68, [2048, 1]);  div_68 = None
        mul_205: "f32[2048, 60]" = torch.ops.aten.mul.Tensor(view_274, slice_270);  view_274 = slice_270 = None
        mul_206: "f32[2048, 60]" = torch.ops.aten.mul.Tensor(mul_205, -1.0);  mul_205 = None
        add_68: "f32[2048, 60]" = torch.ops.aten.add.Tensor(mul_204, mul_206);  mul_204 = mul_206 = None
        slice_scatter_67: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_66, add_68, 1, 68, 9223372036854775807);  slice_scatter_66 = add_68 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_619: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 69)
        select_620: "f32[]" = torch.ops.aten.select.int(select_619, 0, 69);  select_619 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_621: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_67, 1, 69)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_138: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_69: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_622: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_67, 1, 69)
        view_276: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_622, [8, 128, 2]);  select_622 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_69: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_138, view_276, permute_69, alpha = -2.0);  unsqueeze_138 = view_276 = permute_69 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_69: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_69, -1);  baddbmm_69 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_139: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_69, -1)
        expand_138: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_139, [8, 128, 2]);  unsqueeze_139 = None
        gather_69: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_138);  expand_138 = None
        view_277: "f32[2048]" = torch.ops.aten.view.default(gather_69, [-1]);  gather_69 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_624: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_68, 1, 69)
        copy_69: "i64[8, 128]" = torch.ops.aten.copy.default(select_624, argmin_69);  select_624 = argmin_69 = None
        select_scatter_69: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_68, copy_69, 1, 69);  select_scatter_68 = copy_69 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_69: "f32[2048]" = torch.ops.aten.sub.Tensor(select_621, view_277);  select_621 = view_277 = None
        div_69: "f32[2048]" = torch.ops.aten.div.Tensor(sub_69, select_620);  sub_69 = select_620 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_626: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 69)
        slice_274: "f32[59]" = torch.ops.aten.slice.Tensor(select_626, 0, 69, 9223372036854775807);  select_626 = None
        slice_275: "f32[2048, 59]" = torch.ops.aten.slice.Tensor(slice_scatter_67, 1, 69, 9223372036854775807)
        expand_139: "f32[2048, 59]" = torch.ops.aten.expand.default(slice_275, [2048, 59]);  slice_275 = None
        mul_207: "f32[2048, 59]" = torch.ops.aten.mul.Tensor(expand_139, 1);  expand_139 = None
        view_278: "f32[2048, 1]" = torch.ops.aten.view.default(div_69, [2048, 1]);  div_69 = None
        mul_208: "f32[2048, 59]" = torch.ops.aten.mul.Tensor(view_278, slice_274);  view_278 = slice_274 = None
        mul_209: "f32[2048, 59]" = torch.ops.aten.mul.Tensor(mul_208, -1.0);  mul_208 = None
        add_69: "f32[2048, 59]" = torch.ops.aten.add.Tensor(mul_207, mul_209);  mul_207 = mul_209 = None
        slice_scatter_68: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_67, add_69, 1, 69, 9223372036854775807);  slice_scatter_67 = add_69 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_628: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 70)
        select_629: "f32[]" = torch.ops.aten.select.int(select_628, 0, 70);  select_628 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_630: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_68, 1, 70)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_140: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_70: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_631: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_68, 1, 70)
        view_280: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_631, [8, 128, 2]);  select_631 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_70: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_140, view_280, permute_70, alpha = -2.0);  unsqueeze_140 = view_280 = permute_70 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_70: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_70, -1);  baddbmm_70 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_141: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_70, -1)
        expand_140: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_141, [8, 128, 2]);  unsqueeze_141 = None
        gather_70: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_140);  expand_140 = None
        view_281: "f32[2048]" = torch.ops.aten.view.default(gather_70, [-1]);  gather_70 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_633: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_69, 1, 70)
        copy_70: "i64[8, 128]" = torch.ops.aten.copy.default(select_633, argmin_70);  select_633 = argmin_70 = None
        select_scatter_70: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_69, copy_70, 1, 70);  select_scatter_69 = copy_70 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_70: "f32[2048]" = torch.ops.aten.sub.Tensor(select_630, view_281);  select_630 = view_281 = None
        div_70: "f32[2048]" = torch.ops.aten.div.Tensor(sub_70, select_629);  sub_70 = select_629 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_635: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 70)
        slice_278: "f32[58]" = torch.ops.aten.slice.Tensor(select_635, 0, 70, 9223372036854775807);  select_635 = None
        slice_279: "f32[2048, 58]" = torch.ops.aten.slice.Tensor(slice_scatter_68, 1, 70, 9223372036854775807)
        expand_141: "f32[2048, 58]" = torch.ops.aten.expand.default(slice_279, [2048, 58]);  slice_279 = None
        mul_210: "f32[2048, 58]" = torch.ops.aten.mul.Tensor(expand_141, 1);  expand_141 = None
        view_282: "f32[2048, 1]" = torch.ops.aten.view.default(div_70, [2048, 1]);  div_70 = None
        mul_211: "f32[2048, 58]" = torch.ops.aten.mul.Tensor(view_282, slice_278);  view_282 = slice_278 = None
        mul_212: "f32[2048, 58]" = torch.ops.aten.mul.Tensor(mul_211, -1.0);  mul_211 = None
        add_70: "f32[2048, 58]" = torch.ops.aten.add.Tensor(mul_210, mul_212);  mul_210 = mul_212 = None
        slice_scatter_69: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_68, add_70, 1, 70, 9223372036854775807);  slice_scatter_68 = add_70 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_637: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 71)
        select_638: "f32[]" = torch.ops.aten.select.int(select_637, 0, 71);  select_637 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_639: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_69, 1, 71)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_142: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_71: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_640: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_69, 1, 71)
        view_284: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_640, [8, 128, 2]);  select_640 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_71: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_142, view_284, permute_71, alpha = -2.0);  unsqueeze_142 = view_284 = permute_71 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_71: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_71, -1);  baddbmm_71 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_143: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_71, -1)
        expand_142: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_143, [8, 128, 2]);  unsqueeze_143 = None
        gather_71: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_142);  expand_142 = None
        view_285: "f32[2048]" = torch.ops.aten.view.default(gather_71, [-1]);  gather_71 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_642: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_70, 1, 71)
        copy_71: "i64[8, 128]" = torch.ops.aten.copy.default(select_642, argmin_71);  select_642 = argmin_71 = None
        select_scatter_71: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_70, copy_71, 1, 71);  select_scatter_70 = copy_71 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_71: "f32[2048]" = torch.ops.aten.sub.Tensor(select_639, view_285);  select_639 = view_285 = None
        div_71: "f32[2048]" = torch.ops.aten.div.Tensor(sub_71, select_638);  sub_71 = select_638 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_644: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 71)
        slice_282: "f32[57]" = torch.ops.aten.slice.Tensor(select_644, 0, 71, 9223372036854775807);  select_644 = None
        slice_283: "f32[2048, 57]" = torch.ops.aten.slice.Tensor(slice_scatter_69, 1, 71, 9223372036854775807)
        expand_143: "f32[2048, 57]" = torch.ops.aten.expand.default(slice_283, [2048, 57]);  slice_283 = None
        mul_213: "f32[2048, 57]" = torch.ops.aten.mul.Tensor(expand_143, 1);  expand_143 = None
        view_286: "f32[2048, 1]" = torch.ops.aten.view.default(div_71, [2048, 1]);  div_71 = None
        mul_214: "f32[2048, 57]" = torch.ops.aten.mul.Tensor(view_286, slice_282);  view_286 = slice_282 = None
        mul_215: "f32[2048, 57]" = torch.ops.aten.mul.Tensor(mul_214, -1.0);  mul_214 = None
        add_71: "f32[2048, 57]" = torch.ops.aten.add.Tensor(mul_213, mul_215);  mul_213 = mul_215 = None
        slice_scatter_70: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_69, add_71, 1, 71, 9223372036854775807);  slice_scatter_69 = add_71 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_646: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 72)
        select_647: "f32[]" = torch.ops.aten.select.int(select_646, 0, 72);  select_646 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_648: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_70, 1, 72)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_144: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_72: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_649: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_70, 1, 72)
        view_288: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_649, [8, 128, 2]);  select_649 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_72: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_144, view_288, permute_72, alpha = -2.0);  unsqueeze_144 = view_288 = permute_72 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_72: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_72, -1);  baddbmm_72 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_145: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_72, -1)
        expand_144: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_145, [8, 128, 2]);  unsqueeze_145 = None
        gather_72: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_144);  expand_144 = None
        view_289: "f32[2048]" = torch.ops.aten.view.default(gather_72, [-1]);  gather_72 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_651: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_71, 1, 72)
        copy_72: "i64[8, 128]" = torch.ops.aten.copy.default(select_651, argmin_72);  select_651 = argmin_72 = None
        select_scatter_72: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_71, copy_72, 1, 72);  select_scatter_71 = copy_72 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_72: "f32[2048]" = torch.ops.aten.sub.Tensor(select_648, view_289);  select_648 = view_289 = None
        div_72: "f32[2048]" = torch.ops.aten.div.Tensor(sub_72, select_647);  sub_72 = select_647 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_653: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 72)
        slice_286: "f32[56]" = torch.ops.aten.slice.Tensor(select_653, 0, 72, 9223372036854775807);  select_653 = None
        slice_287: "f32[2048, 56]" = torch.ops.aten.slice.Tensor(slice_scatter_70, 1, 72, 9223372036854775807)
        expand_145: "f32[2048, 56]" = torch.ops.aten.expand.default(slice_287, [2048, 56]);  slice_287 = None
        mul_216: "f32[2048, 56]" = torch.ops.aten.mul.Tensor(expand_145, 1);  expand_145 = None
        view_290: "f32[2048, 1]" = torch.ops.aten.view.default(div_72, [2048, 1]);  div_72 = None
        mul_217: "f32[2048, 56]" = torch.ops.aten.mul.Tensor(view_290, slice_286);  view_290 = slice_286 = None
        mul_218: "f32[2048, 56]" = torch.ops.aten.mul.Tensor(mul_217, -1.0);  mul_217 = None
        add_72: "f32[2048, 56]" = torch.ops.aten.add.Tensor(mul_216, mul_218);  mul_216 = mul_218 = None
        slice_scatter_71: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_70, add_72, 1, 72, 9223372036854775807);  slice_scatter_70 = add_72 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_655: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 73)
        select_656: "f32[]" = torch.ops.aten.select.int(select_655, 0, 73);  select_655 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_657: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_71, 1, 73)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_146: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_73: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_658: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_71, 1, 73)
        view_292: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_658, [8, 128, 2]);  select_658 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_73: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_146, view_292, permute_73, alpha = -2.0);  unsqueeze_146 = view_292 = permute_73 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_73: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_73, -1);  baddbmm_73 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_147: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_73, -1)
        expand_146: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_147, [8, 128, 2]);  unsqueeze_147 = None
        gather_73: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_146);  expand_146 = None
        view_293: "f32[2048]" = torch.ops.aten.view.default(gather_73, [-1]);  gather_73 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_660: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_72, 1, 73)
        copy_73: "i64[8, 128]" = torch.ops.aten.copy.default(select_660, argmin_73);  select_660 = argmin_73 = None
        select_scatter_73: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_72, copy_73, 1, 73);  select_scatter_72 = copy_73 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_73: "f32[2048]" = torch.ops.aten.sub.Tensor(select_657, view_293);  select_657 = view_293 = None
        div_73: "f32[2048]" = torch.ops.aten.div.Tensor(sub_73, select_656);  sub_73 = select_656 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_662: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 73)
        slice_290: "f32[55]" = torch.ops.aten.slice.Tensor(select_662, 0, 73, 9223372036854775807);  select_662 = None
        slice_291: "f32[2048, 55]" = torch.ops.aten.slice.Tensor(slice_scatter_71, 1, 73, 9223372036854775807)
        expand_147: "f32[2048, 55]" = torch.ops.aten.expand.default(slice_291, [2048, 55]);  slice_291 = None
        mul_219: "f32[2048, 55]" = torch.ops.aten.mul.Tensor(expand_147, 1);  expand_147 = None
        view_294: "f32[2048, 1]" = torch.ops.aten.view.default(div_73, [2048, 1]);  div_73 = None
        mul_220: "f32[2048, 55]" = torch.ops.aten.mul.Tensor(view_294, slice_290);  view_294 = slice_290 = None
        mul_221: "f32[2048, 55]" = torch.ops.aten.mul.Tensor(mul_220, -1.0);  mul_220 = None
        add_73: "f32[2048, 55]" = torch.ops.aten.add.Tensor(mul_219, mul_221);  mul_219 = mul_221 = None
        slice_scatter_72: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_71, add_73, 1, 73, 9223372036854775807);  slice_scatter_71 = add_73 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_664: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 74)
        select_665: "f32[]" = torch.ops.aten.select.int(select_664, 0, 74);  select_664 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_666: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_72, 1, 74)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_148: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_74: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_667: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_72, 1, 74)
        view_296: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_667, [8, 128, 2]);  select_667 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_74: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_148, view_296, permute_74, alpha = -2.0);  unsqueeze_148 = view_296 = permute_74 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_74: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_74, -1);  baddbmm_74 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_149: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_74, -1)
        expand_148: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_149, [8, 128, 2]);  unsqueeze_149 = None
        gather_74: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_148);  expand_148 = None
        view_297: "f32[2048]" = torch.ops.aten.view.default(gather_74, [-1]);  gather_74 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_669: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_73, 1, 74)
        copy_74: "i64[8, 128]" = torch.ops.aten.copy.default(select_669, argmin_74);  select_669 = argmin_74 = None
        select_scatter_74: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_73, copy_74, 1, 74);  select_scatter_73 = copy_74 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_74: "f32[2048]" = torch.ops.aten.sub.Tensor(select_666, view_297);  select_666 = view_297 = None
        div_74: "f32[2048]" = torch.ops.aten.div.Tensor(sub_74, select_665);  sub_74 = select_665 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_671: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 74)
        slice_294: "f32[54]" = torch.ops.aten.slice.Tensor(select_671, 0, 74, 9223372036854775807);  select_671 = None
        slice_295: "f32[2048, 54]" = torch.ops.aten.slice.Tensor(slice_scatter_72, 1, 74, 9223372036854775807)
        expand_149: "f32[2048, 54]" = torch.ops.aten.expand.default(slice_295, [2048, 54]);  slice_295 = None
        mul_222: "f32[2048, 54]" = torch.ops.aten.mul.Tensor(expand_149, 1);  expand_149 = None
        view_298: "f32[2048, 1]" = torch.ops.aten.view.default(div_74, [2048, 1]);  div_74 = None
        mul_223: "f32[2048, 54]" = torch.ops.aten.mul.Tensor(view_298, slice_294);  view_298 = slice_294 = None
        mul_224: "f32[2048, 54]" = torch.ops.aten.mul.Tensor(mul_223, -1.0);  mul_223 = None
        add_74: "f32[2048, 54]" = torch.ops.aten.add.Tensor(mul_222, mul_224);  mul_222 = mul_224 = None
        slice_scatter_73: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_72, add_74, 1, 74, 9223372036854775807);  slice_scatter_72 = add_74 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_673: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 75)
        select_674: "f32[]" = torch.ops.aten.select.int(select_673, 0, 75);  select_673 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_675: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_73, 1, 75)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_150: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_75: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_676: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_73, 1, 75)
        view_300: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_676, [8, 128, 2]);  select_676 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_75: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_150, view_300, permute_75, alpha = -2.0);  unsqueeze_150 = view_300 = permute_75 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_75: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_75, -1);  baddbmm_75 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_151: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_75, -1)
        expand_150: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_151, [8, 128, 2]);  unsqueeze_151 = None
        gather_75: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_150);  expand_150 = None
        view_301: "f32[2048]" = torch.ops.aten.view.default(gather_75, [-1]);  gather_75 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_678: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_74, 1, 75)
        copy_75: "i64[8, 128]" = torch.ops.aten.copy.default(select_678, argmin_75);  select_678 = argmin_75 = None
        select_scatter_75: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_74, copy_75, 1, 75);  select_scatter_74 = copy_75 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_75: "f32[2048]" = torch.ops.aten.sub.Tensor(select_675, view_301);  select_675 = view_301 = None
        div_75: "f32[2048]" = torch.ops.aten.div.Tensor(sub_75, select_674);  sub_75 = select_674 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_680: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 75)
        slice_298: "f32[53]" = torch.ops.aten.slice.Tensor(select_680, 0, 75, 9223372036854775807);  select_680 = None
        slice_299: "f32[2048, 53]" = torch.ops.aten.slice.Tensor(slice_scatter_73, 1, 75, 9223372036854775807)
        expand_151: "f32[2048, 53]" = torch.ops.aten.expand.default(slice_299, [2048, 53]);  slice_299 = None
        mul_225: "f32[2048, 53]" = torch.ops.aten.mul.Tensor(expand_151, 1);  expand_151 = None
        view_302: "f32[2048, 1]" = torch.ops.aten.view.default(div_75, [2048, 1]);  div_75 = None
        mul_226: "f32[2048, 53]" = torch.ops.aten.mul.Tensor(view_302, slice_298);  view_302 = slice_298 = None
        mul_227: "f32[2048, 53]" = torch.ops.aten.mul.Tensor(mul_226, -1.0);  mul_226 = None
        add_75: "f32[2048, 53]" = torch.ops.aten.add.Tensor(mul_225, mul_227);  mul_225 = mul_227 = None
        slice_scatter_74: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_73, add_75, 1, 75, 9223372036854775807);  slice_scatter_73 = add_75 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_682: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 76)
        select_683: "f32[]" = torch.ops.aten.select.int(select_682, 0, 76);  select_682 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_684: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_74, 1, 76)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_152: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_76: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_685: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_74, 1, 76)
        view_304: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_685, [8, 128, 2]);  select_685 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_76: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_152, view_304, permute_76, alpha = -2.0);  unsqueeze_152 = view_304 = permute_76 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_76: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_76, -1);  baddbmm_76 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_153: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_76, -1)
        expand_152: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_153, [8, 128, 2]);  unsqueeze_153 = None
        gather_76: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_152);  expand_152 = None
        view_305: "f32[2048]" = torch.ops.aten.view.default(gather_76, [-1]);  gather_76 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_687: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_75, 1, 76)
        copy_76: "i64[8, 128]" = torch.ops.aten.copy.default(select_687, argmin_76);  select_687 = argmin_76 = None
        select_scatter_76: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_75, copy_76, 1, 76);  select_scatter_75 = copy_76 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_76: "f32[2048]" = torch.ops.aten.sub.Tensor(select_684, view_305);  select_684 = view_305 = None
        div_76: "f32[2048]" = torch.ops.aten.div.Tensor(sub_76, select_683);  sub_76 = select_683 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_689: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 76)
        slice_302: "f32[52]" = torch.ops.aten.slice.Tensor(select_689, 0, 76, 9223372036854775807);  select_689 = None
        slice_303: "f32[2048, 52]" = torch.ops.aten.slice.Tensor(slice_scatter_74, 1, 76, 9223372036854775807)
        expand_153: "f32[2048, 52]" = torch.ops.aten.expand.default(slice_303, [2048, 52]);  slice_303 = None
        mul_228: "f32[2048, 52]" = torch.ops.aten.mul.Tensor(expand_153, 1);  expand_153 = None
        view_306: "f32[2048, 1]" = torch.ops.aten.view.default(div_76, [2048, 1]);  div_76 = None
        mul_229: "f32[2048, 52]" = torch.ops.aten.mul.Tensor(view_306, slice_302);  view_306 = slice_302 = None
        mul_230: "f32[2048, 52]" = torch.ops.aten.mul.Tensor(mul_229, -1.0);  mul_229 = None
        add_76: "f32[2048, 52]" = torch.ops.aten.add.Tensor(mul_228, mul_230);  mul_228 = mul_230 = None
        slice_scatter_75: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_74, add_76, 1, 76, 9223372036854775807);  slice_scatter_74 = add_76 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_691: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 77)
        select_692: "f32[]" = torch.ops.aten.select.int(select_691, 0, 77);  select_691 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_693: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_75, 1, 77)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_154: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_77: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_694: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_75, 1, 77)
        view_308: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_694, [8, 128, 2]);  select_694 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_77: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_154, view_308, permute_77, alpha = -2.0);  unsqueeze_154 = view_308 = permute_77 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_77: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_77, -1);  baddbmm_77 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_155: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_77, -1)
        expand_154: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_155, [8, 128, 2]);  unsqueeze_155 = None
        gather_77: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_154);  expand_154 = None
        view_309: "f32[2048]" = torch.ops.aten.view.default(gather_77, [-1]);  gather_77 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_696: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_76, 1, 77)
        copy_77: "i64[8, 128]" = torch.ops.aten.copy.default(select_696, argmin_77);  select_696 = argmin_77 = None
        select_scatter_77: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_76, copy_77, 1, 77);  select_scatter_76 = copy_77 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_77: "f32[2048]" = torch.ops.aten.sub.Tensor(select_693, view_309);  select_693 = view_309 = None
        div_77: "f32[2048]" = torch.ops.aten.div.Tensor(sub_77, select_692);  sub_77 = select_692 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_698: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 77)
        slice_306: "f32[51]" = torch.ops.aten.slice.Tensor(select_698, 0, 77, 9223372036854775807);  select_698 = None
        slice_307: "f32[2048, 51]" = torch.ops.aten.slice.Tensor(slice_scatter_75, 1, 77, 9223372036854775807)
        expand_155: "f32[2048, 51]" = torch.ops.aten.expand.default(slice_307, [2048, 51]);  slice_307 = None
        mul_231: "f32[2048, 51]" = torch.ops.aten.mul.Tensor(expand_155, 1);  expand_155 = None
        view_310: "f32[2048, 1]" = torch.ops.aten.view.default(div_77, [2048, 1]);  div_77 = None
        mul_232: "f32[2048, 51]" = torch.ops.aten.mul.Tensor(view_310, slice_306);  view_310 = slice_306 = None
        mul_233: "f32[2048, 51]" = torch.ops.aten.mul.Tensor(mul_232, -1.0);  mul_232 = None
        add_77: "f32[2048, 51]" = torch.ops.aten.add.Tensor(mul_231, mul_233);  mul_231 = mul_233 = None
        slice_scatter_76: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_75, add_77, 1, 77, 9223372036854775807);  slice_scatter_75 = add_77 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_700: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 78)
        select_701: "f32[]" = torch.ops.aten.select.int(select_700, 0, 78);  select_700 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_702: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_76, 1, 78)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_156: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_78: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_703: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_76, 1, 78)
        view_312: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_703, [8, 128, 2]);  select_703 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_78: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_156, view_312, permute_78, alpha = -2.0);  unsqueeze_156 = view_312 = permute_78 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_78: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_78, -1);  baddbmm_78 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_157: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_78, -1)
        expand_156: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_157, [8, 128, 2]);  unsqueeze_157 = None
        gather_78: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_156);  expand_156 = None
        view_313: "f32[2048]" = torch.ops.aten.view.default(gather_78, [-1]);  gather_78 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_705: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_77, 1, 78)
        copy_78: "i64[8, 128]" = torch.ops.aten.copy.default(select_705, argmin_78);  select_705 = argmin_78 = None
        select_scatter_78: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_77, copy_78, 1, 78);  select_scatter_77 = copy_78 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_78: "f32[2048]" = torch.ops.aten.sub.Tensor(select_702, view_313);  select_702 = view_313 = None
        div_78: "f32[2048]" = torch.ops.aten.div.Tensor(sub_78, select_701);  sub_78 = select_701 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_707: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 78)
        slice_310: "f32[50]" = torch.ops.aten.slice.Tensor(select_707, 0, 78, 9223372036854775807);  select_707 = None
        slice_311: "f32[2048, 50]" = torch.ops.aten.slice.Tensor(slice_scatter_76, 1, 78, 9223372036854775807)
        expand_157: "f32[2048, 50]" = torch.ops.aten.expand.default(slice_311, [2048, 50]);  slice_311 = None
        mul_234: "f32[2048, 50]" = torch.ops.aten.mul.Tensor(expand_157, 1);  expand_157 = None
        view_314: "f32[2048, 1]" = torch.ops.aten.view.default(div_78, [2048, 1]);  div_78 = None
        mul_235: "f32[2048, 50]" = torch.ops.aten.mul.Tensor(view_314, slice_310);  view_314 = slice_310 = None
        mul_236: "f32[2048, 50]" = torch.ops.aten.mul.Tensor(mul_235, -1.0);  mul_235 = None
        add_78: "f32[2048, 50]" = torch.ops.aten.add.Tensor(mul_234, mul_236);  mul_234 = mul_236 = None
        slice_scatter_77: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_76, add_78, 1, 78, 9223372036854775807);  slice_scatter_76 = add_78 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_709: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 79)
        select_710: "f32[]" = torch.ops.aten.select.int(select_709, 0, 79);  select_709 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_711: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_77, 1, 79)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_158: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_79: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_712: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_77, 1, 79)
        view_316: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_712, [8, 128, 2]);  select_712 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_79: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_158, view_316, permute_79, alpha = -2.0);  unsqueeze_158 = view_316 = permute_79 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_79: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_79, -1);  baddbmm_79 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_159: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_79, -1)
        expand_158: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_159, [8, 128, 2]);  unsqueeze_159 = None
        gather_79: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_158);  expand_158 = None
        view_317: "f32[2048]" = torch.ops.aten.view.default(gather_79, [-1]);  gather_79 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_714: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_78, 1, 79)
        copy_79: "i64[8, 128]" = torch.ops.aten.copy.default(select_714, argmin_79);  select_714 = argmin_79 = None
        select_scatter_79: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_78, copy_79, 1, 79);  select_scatter_78 = copy_79 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_79: "f32[2048]" = torch.ops.aten.sub.Tensor(select_711, view_317);  select_711 = view_317 = None
        div_79: "f32[2048]" = torch.ops.aten.div.Tensor(sub_79, select_710);  sub_79 = select_710 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_716: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 79)
        slice_314: "f32[49]" = torch.ops.aten.slice.Tensor(select_716, 0, 79, 9223372036854775807);  select_716 = None
        slice_315: "f32[2048, 49]" = torch.ops.aten.slice.Tensor(slice_scatter_77, 1, 79, 9223372036854775807)
        expand_159: "f32[2048, 49]" = torch.ops.aten.expand.default(slice_315, [2048, 49]);  slice_315 = None
        mul_237: "f32[2048, 49]" = torch.ops.aten.mul.Tensor(expand_159, 1);  expand_159 = None
        view_318: "f32[2048, 1]" = torch.ops.aten.view.default(div_79, [2048, 1]);  div_79 = None
        mul_238: "f32[2048, 49]" = torch.ops.aten.mul.Tensor(view_318, slice_314);  view_318 = slice_314 = None
        mul_239: "f32[2048, 49]" = torch.ops.aten.mul.Tensor(mul_238, -1.0);  mul_238 = None
        add_79: "f32[2048, 49]" = torch.ops.aten.add.Tensor(mul_237, mul_239);  mul_237 = mul_239 = None
        slice_scatter_78: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_77, add_79, 1, 79, 9223372036854775807);  slice_scatter_77 = add_79 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_718: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 80)
        select_719: "f32[]" = torch.ops.aten.select.int(select_718, 0, 80);  select_718 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_720: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_78, 1, 80)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_160: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_80: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_721: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_78, 1, 80)
        view_320: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_721, [8, 128, 2]);  select_721 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_80: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_160, view_320, permute_80, alpha = -2.0);  unsqueeze_160 = view_320 = permute_80 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_80: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_80, -1);  baddbmm_80 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_161: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_80, -1)
        expand_160: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_161, [8, 128, 2]);  unsqueeze_161 = None
        gather_80: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_160);  expand_160 = None
        view_321: "f32[2048]" = torch.ops.aten.view.default(gather_80, [-1]);  gather_80 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_723: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_79, 1, 80)
        copy_80: "i64[8, 128]" = torch.ops.aten.copy.default(select_723, argmin_80);  select_723 = argmin_80 = None
        select_scatter_80: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_79, copy_80, 1, 80);  select_scatter_79 = copy_80 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_80: "f32[2048]" = torch.ops.aten.sub.Tensor(select_720, view_321);  select_720 = view_321 = None
        div_80: "f32[2048]" = torch.ops.aten.div.Tensor(sub_80, select_719);  sub_80 = select_719 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_725: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 80)
        slice_318: "f32[48]" = torch.ops.aten.slice.Tensor(select_725, 0, 80, 9223372036854775807);  select_725 = None
        slice_319: "f32[2048, 48]" = torch.ops.aten.slice.Tensor(slice_scatter_78, 1, 80, 9223372036854775807)
        expand_161: "f32[2048, 48]" = torch.ops.aten.expand.default(slice_319, [2048, 48]);  slice_319 = None
        mul_240: "f32[2048, 48]" = torch.ops.aten.mul.Tensor(expand_161, 1);  expand_161 = None
        view_322: "f32[2048, 1]" = torch.ops.aten.view.default(div_80, [2048, 1]);  div_80 = None
        mul_241: "f32[2048, 48]" = torch.ops.aten.mul.Tensor(view_322, slice_318);  view_322 = slice_318 = None
        mul_242: "f32[2048, 48]" = torch.ops.aten.mul.Tensor(mul_241, -1.0);  mul_241 = None
        add_80: "f32[2048, 48]" = torch.ops.aten.add.Tensor(mul_240, mul_242);  mul_240 = mul_242 = None
        slice_scatter_79: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_78, add_80, 1, 80, 9223372036854775807);  slice_scatter_78 = add_80 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_727: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 81)
        select_728: "f32[]" = torch.ops.aten.select.int(select_727, 0, 81);  select_727 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_729: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_79, 1, 81)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_162: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_81: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_730: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_79, 1, 81)
        view_324: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_730, [8, 128, 2]);  select_730 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_81: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_162, view_324, permute_81, alpha = -2.0);  unsqueeze_162 = view_324 = permute_81 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_81: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_81, -1);  baddbmm_81 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_163: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_81, -1)
        expand_162: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_163, [8, 128, 2]);  unsqueeze_163 = None
        gather_81: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_162);  expand_162 = None
        view_325: "f32[2048]" = torch.ops.aten.view.default(gather_81, [-1]);  gather_81 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_732: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_80, 1, 81)
        copy_81: "i64[8, 128]" = torch.ops.aten.copy.default(select_732, argmin_81);  select_732 = argmin_81 = None
        select_scatter_81: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_80, copy_81, 1, 81);  select_scatter_80 = copy_81 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_81: "f32[2048]" = torch.ops.aten.sub.Tensor(select_729, view_325);  select_729 = view_325 = None
        div_81: "f32[2048]" = torch.ops.aten.div.Tensor(sub_81, select_728);  sub_81 = select_728 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_734: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 81)
        slice_322: "f32[47]" = torch.ops.aten.slice.Tensor(select_734, 0, 81, 9223372036854775807);  select_734 = None
        slice_323: "f32[2048, 47]" = torch.ops.aten.slice.Tensor(slice_scatter_79, 1, 81, 9223372036854775807)
        expand_163: "f32[2048, 47]" = torch.ops.aten.expand.default(slice_323, [2048, 47]);  slice_323 = None
        mul_243: "f32[2048, 47]" = torch.ops.aten.mul.Tensor(expand_163, 1);  expand_163 = None
        view_326: "f32[2048, 1]" = torch.ops.aten.view.default(div_81, [2048, 1]);  div_81 = None
        mul_244: "f32[2048, 47]" = torch.ops.aten.mul.Tensor(view_326, slice_322);  view_326 = slice_322 = None
        mul_245: "f32[2048, 47]" = torch.ops.aten.mul.Tensor(mul_244, -1.0);  mul_244 = None
        add_81: "f32[2048, 47]" = torch.ops.aten.add.Tensor(mul_243, mul_245);  mul_243 = mul_245 = None
        slice_scatter_80: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_79, add_81, 1, 81, 9223372036854775807);  slice_scatter_79 = add_81 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_736: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 82)
        select_737: "f32[]" = torch.ops.aten.select.int(select_736, 0, 82);  select_736 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_738: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_80, 1, 82)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_164: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_82: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_739: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_80, 1, 82)
        view_328: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_739, [8, 128, 2]);  select_739 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_82: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_164, view_328, permute_82, alpha = -2.0);  unsqueeze_164 = view_328 = permute_82 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_82: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_82, -1);  baddbmm_82 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_165: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_82, -1)
        expand_164: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_165, [8, 128, 2]);  unsqueeze_165 = None
        gather_82: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_164);  expand_164 = None
        view_329: "f32[2048]" = torch.ops.aten.view.default(gather_82, [-1]);  gather_82 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_741: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_81, 1, 82)
        copy_82: "i64[8, 128]" = torch.ops.aten.copy.default(select_741, argmin_82);  select_741 = argmin_82 = None
        select_scatter_82: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_81, copy_82, 1, 82);  select_scatter_81 = copy_82 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_82: "f32[2048]" = torch.ops.aten.sub.Tensor(select_738, view_329);  select_738 = view_329 = None
        div_82: "f32[2048]" = torch.ops.aten.div.Tensor(sub_82, select_737);  sub_82 = select_737 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_743: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 82)
        slice_326: "f32[46]" = torch.ops.aten.slice.Tensor(select_743, 0, 82, 9223372036854775807);  select_743 = None
        slice_327: "f32[2048, 46]" = torch.ops.aten.slice.Tensor(slice_scatter_80, 1, 82, 9223372036854775807)
        expand_165: "f32[2048, 46]" = torch.ops.aten.expand.default(slice_327, [2048, 46]);  slice_327 = None
        mul_246: "f32[2048, 46]" = torch.ops.aten.mul.Tensor(expand_165, 1);  expand_165 = None
        view_330: "f32[2048, 1]" = torch.ops.aten.view.default(div_82, [2048, 1]);  div_82 = None
        mul_247: "f32[2048, 46]" = torch.ops.aten.mul.Tensor(view_330, slice_326);  view_330 = slice_326 = None
        mul_248: "f32[2048, 46]" = torch.ops.aten.mul.Tensor(mul_247, -1.0);  mul_247 = None
        add_82: "f32[2048, 46]" = torch.ops.aten.add.Tensor(mul_246, mul_248);  mul_246 = mul_248 = None
        slice_scatter_81: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_80, add_82, 1, 82, 9223372036854775807);  slice_scatter_80 = add_82 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_745: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 83)
        select_746: "f32[]" = torch.ops.aten.select.int(select_745, 0, 83);  select_745 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_747: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_81, 1, 83)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_166: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_83: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_748: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_81, 1, 83)
        view_332: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_748, [8, 128, 2]);  select_748 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_83: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_166, view_332, permute_83, alpha = -2.0);  unsqueeze_166 = view_332 = permute_83 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_83: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_83, -1);  baddbmm_83 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_167: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_83, -1)
        expand_166: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_167, [8, 128, 2]);  unsqueeze_167 = None
        gather_83: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_166);  expand_166 = None
        view_333: "f32[2048]" = torch.ops.aten.view.default(gather_83, [-1]);  gather_83 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_750: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_82, 1, 83)
        copy_83: "i64[8, 128]" = torch.ops.aten.copy.default(select_750, argmin_83);  select_750 = argmin_83 = None
        select_scatter_83: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_82, copy_83, 1, 83);  select_scatter_82 = copy_83 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_83: "f32[2048]" = torch.ops.aten.sub.Tensor(select_747, view_333);  select_747 = view_333 = None
        div_83: "f32[2048]" = torch.ops.aten.div.Tensor(sub_83, select_746);  sub_83 = select_746 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_752: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 83)
        slice_330: "f32[45]" = torch.ops.aten.slice.Tensor(select_752, 0, 83, 9223372036854775807);  select_752 = None
        slice_331: "f32[2048, 45]" = torch.ops.aten.slice.Tensor(slice_scatter_81, 1, 83, 9223372036854775807)
        expand_167: "f32[2048, 45]" = torch.ops.aten.expand.default(slice_331, [2048, 45]);  slice_331 = None
        mul_249: "f32[2048, 45]" = torch.ops.aten.mul.Tensor(expand_167, 1);  expand_167 = None
        view_334: "f32[2048, 1]" = torch.ops.aten.view.default(div_83, [2048, 1]);  div_83 = None
        mul_250: "f32[2048, 45]" = torch.ops.aten.mul.Tensor(view_334, slice_330);  view_334 = slice_330 = None
        mul_251: "f32[2048, 45]" = torch.ops.aten.mul.Tensor(mul_250, -1.0);  mul_250 = None
        add_83: "f32[2048, 45]" = torch.ops.aten.add.Tensor(mul_249, mul_251);  mul_249 = mul_251 = None
        slice_scatter_82: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_81, add_83, 1, 83, 9223372036854775807);  slice_scatter_81 = add_83 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_754: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 84)
        select_755: "f32[]" = torch.ops.aten.select.int(select_754, 0, 84);  select_754 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_756: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_82, 1, 84)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_168: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_84: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_757: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_82, 1, 84)
        view_336: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_757, [8, 128, 2]);  select_757 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_84: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_168, view_336, permute_84, alpha = -2.0);  unsqueeze_168 = view_336 = permute_84 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_84: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_84, -1);  baddbmm_84 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_169: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_84, -1)
        expand_168: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_169, [8, 128, 2]);  unsqueeze_169 = None
        gather_84: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_168);  expand_168 = None
        view_337: "f32[2048]" = torch.ops.aten.view.default(gather_84, [-1]);  gather_84 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_759: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_83, 1, 84)
        copy_84: "i64[8, 128]" = torch.ops.aten.copy.default(select_759, argmin_84);  select_759 = argmin_84 = None
        select_scatter_84: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_83, copy_84, 1, 84);  select_scatter_83 = copy_84 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_84: "f32[2048]" = torch.ops.aten.sub.Tensor(select_756, view_337);  select_756 = view_337 = None
        div_84: "f32[2048]" = torch.ops.aten.div.Tensor(sub_84, select_755);  sub_84 = select_755 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_761: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 84)
        slice_334: "f32[44]" = torch.ops.aten.slice.Tensor(select_761, 0, 84, 9223372036854775807);  select_761 = None
        slice_335: "f32[2048, 44]" = torch.ops.aten.slice.Tensor(slice_scatter_82, 1, 84, 9223372036854775807)
        expand_169: "f32[2048, 44]" = torch.ops.aten.expand.default(slice_335, [2048, 44]);  slice_335 = None
        mul_252: "f32[2048, 44]" = torch.ops.aten.mul.Tensor(expand_169, 1);  expand_169 = None
        view_338: "f32[2048, 1]" = torch.ops.aten.view.default(div_84, [2048, 1]);  div_84 = None
        mul_253: "f32[2048, 44]" = torch.ops.aten.mul.Tensor(view_338, slice_334);  view_338 = slice_334 = None
        mul_254: "f32[2048, 44]" = torch.ops.aten.mul.Tensor(mul_253, -1.0);  mul_253 = None
        add_84: "f32[2048, 44]" = torch.ops.aten.add.Tensor(mul_252, mul_254);  mul_252 = mul_254 = None
        slice_scatter_83: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_82, add_84, 1, 84, 9223372036854775807);  slice_scatter_82 = add_84 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_763: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 85)
        select_764: "f32[]" = torch.ops.aten.select.int(select_763, 0, 85);  select_763 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_765: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_83, 1, 85)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_170: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_85: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_766: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_83, 1, 85)
        view_340: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_766, [8, 128, 2]);  select_766 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_85: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_170, view_340, permute_85, alpha = -2.0);  unsqueeze_170 = view_340 = permute_85 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_85: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_85, -1);  baddbmm_85 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_171: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_85, -1)
        expand_170: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_171, [8, 128, 2]);  unsqueeze_171 = None
        gather_85: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_170);  expand_170 = None
        view_341: "f32[2048]" = torch.ops.aten.view.default(gather_85, [-1]);  gather_85 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_768: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_84, 1, 85)
        copy_85: "i64[8, 128]" = torch.ops.aten.copy.default(select_768, argmin_85);  select_768 = argmin_85 = None
        select_scatter_85: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_84, copy_85, 1, 85);  select_scatter_84 = copy_85 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_85: "f32[2048]" = torch.ops.aten.sub.Tensor(select_765, view_341);  select_765 = view_341 = None
        div_85: "f32[2048]" = torch.ops.aten.div.Tensor(sub_85, select_764);  sub_85 = select_764 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_770: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 85)
        slice_338: "f32[43]" = torch.ops.aten.slice.Tensor(select_770, 0, 85, 9223372036854775807);  select_770 = None
        slice_339: "f32[2048, 43]" = torch.ops.aten.slice.Tensor(slice_scatter_83, 1, 85, 9223372036854775807)
        expand_171: "f32[2048, 43]" = torch.ops.aten.expand.default(slice_339, [2048, 43]);  slice_339 = None
        mul_255: "f32[2048, 43]" = torch.ops.aten.mul.Tensor(expand_171, 1);  expand_171 = None
        view_342: "f32[2048, 1]" = torch.ops.aten.view.default(div_85, [2048, 1]);  div_85 = None
        mul_256: "f32[2048, 43]" = torch.ops.aten.mul.Tensor(view_342, slice_338);  view_342 = slice_338 = None
        mul_257: "f32[2048, 43]" = torch.ops.aten.mul.Tensor(mul_256, -1.0);  mul_256 = None
        add_85: "f32[2048, 43]" = torch.ops.aten.add.Tensor(mul_255, mul_257);  mul_255 = mul_257 = None
        slice_scatter_84: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_83, add_85, 1, 85, 9223372036854775807);  slice_scatter_83 = add_85 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_772: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 86)
        select_773: "f32[]" = torch.ops.aten.select.int(select_772, 0, 86);  select_772 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_774: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_84, 1, 86)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_172: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_86: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_775: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_84, 1, 86)
        view_344: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_775, [8, 128, 2]);  select_775 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_86: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_172, view_344, permute_86, alpha = -2.0);  unsqueeze_172 = view_344 = permute_86 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_86: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_86, -1);  baddbmm_86 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_173: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_86, -1)
        expand_172: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_173, [8, 128, 2]);  unsqueeze_173 = None
        gather_86: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_172);  expand_172 = None
        view_345: "f32[2048]" = torch.ops.aten.view.default(gather_86, [-1]);  gather_86 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_777: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_85, 1, 86)
        copy_86: "i64[8, 128]" = torch.ops.aten.copy.default(select_777, argmin_86);  select_777 = argmin_86 = None
        select_scatter_86: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_85, copy_86, 1, 86);  select_scatter_85 = copy_86 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_86: "f32[2048]" = torch.ops.aten.sub.Tensor(select_774, view_345);  select_774 = view_345 = None
        div_86: "f32[2048]" = torch.ops.aten.div.Tensor(sub_86, select_773);  sub_86 = select_773 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_779: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 86)
        slice_342: "f32[42]" = torch.ops.aten.slice.Tensor(select_779, 0, 86, 9223372036854775807);  select_779 = None
        slice_343: "f32[2048, 42]" = torch.ops.aten.slice.Tensor(slice_scatter_84, 1, 86, 9223372036854775807)
        expand_173: "f32[2048, 42]" = torch.ops.aten.expand.default(slice_343, [2048, 42]);  slice_343 = None
        mul_258: "f32[2048, 42]" = torch.ops.aten.mul.Tensor(expand_173, 1);  expand_173 = None
        view_346: "f32[2048, 1]" = torch.ops.aten.view.default(div_86, [2048, 1]);  div_86 = None
        mul_259: "f32[2048, 42]" = torch.ops.aten.mul.Tensor(view_346, slice_342);  view_346 = slice_342 = None
        mul_260: "f32[2048, 42]" = torch.ops.aten.mul.Tensor(mul_259, -1.0);  mul_259 = None
        add_86: "f32[2048, 42]" = torch.ops.aten.add.Tensor(mul_258, mul_260);  mul_258 = mul_260 = None
        slice_scatter_85: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_84, add_86, 1, 86, 9223372036854775807);  slice_scatter_84 = add_86 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_781: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 87)
        select_782: "f32[]" = torch.ops.aten.select.int(select_781, 0, 87);  select_781 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_783: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_85, 1, 87)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_174: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_87: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_784: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_85, 1, 87)
        view_348: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_784, [8, 128, 2]);  select_784 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_87: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_174, view_348, permute_87, alpha = -2.0);  unsqueeze_174 = view_348 = permute_87 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_87: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_87, -1);  baddbmm_87 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_175: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_87, -1)
        expand_174: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_175, [8, 128, 2]);  unsqueeze_175 = None
        gather_87: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_174);  expand_174 = None
        view_349: "f32[2048]" = torch.ops.aten.view.default(gather_87, [-1]);  gather_87 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_786: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_86, 1, 87)
        copy_87: "i64[8, 128]" = torch.ops.aten.copy.default(select_786, argmin_87);  select_786 = argmin_87 = None
        select_scatter_87: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_86, copy_87, 1, 87);  select_scatter_86 = copy_87 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_87: "f32[2048]" = torch.ops.aten.sub.Tensor(select_783, view_349);  select_783 = view_349 = None
        div_87: "f32[2048]" = torch.ops.aten.div.Tensor(sub_87, select_782);  sub_87 = select_782 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_788: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 87)
        slice_346: "f32[41]" = torch.ops.aten.slice.Tensor(select_788, 0, 87, 9223372036854775807);  select_788 = None
        slice_347: "f32[2048, 41]" = torch.ops.aten.slice.Tensor(slice_scatter_85, 1, 87, 9223372036854775807)
        expand_175: "f32[2048, 41]" = torch.ops.aten.expand.default(slice_347, [2048, 41]);  slice_347 = None
        mul_261: "f32[2048, 41]" = torch.ops.aten.mul.Tensor(expand_175, 1);  expand_175 = None
        view_350: "f32[2048, 1]" = torch.ops.aten.view.default(div_87, [2048, 1]);  div_87 = None
        mul_262: "f32[2048, 41]" = torch.ops.aten.mul.Tensor(view_350, slice_346);  view_350 = slice_346 = None
        mul_263: "f32[2048, 41]" = torch.ops.aten.mul.Tensor(mul_262, -1.0);  mul_262 = None
        add_87: "f32[2048, 41]" = torch.ops.aten.add.Tensor(mul_261, mul_263);  mul_261 = mul_263 = None
        slice_scatter_86: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_85, add_87, 1, 87, 9223372036854775807);  slice_scatter_85 = add_87 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_790: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 88)
        select_791: "f32[]" = torch.ops.aten.select.int(select_790, 0, 88);  select_790 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_792: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_86, 1, 88)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_176: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_88: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_793: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_86, 1, 88)
        view_352: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_793, [8, 128, 2]);  select_793 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_88: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_176, view_352, permute_88, alpha = -2.0);  unsqueeze_176 = view_352 = permute_88 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_88: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_88, -1);  baddbmm_88 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_177: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_88, -1)
        expand_176: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_177, [8, 128, 2]);  unsqueeze_177 = None
        gather_88: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_176);  expand_176 = None
        view_353: "f32[2048]" = torch.ops.aten.view.default(gather_88, [-1]);  gather_88 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_795: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_87, 1, 88)
        copy_88: "i64[8, 128]" = torch.ops.aten.copy.default(select_795, argmin_88);  select_795 = argmin_88 = None
        select_scatter_88: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_87, copy_88, 1, 88);  select_scatter_87 = copy_88 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_88: "f32[2048]" = torch.ops.aten.sub.Tensor(select_792, view_353);  select_792 = view_353 = None
        div_88: "f32[2048]" = torch.ops.aten.div.Tensor(sub_88, select_791);  sub_88 = select_791 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_797: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 88)
        slice_350: "f32[40]" = torch.ops.aten.slice.Tensor(select_797, 0, 88, 9223372036854775807);  select_797 = None
        slice_351: "f32[2048, 40]" = torch.ops.aten.slice.Tensor(slice_scatter_86, 1, 88, 9223372036854775807)
        expand_177: "f32[2048, 40]" = torch.ops.aten.expand.default(slice_351, [2048, 40]);  slice_351 = None
        mul_264: "f32[2048, 40]" = torch.ops.aten.mul.Tensor(expand_177, 1);  expand_177 = None
        view_354: "f32[2048, 1]" = torch.ops.aten.view.default(div_88, [2048, 1]);  div_88 = None
        mul_265: "f32[2048, 40]" = torch.ops.aten.mul.Tensor(view_354, slice_350);  view_354 = slice_350 = None
        mul_266: "f32[2048, 40]" = torch.ops.aten.mul.Tensor(mul_265, -1.0);  mul_265 = None
        add_88: "f32[2048, 40]" = torch.ops.aten.add.Tensor(mul_264, mul_266);  mul_264 = mul_266 = None
        slice_scatter_87: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_86, add_88, 1, 88, 9223372036854775807);  slice_scatter_86 = add_88 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_799: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 89)
        select_800: "f32[]" = torch.ops.aten.select.int(select_799, 0, 89);  select_799 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_801: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_87, 1, 89)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_178: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_89: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_802: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_87, 1, 89)
        view_356: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_802, [8, 128, 2]);  select_802 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_89: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_178, view_356, permute_89, alpha = -2.0);  unsqueeze_178 = view_356 = permute_89 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_89: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_89, -1);  baddbmm_89 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_179: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_89, -1)
        expand_178: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_179, [8, 128, 2]);  unsqueeze_179 = None
        gather_89: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_178);  expand_178 = None
        view_357: "f32[2048]" = torch.ops.aten.view.default(gather_89, [-1]);  gather_89 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_804: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_88, 1, 89)
        copy_89: "i64[8, 128]" = torch.ops.aten.copy.default(select_804, argmin_89);  select_804 = argmin_89 = None
        select_scatter_89: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_88, copy_89, 1, 89);  select_scatter_88 = copy_89 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_89: "f32[2048]" = torch.ops.aten.sub.Tensor(select_801, view_357);  select_801 = view_357 = None
        div_89: "f32[2048]" = torch.ops.aten.div.Tensor(sub_89, select_800);  sub_89 = select_800 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_806: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 89)
        slice_354: "f32[39]" = torch.ops.aten.slice.Tensor(select_806, 0, 89, 9223372036854775807);  select_806 = None
        slice_355: "f32[2048, 39]" = torch.ops.aten.slice.Tensor(slice_scatter_87, 1, 89, 9223372036854775807)
        expand_179: "f32[2048, 39]" = torch.ops.aten.expand.default(slice_355, [2048, 39]);  slice_355 = None
        mul_267: "f32[2048, 39]" = torch.ops.aten.mul.Tensor(expand_179, 1);  expand_179 = None
        view_358: "f32[2048, 1]" = torch.ops.aten.view.default(div_89, [2048, 1]);  div_89 = None
        mul_268: "f32[2048, 39]" = torch.ops.aten.mul.Tensor(view_358, slice_354);  view_358 = slice_354 = None
        mul_269: "f32[2048, 39]" = torch.ops.aten.mul.Tensor(mul_268, -1.0);  mul_268 = None
        add_89: "f32[2048, 39]" = torch.ops.aten.add.Tensor(mul_267, mul_269);  mul_267 = mul_269 = None
        slice_scatter_88: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_87, add_89, 1, 89, 9223372036854775807);  slice_scatter_87 = add_89 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_808: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 90)
        select_809: "f32[]" = torch.ops.aten.select.int(select_808, 0, 90);  select_808 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_810: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_88, 1, 90)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_180: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_90: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_811: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_88, 1, 90)
        view_360: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_811, [8, 128, 2]);  select_811 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_90: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_180, view_360, permute_90, alpha = -2.0);  unsqueeze_180 = view_360 = permute_90 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_90: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_90, -1);  baddbmm_90 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_181: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_90, -1)
        expand_180: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_181, [8, 128, 2]);  unsqueeze_181 = None
        gather_90: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_180);  expand_180 = None
        view_361: "f32[2048]" = torch.ops.aten.view.default(gather_90, [-1]);  gather_90 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_813: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_89, 1, 90)
        copy_90: "i64[8, 128]" = torch.ops.aten.copy.default(select_813, argmin_90);  select_813 = argmin_90 = None
        select_scatter_90: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_89, copy_90, 1, 90);  select_scatter_89 = copy_90 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_90: "f32[2048]" = torch.ops.aten.sub.Tensor(select_810, view_361);  select_810 = view_361 = None
        div_90: "f32[2048]" = torch.ops.aten.div.Tensor(sub_90, select_809);  sub_90 = select_809 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_815: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 90)
        slice_358: "f32[38]" = torch.ops.aten.slice.Tensor(select_815, 0, 90, 9223372036854775807);  select_815 = None
        slice_359: "f32[2048, 38]" = torch.ops.aten.slice.Tensor(slice_scatter_88, 1, 90, 9223372036854775807)
        expand_181: "f32[2048, 38]" = torch.ops.aten.expand.default(slice_359, [2048, 38]);  slice_359 = None
        mul_270: "f32[2048, 38]" = torch.ops.aten.mul.Tensor(expand_181, 1);  expand_181 = None
        view_362: "f32[2048, 1]" = torch.ops.aten.view.default(div_90, [2048, 1]);  div_90 = None
        mul_271: "f32[2048, 38]" = torch.ops.aten.mul.Tensor(view_362, slice_358);  view_362 = slice_358 = None
        mul_272: "f32[2048, 38]" = torch.ops.aten.mul.Tensor(mul_271, -1.0);  mul_271 = None
        add_90: "f32[2048, 38]" = torch.ops.aten.add.Tensor(mul_270, mul_272);  mul_270 = mul_272 = None
        slice_scatter_89: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_88, add_90, 1, 90, 9223372036854775807);  slice_scatter_88 = add_90 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_817: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 91)
        select_818: "f32[]" = torch.ops.aten.select.int(select_817, 0, 91);  select_817 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_819: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_89, 1, 91)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_182: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_91: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_820: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_89, 1, 91)
        view_364: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_820, [8, 128, 2]);  select_820 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_91: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_182, view_364, permute_91, alpha = -2.0);  unsqueeze_182 = view_364 = permute_91 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_91: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_91, -1);  baddbmm_91 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_183: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_91, -1)
        expand_182: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_183, [8, 128, 2]);  unsqueeze_183 = None
        gather_91: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_182);  expand_182 = None
        view_365: "f32[2048]" = torch.ops.aten.view.default(gather_91, [-1]);  gather_91 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_822: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_90, 1, 91)
        copy_91: "i64[8, 128]" = torch.ops.aten.copy.default(select_822, argmin_91);  select_822 = argmin_91 = None
        select_scatter_91: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_90, copy_91, 1, 91);  select_scatter_90 = copy_91 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_91: "f32[2048]" = torch.ops.aten.sub.Tensor(select_819, view_365);  select_819 = view_365 = None
        div_91: "f32[2048]" = torch.ops.aten.div.Tensor(sub_91, select_818);  sub_91 = select_818 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_824: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 91)
        slice_362: "f32[37]" = torch.ops.aten.slice.Tensor(select_824, 0, 91, 9223372036854775807);  select_824 = None
        slice_363: "f32[2048, 37]" = torch.ops.aten.slice.Tensor(slice_scatter_89, 1, 91, 9223372036854775807)
        expand_183: "f32[2048, 37]" = torch.ops.aten.expand.default(slice_363, [2048, 37]);  slice_363 = None
        mul_273: "f32[2048, 37]" = torch.ops.aten.mul.Tensor(expand_183, 1);  expand_183 = None
        view_366: "f32[2048, 1]" = torch.ops.aten.view.default(div_91, [2048, 1]);  div_91 = None
        mul_274: "f32[2048, 37]" = torch.ops.aten.mul.Tensor(view_366, slice_362);  view_366 = slice_362 = None
        mul_275: "f32[2048, 37]" = torch.ops.aten.mul.Tensor(mul_274, -1.0);  mul_274 = None
        add_91: "f32[2048, 37]" = torch.ops.aten.add.Tensor(mul_273, mul_275);  mul_273 = mul_275 = None
        slice_scatter_90: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_89, add_91, 1, 91, 9223372036854775807);  slice_scatter_89 = add_91 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_826: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 92)
        select_827: "f32[]" = torch.ops.aten.select.int(select_826, 0, 92);  select_826 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_828: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_90, 1, 92)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_184: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_92: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_829: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_90, 1, 92)
        view_368: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_829, [8, 128, 2]);  select_829 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_92: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_184, view_368, permute_92, alpha = -2.0);  unsqueeze_184 = view_368 = permute_92 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_92: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_92, -1);  baddbmm_92 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_185: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_92, -1)
        expand_184: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_185, [8, 128, 2]);  unsqueeze_185 = None
        gather_92: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_184);  expand_184 = None
        view_369: "f32[2048]" = torch.ops.aten.view.default(gather_92, [-1]);  gather_92 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_831: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_91, 1, 92)
        copy_92: "i64[8, 128]" = torch.ops.aten.copy.default(select_831, argmin_92);  select_831 = argmin_92 = None
        select_scatter_92: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_91, copy_92, 1, 92);  select_scatter_91 = copy_92 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_92: "f32[2048]" = torch.ops.aten.sub.Tensor(select_828, view_369);  select_828 = view_369 = None
        div_92: "f32[2048]" = torch.ops.aten.div.Tensor(sub_92, select_827);  sub_92 = select_827 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_833: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 92)
        slice_366: "f32[36]" = torch.ops.aten.slice.Tensor(select_833, 0, 92, 9223372036854775807);  select_833 = None
        slice_367: "f32[2048, 36]" = torch.ops.aten.slice.Tensor(slice_scatter_90, 1, 92, 9223372036854775807)
        expand_185: "f32[2048, 36]" = torch.ops.aten.expand.default(slice_367, [2048, 36]);  slice_367 = None
        mul_276: "f32[2048, 36]" = torch.ops.aten.mul.Tensor(expand_185, 1);  expand_185 = None
        view_370: "f32[2048, 1]" = torch.ops.aten.view.default(div_92, [2048, 1]);  div_92 = None
        mul_277: "f32[2048, 36]" = torch.ops.aten.mul.Tensor(view_370, slice_366);  view_370 = slice_366 = None
        mul_278: "f32[2048, 36]" = torch.ops.aten.mul.Tensor(mul_277, -1.0);  mul_277 = None
        add_92: "f32[2048, 36]" = torch.ops.aten.add.Tensor(mul_276, mul_278);  mul_276 = mul_278 = None
        slice_scatter_91: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_90, add_92, 1, 92, 9223372036854775807);  slice_scatter_90 = add_92 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_835: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 93)
        select_836: "f32[]" = torch.ops.aten.select.int(select_835, 0, 93);  select_835 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_837: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_91, 1, 93)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_186: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_93: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_838: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_91, 1, 93)
        view_372: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_838, [8, 128, 2]);  select_838 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_93: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_186, view_372, permute_93, alpha = -2.0);  unsqueeze_186 = view_372 = permute_93 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_93: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_93, -1);  baddbmm_93 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_187: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_93, -1)
        expand_186: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_187, [8, 128, 2]);  unsqueeze_187 = None
        gather_93: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_186);  expand_186 = None
        view_373: "f32[2048]" = torch.ops.aten.view.default(gather_93, [-1]);  gather_93 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_840: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_92, 1, 93)
        copy_93: "i64[8, 128]" = torch.ops.aten.copy.default(select_840, argmin_93);  select_840 = argmin_93 = None
        select_scatter_93: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_92, copy_93, 1, 93);  select_scatter_92 = copy_93 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_93: "f32[2048]" = torch.ops.aten.sub.Tensor(select_837, view_373);  select_837 = view_373 = None
        div_93: "f32[2048]" = torch.ops.aten.div.Tensor(sub_93, select_836);  sub_93 = select_836 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_842: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 93)
        slice_370: "f32[35]" = torch.ops.aten.slice.Tensor(select_842, 0, 93, 9223372036854775807);  select_842 = None
        slice_371: "f32[2048, 35]" = torch.ops.aten.slice.Tensor(slice_scatter_91, 1, 93, 9223372036854775807)
        expand_187: "f32[2048, 35]" = torch.ops.aten.expand.default(slice_371, [2048, 35]);  slice_371 = None
        mul_279: "f32[2048, 35]" = torch.ops.aten.mul.Tensor(expand_187, 1);  expand_187 = None
        view_374: "f32[2048, 1]" = torch.ops.aten.view.default(div_93, [2048, 1]);  div_93 = None
        mul_280: "f32[2048, 35]" = torch.ops.aten.mul.Tensor(view_374, slice_370);  view_374 = slice_370 = None
        mul_281: "f32[2048, 35]" = torch.ops.aten.mul.Tensor(mul_280, -1.0);  mul_280 = None
        add_93: "f32[2048, 35]" = torch.ops.aten.add.Tensor(mul_279, mul_281);  mul_279 = mul_281 = None
        slice_scatter_92: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_91, add_93, 1, 93, 9223372036854775807);  slice_scatter_91 = add_93 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_844: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 94)
        select_845: "f32[]" = torch.ops.aten.select.int(select_844, 0, 94);  select_844 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_846: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_92, 1, 94)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_188: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_94: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_847: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_92, 1, 94)
        view_376: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_847, [8, 128, 2]);  select_847 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_94: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_188, view_376, permute_94, alpha = -2.0);  unsqueeze_188 = view_376 = permute_94 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_94: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_94, -1);  baddbmm_94 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_189: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_94, -1)
        expand_188: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_189, [8, 128, 2]);  unsqueeze_189 = None
        gather_94: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_188);  expand_188 = None
        view_377: "f32[2048]" = torch.ops.aten.view.default(gather_94, [-1]);  gather_94 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_849: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_93, 1, 94)
        copy_94: "i64[8, 128]" = torch.ops.aten.copy.default(select_849, argmin_94);  select_849 = argmin_94 = None
        select_scatter_94: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_93, copy_94, 1, 94);  select_scatter_93 = copy_94 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_94: "f32[2048]" = torch.ops.aten.sub.Tensor(select_846, view_377);  select_846 = view_377 = None
        div_94: "f32[2048]" = torch.ops.aten.div.Tensor(sub_94, select_845);  sub_94 = select_845 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_851: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 94)
        slice_374: "f32[34]" = torch.ops.aten.slice.Tensor(select_851, 0, 94, 9223372036854775807);  select_851 = None
        slice_375: "f32[2048, 34]" = torch.ops.aten.slice.Tensor(slice_scatter_92, 1, 94, 9223372036854775807)
        expand_189: "f32[2048, 34]" = torch.ops.aten.expand.default(slice_375, [2048, 34]);  slice_375 = None
        mul_282: "f32[2048, 34]" = torch.ops.aten.mul.Tensor(expand_189, 1);  expand_189 = None
        view_378: "f32[2048, 1]" = torch.ops.aten.view.default(div_94, [2048, 1]);  div_94 = None
        mul_283: "f32[2048, 34]" = torch.ops.aten.mul.Tensor(view_378, slice_374);  view_378 = slice_374 = None
        mul_284: "f32[2048, 34]" = torch.ops.aten.mul.Tensor(mul_283, -1.0);  mul_283 = None
        add_94: "f32[2048, 34]" = torch.ops.aten.add.Tensor(mul_282, mul_284);  mul_282 = mul_284 = None
        slice_scatter_93: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_92, add_94, 1, 94, 9223372036854775807);  slice_scatter_92 = add_94 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_853: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 95)
        select_854: "f32[]" = torch.ops.aten.select.int(select_853, 0, 95);  select_853 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_855: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_93, 1, 95)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_190: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_95: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_856: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_93, 1, 95)
        view_380: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_856, [8, 128, 2]);  select_856 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_95: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_190, view_380, permute_95, alpha = -2.0);  unsqueeze_190 = view_380 = permute_95 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_95: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_95, -1);  baddbmm_95 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_191: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_95, -1)
        expand_190: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_191, [8, 128, 2]);  unsqueeze_191 = None
        gather_95: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_190);  expand_190 = None
        view_381: "f32[2048]" = torch.ops.aten.view.default(gather_95, [-1]);  gather_95 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_858: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_94, 1, 95)
        copy_95: "i64[8, 128]" = torch.ops.aten.copy.default(select_858, argmin_95);  select_858 = argmin_95 = None
        select_scatter_95: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_94, copy_95, 1, 95);  select_scatter_94 = copy_95 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_95: "f32[2048]" = torch.ops.aten.sub.Tensor(select_855, view_381);  select_855 = view_381 = None
        div_95: "f32[2048]" = torch.ops.aten.div.Tensor(sub_95, select_854);  sub_95 = select_854 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_860: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 95)
        slice_378: "f32[33]" = torch.ops.aten.slice.Tensor(select_860, 0, 95, 9223372036854775807);  select_860 = None
        slice_379: "f32[2048, 33]" = torch.ops.aten.slice.Tensor(slice_scatter_93, 1, 95, 9223372036854775807)
        expand_191: "f32[2048, 33]" = torch.ops.aten.expand.default(slice_379, [2048, 33]);  slice_379 = None
        mul_285: "f32[2048, 33]" = torch.ops.aten.mul.Tensor(expand_191, 1);  expand_191 = None
        view_382: "f32[2048, 1]" = torch.ops.aten.view.default(div_95, [2048, 1]);  div_95 = None
        mul_286: "f32[2048, 33]" = torch.ops.aten.mul.Tensor(view_382, slice_378);  view_382 = slice_378 = None
        mul_287: "f32[2048, 33]" = torch.ops.aten.mul.Tensor(mul_286, -1.0);  mul_286 = None
        add_95: "f32[2048, 33]" = torch.ops.aten.add.Tensor(mul_285, mul_287);  mul_285 = mul_287 = None
        slice_scatter_94: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_93, add_95, 1, 95, 9223372036854775807);  slice_scatter_93 = add_95 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_862: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 96)
        select_863: "f32[]" = torch.ops.aten.select.int(select_862, 0, 96);  select_862 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_864: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_94, 1, 96)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_192: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_96: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_865: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_94, 1, 96)
        view_384: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_865, [8, 128, 2]);  select_865 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_96: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_192, view_384, permute_96, alpha = -2.0);  unsqueeze_192 = view_384 = permute_96 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_96: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_96, -1);  baddbmm_96 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_193: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_96, -1)
        expand_192: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_193, [8, 128, 2]);  unsqueeze_193 = None
        gather_96: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_192);  expand_192 = None
        view_385: "f32[2048]" = torch.ops.aten.view.default(gather_96, [-1]);  gather_96 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_867: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_95, 1, 96)
        copy_96: "i64[8, 128]" = torch.ops.aten.copy.default(select_867, argmin_96);  select_867 = argmin_96 = None
        select_scatter_96: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_95, copy_96, 1, 96);  select_scatter_95 = copy_96 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_96: "f32[2048]" = torch.ops.aten.sub.Tensor(select_864, view_385);  select_864 = view_385 = None
        div_96: "f32[2048]" = torch.ops.aten.div.Tensor(sub_96, select_863);  sub_96 = select_863 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_869: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 96)
        slice_382: "f32[32]" = torch.ops.aten.slice.Tensor(select_869, 0, 96, 9223372036854775807);  select_869 = None
        slice_383: "f32[2048, 32]" = torch.ops.aten.slice.Tensor(slice_scatter_94, 1, 96, 9223372036854775807)
        expand_193: "f32[2048, 32]" = torch.ops.aten.expand.default(slice_383, [2048, 32]);  slice_383 = None
        mul_288: "f32[2048, 32]" = torch.ops.aten.mul.Tensor(expand_193, 1);  expand_193 = None
        view_386: "f32[2048, 1]" = torch.ops.aten.view.default(div_96, [2048, 1]);  div_96 = None
        mul_289: "f32[2048, 32]" = torch.ops.aten.mul.Tensor(view_386, slice_382);  view_386 = slice_382 = None
        mul_290: "f32[2048, 32]" = torch.ops.aten.mul.Tensor(mul_289, -1.0);  mul_289 = None
        add_96: "f32[2048, 32]" = torch.ops.aten.add.Tensor(mul_288, mul_290);  mul_288 = mul_290 = None
        slice_scatter_95: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_94, add_96, 1, 96, 9223372036854775807);  slice_scatter_94 = add_96 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_871: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 97)
        select_872: "f32[]" = torch.ops.aten.select.int(select_871, 0, 97);  select_871 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_873: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_95, 1, 97)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_194: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_97: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_874: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_95, 1, 97)
        view_388: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_874, [8, 128, 2]);  select_874 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_97: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_194, view_388, permute_97, alpha = -2.0);  unsqueeze_194 = view_388 = permute_97 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_97: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_97, -1);  baddbmm_97 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_195: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_97, -1)
        expand_194: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_195, [8, 128, 2]);  unsqueeze_195 = None
        gather_97: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_194);  expand_194 = None
        view_389: "f32[2048]" = torch.ops.aten.view.default(gather_97, [-1]);  gather_97 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_876: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_96, 1, 97)
        copy_97: "i64[8, 128]" = torch.ops.aten.copy.default(select_876, argmin_97);  select_876 = argmin_97 = None
        select_scatter_97: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_96, copy_97, 1, 97);  select_scatter_96 = copy_97 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_97: "f32[2048]" = torch.ops.aten.sub.Tensor(select_873, view_389);  select_873 = view_389 = None
        div_97: "f32[2048]" = torch.ops.aten.div.Tensor(sub_97, select_872);  sub_97 = select_872 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_878: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 97)
        slice_386: "f32[31]" = torch.ops.aten.slice.Tensor(select_878, 0, 97, 9223372036854775807);  select_878 = None
        slice_387: "f32[2048, 31]" = torch.ops.aten.slice.Tensor(slice_scatter_95, 1, 97, 9223372036854775807)
        expand_195: "f32[2048, 31]" = torch.ops.aten.expand.default(slice_387, [2048, 31]);  slice_387 = None
        mul_291: "f32[2048, 31]" = torch.ops.aten.mul.Tensor(expand_195, 1);  expand_195 = None
        view_390: "f32[2048, 1]" = torch.ops.aten.view.default(div_97, [2048, 1]);  div_97 = None
        mul_292: "f32[2048, 31]" = torch.ops.aten.mul.Tensor(view_390, slice_386);  view_390 = slice_386 = None
        mul_293: "f32[2048, 31]" = torch.ops.aten.mul.Tensor(mul_292, -1.0);  mul_292 = None
        add_97: "f32[2048, 31]" = torch.ops.aten.add.Tensor(mul_291, mul_293);  mul_291 = mul_293 = None
        slice_scatter_96: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_95, add_97, 1, 97, 9223372036854775807);  slice_scatter_95 = add_97 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_880: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 98)
        select_881: "f32[]" = torch.ops.aten.select.int(select_880, 0, 98);  select_880 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_882: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_96, 1, 98)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_196: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_98: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_883: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_96, 1, 98)
        view_392: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_883, [8, 128, 2]);  select_883 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_98: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_196, view_392, permute_98, alpha = -2.0);  unsqueeze_196 = view_392 = permute_98 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_98: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_98, -1);  baddbmm_98 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_197: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_98, -1)
        expand_196: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_197, [8, 128, 2]);  unsqueeze_197 = None
        gather_98: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_196);  expand_196 = None
        view_393: "f32[2048]" = torch.ops.aten.view.default(gather_98, [-1]);  gather_98 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_885: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_97, 1, 98)
        copy_98: "i64[8, 128]" = torch.ops.aten.copy.default(select_885, argmin_98);  select_885 = argmin_98 = None
        select_scatter_98: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_97, copy_98, 1, 98);  select_scatter_97 = copy_98 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_98: "f32[2048]" = torch.ops.aten.sub.Tensor(select_882, view_393);  select_882 = view_393 = None
        div_98: "f32[2048]" = torch.ops.aten.div.Tensor(sub_98, select_881);  sub_98 = select_881 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_887: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 98)
        slice_390: "f32[30]" = torch.ops.aten.slice.Tensor(select_887, 0, 98, 9223372036854775807);  select_887 = None
        slice_391: "f32[2048, 30]" = torch.ops.aten.slice.Tensor(slice_scatter_96, 1, 98, 9223372036854775807)
        expand_197: "f32[2048, 30]" = torch.ops.aten.expand.default(slice_391, [2048, 30]);  slice_391 = None
        mul_294: "f32[2048, 30]" = torch.ops.aten.mul.Tensor(expand_197, 1);  expand_197 = None
        view_394: "f32[2048, 1]" = torch.ops.aten.view.default(div_98, [2048, 1]);  div_98 = None
        mul_295: "f32[2048, 30]" = torch.ops.aten.mul.Tensor(view_394, slice_390);  view_394 = slice_390 = None
        mul_296: "f32[2048, 30]" = torch.ops.aten.mul.Tensor(mul_295, -1.0);  mul_295 = None
        add_98: "f32[2048, 30]" = torch.ops.aten.add.Tensor(mul_294, mul_296);  mul_294 = mul_296 = None
        slice_scatter_97: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_96, add_98, 1, 98, 9223372036854775807);  slice_scatter_96 = add_98 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_889: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 99)
        select_890: "f32[]" = torch.ops.aten.select.int(select_889, 0, 99);  select_889 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_891: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_97, 1, 99)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_198: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_99: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_892: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_97, 1, 99)
        view_396: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_892, [8, 128, 2]);  select_892 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_99: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_198, view_396, permute_99, alpha = -2.0);  unsqueeze_198 = view_396 = permute_99 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_99: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_99, -1);  baddbmm_99 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_199: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_99, -1)
        expand_198: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_199, [8, 128, 2]);  unsqueeze_199 = None
        gather_99: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_198);  expand_198 = None
        view_397: "f32[2048]" = torch.ops.aten.view.default(gather_99, [-1]);  gather_99 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_894: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_98, 1, 99)
        copy_99: "i64[8, 128]" = torch.ops.aten.copy.default(select_894, argmin_99);  select_894 = argmin_99 = None
        select_scatter_99: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_98, copy_99, 1, 99);  select_scatter_98 = copy_99 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_99: "f32[2048]" = torch.ops.aten.sub.Tensor(select_891, view_397);  select_891 = view_397 = None
        div_99: "f32[2048]" = torch.ops.aten.div.Tensor(sub_99, select_890);  sub_99 = select_890 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_896: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 99)
        slice_394: "f32[29]" = torch.ops.aten.slice.Tensor(select_896, 0, 99, 9223372036854775807);  select_896 = None
        slice_395: "f32[2048, 29]" = torch.ops.aten.slice.Tensor(slice_scatter_97, 1, 99, 9223372036854775807)
        expand_199: "f32[2048, 29]" = torch.ops.aten.expand.default(slice_395, [2048, 29]);  slice_395 = None
        mul_297: "f32[2048, 29]" = torch.ops.aten.mul.Tensor(expand_199, 1);  expand_199 = None
        view_398: "f32[2048, 1]" = torch.ops.aten.view.default(div_99, [2048, 1]);  div_99 = None
        mul_298: "f32[2048, 29]" = torch.ops.aten.mul.Tensor(view_398, slice_394);  view_398 = slice_394 = None
        mul_299: "f32[2048, 29]" = torch.ops.aten.mul.Tensor(mul_298, -1.0);  mul_298 = None
        add_99: "f32[2048, 29]" = torch.ops.aten.add.Tensor(mul_297, mul_299);  mul_297 = mul_299 = None
        slice_scatter_98: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_97, add_99, 1, 99, 9223372036854775807);  slice_scatter_97 = add_99 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_898: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 100)
        select_899: "f32[]" = torch.ops.aten.select.int(select_898, 0, 100);  select_898 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_900: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_98, 1, 100)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_200: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_100: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_901: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_98, 1, 100)
        view_400: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_901, [8, 128, 2]);  select_901 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_100: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_200, view_400, permute_100, alpha = -2.0);  unsqueeze_200 = view_400 = permute_100 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_100: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_100, -1);  baddbmm_100 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_201: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_100, -1)
        expand_200: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_201, [8, 128, 2]);  unsqueeze_201 = None
        gather_100: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_200);  expand_200 = None
        view_401: "f32[2048]" = torch.ops.aten.view.default(gather_100, [-1]);  gather_100 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_903: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_99, 1, 100)
        copy_100: "i64[8, 128]" = torch.ops.aten.copy.default(select_903, argmin_100);  select_903 = argmin_100 = None
        select_scatter_100: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_99, copy_100, 1, 100);  select_scatter_99 = copy_100 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_100: "f32[2048]" = torch.ops.aten.sub.Tensor(select_900, view_401);  select_900 = view_401 = None
        div_100: "f32[2048]" = torch.ops.aten.div.Tensor(sub_100, select_899);  sub_100 = select_899 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_905: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 100)
        slice_398: "f32[28]" = torch.ops.aten.slice.Tensor(select_905, 0, 100, 9223372036854775807);  select_905 = None
        slice_399: "f32[2048, 28]" = torch.ops.aten.slice.Tensor(slice_scatter_98, 1, 100, 9223372036854775807)
        expand_201: "f32[2048, 28]" = torch.ops.aten.expand.default(slice_399, [2048, 28]);  slice_399 = None
        mul_300: "f32[2048, 28]" = torch.ops.aten.mul.Tensor(expand_201, 1);  expand_201 = None
        view_402: "f32[2048, 1]" = torch.ops.aten.view.default(div_100, [2048, 1]);  div_100 = None
        mul_301: "f32[2048, 28]" = torch.ops.aten.mul.Tensor(view_402, slice_398);  view_402 = slice_398 = None
        mul_302: "f32[2048, 28]" = torch.ops.aten.mul.Tensor(mul_301, -1.0);  mul_301 = None
        add_100: "f32[2048, 28]" = torch.ops.aten.add.Tensor(mul_300, mul_302);  mul_300 = mul_302 = None
        slice_scatter_99: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_98, add_100, 1, 100, 9223372036854775807);  slice_scatter_98 = add_100 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_907: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 101)
        select_908: "f32[]" = torch.ops.aten.select.int(select_907, 0, 101);  select_907 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_909: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_99, 1, 101)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_202: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_101: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_910: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_99, 1, 101)
        view_404: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_910, [8, 128, 2]);  select_910 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_101: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_202, view_404, permute_101, alpha = -2.0);  unsqueeze_202 = view_404 = permute_101 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_101: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_101, -1);  baddbmm_101 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_203: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_101, -1)
        expand_202: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_203, [8, 128, 2]);  unsqueeze_203 = None
        gather_101: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_202);  expand_202 = None
        view_405: "f32[2048]" = torch.ops.aten.view.default(gather_101, [-1]);  gather_101 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_912: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_100, 1, 101)
        copy_101: "i64[8, 128]" = torch.ops.aten.copy.default(select_912, argmin_101);  select_912 = argmin_101 = None
        select_scatter_101: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_100, copy_101, 1, 101);  select_scatter_100 = copy_101 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_101: "f32[2048]" = torch.ops.aten.sub.Tensor(select_909, view_405);  select_909 = view_405 = None
        div_101: "f32[2048]" = torch.ops.aten.div.Tensor(sub_101, select_908);  sub_101 = select_908 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_914: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 101)
        slice_402: "f32[27]" = torch.ops.aten.slice.Tensor(select_914, 0, 101, 9223372036854775807);  select_914 = None
        slice_403: "f32[2048, 27]" = torch.ops.aten.slice.Tensor(slice_scatter_99, 1, 101, 9223372036854775807)
        expand_203: "f32[2048, 27]" = torch.ops.aten.expand.default(slice_403, [2048, 27]);  slice_403 = None
        mul_303: "f32[2048, 27]" = torch.ops.aten.mul.Tensor(expand_203, 1);  expand_203 = None
        view_406: "f32[2048, 1]" = torch.ops.aten.view.default(div_101, [2048, 1]);  div_101 = None
        mul_304: "f32[2048, 27]" = torch.ops.aten.mul.Tensor(view_406, slice_402);  view_406 = slice_402 = None
        mul_305: "f32[2048, 27]" = torch.ops.aten.mul.Tensor(mul_304, -1.0);  mul_304 = None
        add_101: "f32[2048, 27]" = torch.ops.aten.add.Tensor(mul_303, mul_305);  mul_303 = mul_305 = None
        slice_scatter_100: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_99, add_101, 1, 101, 9223372036854775807);  slice_scatter_99 = add_101 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_916: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 102)
        select_917: "f32[]" = torch.ops.aten.select.int(select_916, 0, 102);  select_916 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_918: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_100, 1, 102)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_204: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_102: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_919: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_100, 1, 102)
        view_408: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_919, [8, 128, 2]);  select_919 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_102: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_204, view_408, permute_102, alpha = -2.0);  unsqueeze_204 = view_408 = permute_102 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_102: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_102, -1);  baddbmm_102 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_205: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_102, -1)
        expand_204: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_205, [8, 128, 2]);  unsqueeze_205 = None
        gather_102: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_204);  expand_204 = None
        view_409: "f32[2048]" = torch.ops.aten.view.default(gather_102, [-1]);  gather_102 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_921: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_101, 1, 102)
        copy_102: "i64[8, 128]" = torch.ops.aten.copy.default(select_921, argmin_102);  select_921 = argmin_102 = None
        select_scatter_102: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_101, copy_102, 1, 102);  select_scatter_101 = copy_102 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_102: "f32[2048]" = torch.ops.aten.sub.Tensor(select_918, view_409);  select_918 = view_409 = None
        div_102: "f32[2048]" = torch.ops.aten.div.Tensor(sub_102, select_917);  sub_102 = select_917 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_923: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 102)
        slice_406: "f32[26]" = torch.ops.aten.slice.Tensor(select_923, 0, 102, 9223372036854775807);  select_923 = None
        slice_407: "f32[2048, 26]" = torch.ops.aten.slice.Tensor(slice_scatter_100, 1, 102, 9223372036854775807)
        expand_205: "f32[2048, 26]" = torch.ops.aten.expand.default(slice_407, [2048, 26]);  slice_407 = None
        mul_306: "f32[2048, 26]" = torch.ops.aten.mul.Tensor(expand_205, 1);  expand_205 = None
        view_410: "f32[2048, 1]" = torch.ops.aten.view.default(div_102, [2048, 1]);  div_102 = None
        mul_307: "f32[2048, 26]" = torch.ops.aten.mul.Tensor(view_410, slice_406);  view_410 = slice_406 = None
        mul_308: "f32[2048, 26]" = torch.ops.aten.mul.Tensor(mul_307, -1.0);  mul_307 = None
        add_102: "f32[2048, 26]" = torch.ops.aten.add.Tensor(mul_306, mul_308);  mul_306 = mul_308 = None
        slice_scatter_101: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_100, add_102, 1, 102, 9223372036854775807);  slice_scatter_100 = add_102 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_925: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 103)
        select_926: "f32[]" = torch.ops.aten.select.int(select_925, 0, 103);  select_925 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_927: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_101, 1, 103)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_206: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_103: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_928: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_101, 1, 103)
        view_412: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_928, [8, 128, 2]);  select_928 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_103: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_206, view_412, permute_103, alpha = -2.0);  unsqueeze_206 = view_412 = permute_103 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_103: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_103, -1);  baddbmm_103 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_207: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_103, -1)
        expand_206: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_207, [8, 128, 2]);  unsqueeze_207 = None
        gather_103: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_206);  expand_206 = None
        view_413: "f32[2048]" = torch.ops.aten.view.default(gather_103, [-1]);  gather_103 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_930: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_102, 1, 103)
        copy_103: "i64[8, 128]" = torch.ops.aten.copy.default(select_930, argmin_103);  select_930 = argmin_103 = None
        select_scatter_103: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_102, copy_103, 1, 103);  select_scatter_102 = copy_103 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_103: "f32[2048]" = torch.ops.aten.sub.Tensor(select_927, view_413);  select_927 = view_413 = None
        div_103: "f32[2048]" = torch.ops.aten.div.Tensor(sub_103, select_926);  sub_103 = select_926 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_932: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 103)
        slice_410: "f32[25]" = torch.ops.aten.slice.Tensor(select_932, 0, 103, 9223372036854775807);  select_932 = None
        slice_411: "f32[2048, 25]" = torch.ops.aten.slice.Tensor(slice_scatter_101, 1, 103, 9223372036854775807)
        expand_207: "f32[2048, 25]" = torch.ops.aten.expand.default(slice_411, [2048, 25]);  slice_411 = None
        mul_309: "f32[2048, 25]" = torch.ops.aten.mul.Tensor(expand_207, 1);  expand_207 = None
        view_414: "f32[2048, 1]" = torch.ops.aten.view.default(div_103, [2048, 1]);  div_103 = None
        mul_310: "f32[2048, 25]" = torch.ops.aten.mul.Tensor(view_414, slice_410);  view_414 = slice_410 = None
        mul_311: "f32[2048, 25]" = torch.ops.aten.mul.Tensor(mul_310, -1.0);  mul_310 = None
        add_103: "f32[2048, 25]" = torch.ops.aten.add.Tensor(mul_309, mul_311);  mul_309 = mul_311 = None
        slice_scatter_102: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_101, add_103, 1, 103, 9223372036854775807);  slice_scatter_101 = add_103 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_934: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 104)
        select_935: "f32[]" = torch.ops.aten.select.int(select_934, 0, 104);  select_934 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_936: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_102, 1, 104)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_208: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_104: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_937: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_102, 1, 104)
        view_416: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_937, [8, 128, 2]);  select_937 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_104: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_208, view_416, permute_104, alpha = -2.0);  unsqueeze_208 = view_416 = permute_104 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_104: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_104, -1);  baddbmm_104 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_209: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_104, -1)
        expand_208: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_209, [8, 128, 2]);  unsqueeze_209 = None
        gather_104: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_208);  expand_208 = None
        view_417: "f32[2048]" = torch.ops.aten.view.default(gather_104, [-1]);  gather_104 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_939: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_103, 1, 104)
        copy_104: "i64[8, 128]" = torch.ops.aten.copy.default(select_939, argmin_104);  select_939 = argmin_104 = None
        select_scatter_104: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_103, copy_104, 1, 104);  select_scatter_103 = copy_104 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_104: "f32[2048]" = torch.ops.aten.sub.Tensor(select_936, view_417);  select_936 = view_417 = None
        div_104: "f32[2048]" = torch.ops.aten.div.Tensor(sub_104, select_935);  sub_104 = select_935 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_941: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 104)
        slice_414: "f32[24]" = torch.ops.aten.slice.Tensor(select_941, 0, 104, 9223372036854775807);  select_941 = None
        slice_415: "f32[2048, 24]" = torch.ops.aten.slice.Tensor(slice_scatter_102, 1, 104, 9223372036854775807)
        expand_209: "f32[2048, 24]" = torch.ops.aten.expand.default(slice_415, [2048, 24]);  slice_415 = None
        mul_312: "f32[2048, 24]" = torch.ops.aten.mul.Tensor(expand_209, 1);  expand_209 = None
        view_418: "f32[2048, 1]" = torch.ops.aten.view.default(div_104, [2048, 1]);  div_104 = None
        mul_313: "f32[2048, 24]" = torch.ops.aten.mul.Tensor(view_418, slice_414);  view_418 = slice_414 = None
        mul_314: "f32[2048, 24]" = torch.ops.aten.mul.Tensor(mul_313, -1.0);  mul_313 = None
        add_104: "f32[2048, 24]" = torch.ops.aten.add.Tensor(mul_312, mul_314);  mul_312 = mul_314 = None
        slice_scatter_103: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_102, add_104, 1, 104, 9223372036854775807);  slice_scatter_102 = add_104 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_943: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 105)
        select_944: "f32[]" = torch.ops.aten.select.int(select_943, 0, 105);  select_943 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_945: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_103, 1, 105)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_210: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_105: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_946: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_103, 1, 105)
        view_420: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_946, [8, 128, 2]);  select_946 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_105: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_210, view_420, permute_105, alpha = -2.0);  unsqueeze_210 = view_420 = permute_105 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_105: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_105, -1);  baddbmm_105 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_211: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_105, -1)
        expand_210: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_211, [8, 128, 2]);  unsqueeze_211 = None
        gather_105: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_210);  expand_210 = None
        view_421: "f32[2048]" = torch.ops.aten.view.default(gather_105, [-1]);  gather_105 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_948: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_104, 1, 105)
        copy_105: "i64[8, 128]" = torch.ops.aten.copy.default(select_948, argmin_105);  select_948 = argmin_105 = None
        select_scatter_105: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_104, copy_105, 1, 105);  select_scatter_104 = copy_105 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_105: "f32[2048]" = torch.ops.aten.sub.Tensor(select_945, view_421);  select_945 = view_421 = None
        div_105: "f32[2048]" = torch.ops.aten.div.Tensor(sub_105, select_944);  sub_105 = select_944 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_950: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 105)
        slice_418: "f32[23]" = torch.ops.aten.slice.Tensor(select_950, 0, 105, 9223372036854775807);  select_950 = None
        slice_419: "f32[2048, 23]" = torch.ops.aten.slice.Tensor(slice_scatter_103, 1, 105, 9223372036854775807)
        expand_211: "f32[2048, 23]" = torch.ops.aten.expand.default(slice_419, [2048, 23]);  slice_419 = None
        mul_315: "f32[2048, 23]" = torch.ops.aten.mul.Tensor(expand_211, 1);  expand_211 = None
        view_422: "f32[2048, 1]" = torch.ops.aten.view.default(div_105, [2048, 1]);  div_105 = None
        mul_316: "f32[2048, 23]" = torch.ops.aten.mul.Tensor(view_422, slice_418);  view_422 = slice_418 = None
        mul_317: "f32[2048, 23]" = torch.ops.aten.mul.Tensor(mul_316, -1.0);  mul_316 = None
        add_105: "f32[2048, 23]" = torch.ops.aten.add.Tensor(mul_315, mul_317);  mul_315 = mul_317 = None
        slice_scatter_104: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_103, add_105, 1, 105, 9223372036854775807);  slice_scatter_103 = add_105 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_952: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 106)
        select_953: "f32[]" = torch.ops.aten.select.int(select_952, 0, 106);  select_952 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_954: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_104, 1, 106)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_212: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_106: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_955: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_104, 1, 106)
        view_424: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_955, [8, 128, 2]);  select_955 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_106: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_212, view_424, permute_106, alpha = -2.0);  unsqueeze_212 = view_424 = permute_106 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_106: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_106, -1);  baddbmm_106 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_213: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_106, -1)
        expand_212: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_213, [8, 128, 2]);  unsqueeze_213 = None
        gather_106: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_212);  expand_212 = None
        view_425: "f32[2048]" = torch.ops.aten.view.default(gather_106, [-1]);  gather_106 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_957: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_105, 1, 106)
        copy_106: "i64[8, 128]" = torch.ops.aten.copy.default(select_957, argmin_106);  select_957 = argmin_106 = None
        select_scatter_106: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_105, copy_106, 1, 106);  select_scatter_105 = copy_106 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_106: "f32[2048]" = torch.ops.aten.sub.Tensor(select_954, view_425);  select_954 = view_425 = None
        div_106: "f32[2048]" = torch.ops.aten.div.Tensor(sub_106, select_953);  sub_106 = select_953 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_959: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 106)
        slice_422: "f32[22]" = torch.ops.aten.slice.Tensor(select_959, 0, 106, 9223372036854775807);  select_959 = None
        slice_423: "f32[2048, 22]" = torch.ops.aten.slice.Tensor(slice_scatter_104, 1, 106, 9223372036854775807)
        expand_213: "f32[2048, 22]" = torch.ops.aten.expand.default(slice_423, [2048, 22]);  slice_423 = None
        mul_318: "f32[2048, 22]" = torch.ops.aten.mul.Tensor(expand_213, 1);  expand_213 = None
        view_426: "f32[2048, 1]" = torch.ops.aten.view.default(div_106, [2048, 1]);  div_106 = None
        mul_319: "f32[2048, 22]" = torch.ops.aten.mul.Tensor(view_426, slice_422);  view_426 = slice_422 = None
        mul_320: "f32[2048, 22]" = torch.ops.aten.mul.Tensor(mul_319, -1.0);  mul_319 = None
        add_106: "f32[2048, 22]" = torch.ops.aten.add.Tensor(mul_318, mul_320);  mul_318 = mul_320 = None
        slice_scatter_105: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_104, add_106, 1, 106, 9223372036854775807);  slice_scatter_104 = add_106 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_961: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 107)
        select_962: "f32[]" = torch.ops.aten.select.int(select_961, 0, 107);  select_961 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_963: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_105, 1, 107)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_214: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_107: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_964: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_105, 1, 107)
        view_428: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_964, [8, 128, 2]);  select_964 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_107: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_214, view_428, permute_107, alpha = -2.0);  unsqueeze_214 = view_428 = permute_107 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_107: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_107, -1);  baddbmm_107 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_215: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_107, -1)
        expand_214: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_215, [8, 128, 2]);  unsqueeze_215 = None
        gather_107: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_214);  expand_214 = None
        view_429: "f32[2048]" = torch.ops.aten.view.default(gather_107, [-1]);  gather_107 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_966: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_106, 1, 107)
        copy_107: "i64[8, 128]" = torch.ops.aten.copy.default(select_966, argmin_107);  select_966 = argmin_107 = None
        select_scatter_107: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_106, copy_107, 1, 107);  select_scatter_106 = copy_107 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_107: "f32[2048]" = torch.ops.aten.sub.Tensor(select_963, view_429);  select_963 = view_429 = None
        div_107: "f32[2048]" = torch.ops.aten.div.Tensor(sub_107, select_962);  sub_107 = select_962 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_968: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 107)
        slice_426: "f32[21]" = torch.ops.aten.slice.Tensor(select_968, 0, 107, 9223372036854775807);  select_968 = None
        slice_427: "f32[2048, 21]" = torch.ops.aten.slice.Tensor(slice_scatter_105, 1, 107, 9223372036854775807)
        expand_215: "f32[2048, 21]" = torch.ops.aten.expand.default(slice_427, [2048, 21]);  slice_427 = None
        mul_321: "f32[2048, 21]" = torch.ops.aten.mul.Tensor(expand_215, 1);  expand_215 = None
        view_430: "f32[2048, 1]" = torch.ops.aten.view.default(div_107, [2048, 1]);  div_107 = None
        mul_322: "f32[2048, 21]" = torch.ops.aten.mul.Tensor(view_430, slice_426);  view_430 = slice_426 = None
        mul_323: "f32[2048, 21]" = torch.ops.aten.mul.Tensor(mul_322, -1.0);  mul_322 = None
        add_107: "f32[2048, 21]" = torch.ops.aten.add.Tensor(mul_321, mul_323);  mul_321 = mul_323 = None
        slice_scatter_106: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_105, add_107, 1, 107, 9223372036854775807);  slice_scatter_105 = add_107 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_970: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 108)
        select_971: "f32[]" = torch.ops.aten.select.int(select_970, 0, 108);  select_970 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_972: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_106, 1, 108)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_216: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_108: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_973: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_106, 1, 108)
        view_432: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_973, [8, 128, 2]);  select_973 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_108: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_216, view_432, permute_108, alpha = -2.0);  unsqueeze_216 = view_432 = permute_108 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_108: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_108, -1);  baddbmm_108 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_217: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_108, -1)
        expand_216: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_217, [8, 128, 2]);  unsqueeze_217 = None
        gather_108: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_216);  expand_216 = None
        view_433: "f32[2048]" = torch.ops.aten.view.default(gather_108, [-1]);  gather_108 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_975: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_107, 1, 108)
        copy_108: "i64[8, 128]" = torch.ops.aten.copy.default(select_975, argmin_108);  select_975 = argmin_108 = None
        select_scatter_108: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_107, copy_108, 1, 108);  select_scatter_107 = copy_108 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_108: "f32[2048]" = torch.ops.aten.sub.Tensor(select_972, view_433);  select_972 = view_433 = None
        div_108: "f32[2048]" = torch.ops.aten.div.Tensor(sub_108, select_971);  sub_108 = select_971 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_977: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 108)
        slice_430: "f32[20]" = torch.ops.aten.slice.Tensor(select_977, 0, 108, 9223372036854775807);  select_977 = None
        slice_431: "f32[2048, 20]" = torch.ops.aten.slice.Tensor(slice_scatter_106, 1, 108, 9223372036854775807)
        expand_217: "f32[2048, 20]" = torch.ops.aten.expand.default(slice_431, [2048, 20]);  slice_431 = None
        mul_324: "f32[2048, 20]" = torch.ops.aten.mul.Tensor(expand_217, 1);  expand_217 = None
        view_434: "f32[2048, 1]" = torch.ops.aten.view.default(div_108, [2048, 1]);  div_108 = None
        mul_325: "f32[2048, 20]" = torch.ops.aten.mul.Tensor(view_434, slice_430);  view_434 = slice_430 = None
        mul_326: "f32[2048, 20]" = torch.ops.aten.mul.Tensor(mul_325, -1.0);  mul_325 = None
        add_108: "f32[2048, 20]" = torch.ops.aten.add.Tensor(mul_324, mul_326);  mul_324 = mul_326 = None
        slice_scatter_107: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_106, add_108, 1, 108, 9223372036854775807);  slice_scatter_106 = add_108 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_979: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 109)
        select_980: "f32[]" = torch.ops.aten.select.int(select_979, 0, 109);  select_979 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_981: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_107, 1, 109)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_218: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_109: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_982: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_107, 1, 109)
        view_436: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_982, [8, 128, 2]);  select_982 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_109: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_218, view_436, permute_109, alpha = -2.0);  unsqueeze_218 = view_436 = permute_109 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_109: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_109, -1);  baddbmm_109 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_219: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_109, -1)
        expand_218: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_219, [8, 128, 2]);  unsqueeze_219 = None
        gather_109: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_218);  expand_218 = None
        view_437: "f32[2048]" = torch.ops.aten.view.default(gather_109, [-1]);  gather_109 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_984: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_108, 1, 109)
        copy_109: "i64[8, 128]" = torch.ops.aten.copy.default(select_984, argmin_109);  select_984 = argmin_109 = None
        select_scatter_109: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_108, copy_109, 1, 109);  select_scatter_108 = copy_109 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_109: "f32[2048]" = torch.ops.aten.sub.Tensor(select_981, view_437);  select_981 = view_437 = None
        div_109: "f32[2048]" = torch.ops.aten.div.Tensor(sub_109, select_980);  sub_109 = select_980 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_986: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 109)
        slice_434: "f32[19]" = torch.ops.aten.slice.Tensor(select_986, 0, 109, 9223372036854775807);  select_986 = None
        slice_435: "f32[2048, 19]" = torch.ops.aten.slice.Tensor(slice_scatter_107, 1, 109, 9223372036854775807)
        expand_219: "f32[2048, 19]" = torch.ops.aten.expand.default(slice_435, [2048, 19]);  slice_435 = None
        mul_327: "f32[2048, 19]" = torch.ops.aten.mul.Tensor(expand_219, 1);  expand_219 = None
        view_438: "f32[2048, 1]" = torch.ops.aten.view.default(div_109, [2048, 1]);  div_109 = None
        mul_328: "f32[2048, 19]" = torch.ops.aten.mul.Tensor(view_438, slice_434);  view_438 = slice_434 = None
        mul_329: "f32[2048, 19]" = torch.ops.aten.mul.Tensor(mul_328, -1.0);  mul_328 = None
        add_109: "f32[2048, 19]" = torch.ops.aten.add.Tensor(mul_327, mul_329);  mul_327 = mul_329 = None
        slice_scatter_108: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_107, add_109, 1, 109, 9223372036854775807);  slice_scatter_107 = add_109 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_988: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 110)
        select_989: "f32[]" = torch.ops.aten.select.int(select_988, 0, 110);  select_988 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_990: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_108, 1, 110)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_220: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_110: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_991: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_108, 1, 110)
        view_440: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_991, [8, 128, 2]);  select_991 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_110: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_220, view_440, permute_110, alpha = -2.0);  unsqueeze_220 = view_440 = permute_110 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_110: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_110, -1);  baddbmm_110 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_221: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_110, -1)
        expand_220: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_221, [8, 128, 2]);  unsqueeze_221 = None
        gather_110: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_220);  expand_220 = None
        view_441: "f32[2048]" = torch.ops.aten.view.default(gather_110, [-1]);  gather_110 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_993: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_109, 1, 110)
        copy_110: "i64[8, 128]" = torch.ops.aten.copy.default(select_993, argmin_110);  select_993 = argmin_110 = None
        select_scatter_110: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_109, copy_110, 1, 110);  select_scatter_109 = copy_110 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_110: "f32[2048]" = torch.ops.aten.sub.Tensor(select_990, view_441);  select_990 = view_441 = None
        div_110: "f32[2048]" = torch.ops.aten.div.Tensor(sub_110, select_989);  sub_110 = select_989 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_995: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 110)
        slice_438: "f32[18]" = torch.ops.aten.slice.Tensor(select_995, 0, 110, 9223372036854775807);  select_995 = None
        slice_439: "f32[2048, 18]" = torch.ops.aten.slice.Tensor(slice_scatter_108, 1, 110, 9223372036854775807)
        expand_221: "f32[2048, 18]" = torch.ops.aten.expand.default(slice_439, [2048, 18]);  slice_439 = None
        mul_330: "f32[2048, 18]" = torch.ops.aten.mul.Tensor(expand_221, 1);  expand_221 = None
        view_442: "f32[2048, 1]" = torch.ops.aten.view.default(div_110, [2048, 1]);  div_110 = None
        mul_331: "f32[2048, 18]" = torch.ops.aten.mul.Tensor(view_442, slice_438);  view_442 = slice_438 = None
        mul_332: "f32[2048, 18]" = torch.ops.aten.mul.Tensor(mul_331, -1.0);  mul_331 = None
        add_110: "f32[2048, 18]" = torch.ops.aten.add.Tensor(mul_330, mul_332);  mul_330 = mul_332 = None
        slice_scatter_109: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_108, add_110, 1, 110, 9223372036854775807);  slice_scatter_108 = add_110 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_997: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 111)
        select_998: "f32[]" = torch.ops.aten.select.int(select_997, 0, 111);  select_997 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_999: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_109, 1, 111)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_222: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_111: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_1000: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_109, 1, 111)
        view_444: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_1000, [8, 128, 2]);  select_1000 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_111: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_222, view_444, permute_111, alpha = -2.0);  unsqueeze_222 = view_444 = permute_111 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_111: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_111, -1);  baddbmm_111 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_223: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_111, -1)
        expand_222: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_223, [8, 128, 2]);  unsqueeze_223 = None
        gather_111: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_222);  expand_222 = None
        view_445: "f32[2048]" = torch.ops.aten.view.default(gather_111, [-1]);  gather_111 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_1002: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_110, 1, 111)
        copy_111: "i64[8, 128]" = torch.ops.aten.copy.default(select_1002, argmin_111);  select_1002 = argmin_111 = None
        select_scatter_111: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_110, copy_111, 1, 111);  select_scatter_110 = copy_111 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_111: "f32[2048]" = torch.ops.aten.sub.Tensor(select_999, view_445);  select_999 = view_445 = None
        div_111: "f32[2048]" = torch.ops.aten.div.Tensor(sub_111, select_998);  sub_111 = select_998 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_1004: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 111)
        slice_442: "f32[17]" = torch.ops.aten.slice.Tensor(select_1004, 0, 111, 9223372036854775807);  select_1004 = None
        slice_443: "f32[2048, 17]" = torch.ops.aten.slice.Tensor(slice_scatter_109, 1, 111, 9223372036854775807)
        expand_223: "f32[2048, 17]" = torch.ops.aten.expand.default(slice_443, [2048, 17]);  slice_443 = None
        mul_333: "f32[2048, 17]" = torch.ops.aten.mul.Tensor(expand_223, 1);  expand_223 = None
        view_446: "f32[2048, 1]" = torch.ops.aten.view.default(div_111, [2048, 1]);  div_111 = None
        mul_334: "f32[2048, 17]" = torch.ops.aten.mul.Tensor(view_446, slice_442);  view_446 = slice_442 = None
        mul_335: "f32[2048, 17]" = torch.ops.aten.mul.Tensor(mul_334, -1.0);  mul_334 = None
        add_111: "f32[2048, 17]" = torch.ops.aten.add.Tensor(mul_333, mul_335);  mul_333 = mul_335 = None
        slice_scatter_110: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_109, add_111, 1, 111, 9223372036854775807);  slice_scatter_109 = add_111 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_1006: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 112)
        select_1007: "f32[]" = torch.ops.aten.select.int(select_1006, 0, 112);  select_1006 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_1008: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_110, 1, 112)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_224: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_112: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_1009: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_110, 1, 112)
        view_448: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_1009, [8, 128, 2]);  select_1009 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_112: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_224, view_448, permute_112, alpha = -2.0);  unsqueeze_224 = view_448 = permute_112 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_112: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_112, -1);  baddbmm_112 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_225: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_112, -1)
        expand_224: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_225, [8, 128, 2]);  unsqueeze_225 = None
        gather_112: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_224);  expand_224 = None
        view_449: "f32[2048]" = torch.ops.aten.view.default(gather_112, [-1]);  gather_112 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_1011: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_111, 1, 112)
        copy_112: "i64[8, 128]" = torch.ops.aten.copy.default(select_1011, argmin_112);  select_1011 = argmin_112 = None
        select_scatter_112: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_111, copy_112, 1, 112);  select_scatter_111 = copy_112 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_112: "f32[2048]" = torch.ops.aten.sub.Tensor(select_1008, view_449);  select_1008 = view_449 = None
        div_112: "f32[2048]" = torch.ops.aten.div.Tensor(sub_112, select_1007);  sub_112 = select_1007 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_1013: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 112)
        slice_446: "f32[16]" = torch.ops.aten.slice.Tensor(select_1013, 0, 112, 9223372036854775807);  select_1013 = None
        slice_447: "f32[2048, 16]" = torch.ops.aten.slice.Tensor(slice_scatter_110, 1, 112, 9223372036854775807)
        expand_225: "f32[2048, 16]" = torch.ops.aten.expand.default(slice_447, [2048, 16]);  slice_447 = None
        mul_336: "f32[2048, 16]" = torch.ops.aten.mul.Tensor(expand_225, 1);  expand_225 = None
        view_450: "f32[2048, 1]" = torch.ops.aten.view.default(div_112, [2048, 1]);  div_112 = None
        mul_337: "f32[2048, 16]" = torch.ops.aten.mul.Tensor(view_450, slice_446);  view_450 = slice_446 = None
        mul_338: "f32[2048, 16]" = torch.ops.aten.mul.Tensor(mul_337, -1.0);  mul_337 = None
        add_112: "f32[2048, 16]" = torch.ops.aten.add.Tensor(mul_336, mul_338);  mul_336 = mul_338 = None
        slice_scatter_111: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_110, add_112, 1, 112, 9223372036854775807);  slice_scatter_110 = add_112 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_1015: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 113)
        select_1016: "f32[]" = torch.ops.aten.select.int(select_1015, 0, 113);  select_1015 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_1017: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_111, 1, 113)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_226: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_113: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_1018: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_111, 1, 113)
        view_452: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_1018, [8, 128, 2]);  select_1018 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_113: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_226, view_452, permute_113, alpha = -2.0);  unsqueeze_226 = view_452 = permute_113 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_113: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_113, -1);  baddbmm_113 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_227: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_113, -1)
        expand_226: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_227, [8, 128, 2]);  unsqueeze_227 = None
        gather_113: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_226);  expand_226 = None
        view_453: "f32[2048]" = torch.ops.aten.view.default(gather_113, [-1]);  gather_113 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_1020: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_112, 1, 113)
        copy_113: "i64[8, 128]" = torch.ops.aten.copy.default(select_1020, argmin_113);  select_1020 = argmin_113 = None
        select_scatter_113: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_112, copy_113, 1, 113);  select_scatter_112 = copy_113 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_113: "f32[2048]" = torch.ops.aten.sub.Tensor(select_1017, view_453);  select_1017 = view_453 = None
        div_113: "f32[2048]" = torch.ops.aten.div.Tensor(sub_113, select_1016);  sub_113 = select_1016 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_1022: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 113)
        slice_450: "f32[15]" = torch.ops.aten.slice.Tensor(select_1022, 0, 113, 9223372036854775807);  select_1022 = None
        slice_451: "f32[2048, 15]" = torch.ops.aten.slice.Tensor(slice_scatter_111, 1, 113, 9223372036854775807)
        expand_227: "f32[2048, 15]" = torch.ops.aten.expand.default(slice_451, [2048, 15]);  slice_451 = None
        mul_339: "f32[2048, 15]" = torch.ops.aten.mul.Tensor(expand_227, 1);  expand_227 = None
        view_454: "f32[2048, 1]" = torch.ops.aten.view.default(div_113, [2048, 1]);  div_113 = None
        mul_340: "f32[2048, 15]" = torch.ops.aten.mul.Tensor(view_454, slice_450);  view_454 = slice_450 = None
        mul_341: "f32[2048, 15]" = torch.ops.aten.mul.Tensor(mul_340, -1.0);  mul_340 = None
        add_113: "f32[2048, 15]" = torch.ops.aten.add.Tensor(mul_339, mul_341);  mul_339 = mul_341 = None
        slice_scatter_112: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_111, add_113, 1, 113, 9223372036854775807);  slice_scatter_111 = add_113 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_1024: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 114)
        select_1025: "f32[]" = torch.ops.aten.select.int(select_1024, 0, 114);  select_1024 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_1026: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_112, 1, 114)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_228: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_114: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_1027: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_112, 1, 114)
        view_456: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_1027, [8, 128, 2]);  select_1027 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_114: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_228, view_456, permute_114, alpha = -2.0);  unsqueeze_228 = view_456 = permute_114 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_114: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_114, -1);  baddbmm_114 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_229: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_114, -1)
        expand_228: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_229, [8, 128, 2]);  unsqueeze_229 = None
        gather_114: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_228);  expand_228 = None
        view_457: "f32[2048]" = torch.ops.aten.view.default(gather_114, [-1]);  gather_114 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_1029: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_113, 1, 114)
        copy_114: "i64[8, 128]" = torch.ops.aten.copy.default(select_1029, argmin_114);  select_1029 = argmin_114 = None
        select_scatter_114: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_113, copy_114, 1, 114);  select_scatter_113 = copy_114 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_114: "f32[2048]" = torch.ops.aten.sub.Tensor(select_1026, view_457);  select_1026 = view_457 = None
        div_114: "f32[2048]" = torch.ops.aten.div.Tensor(sub_114, select_1025);  sub_114 = select_1025 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_1031: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 114)
        slice_454: "f32[14]" = torch.ops.aten.slice.Tensor(select_1031, 0, 114, 9223372036854775807);  select_1031 = None
        slice_455: "f32[2048, 14]" = torch.ops.aten.slice.Tensor(slice_scatter_112, 1, 114, 9223372036854775807)
        expand_229: "f32[2048, 14]" = torch.ops.aten.expand.default(slice_455, [2048, 14]);  slice_455 = None
        mul_342: "f32[2048, 14]" = torch.ops.aten.mul.Tensor(expand_229, 1);  expand_229 = None
        view_458: "f32[2048, 1]" = torch.ops.aten.view.default(div_114, [2048, 1]);  div_114 = None
        mul_343: "f32[2048, 14]" = torch.ops.aten.mul.Tensor(view_458, slice_454);  view_458 = slice_454 = None
        mul_344: "f32[2048, 14]" = torch.ops.aten.mul.Tensor(mul_343, -1.0);  mul_343 = None
        add_114: "f32[2048, 14]" = torch.ops.aten.add.Tensor(mul_342, mul_344);  mul_342 = mul_344 = None
        slice_scatter_113: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_112, add_114, 1, 114, 9223372036854775807);  slice_scatter_112 = add_114 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_1033: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 115)
        select_1034: "f32[]" = torch.ops.aten.select.int(select_1033, 0, 115);  select_1033 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_1035: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_113, 1, 115)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_230: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_115: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_1036: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_113, 1, 115)
        view_460: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_1036, [8, 128, 2]);  select_1036 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_115: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_230, view_460, permute_115, alpha = -2.0);  unsqueeze_230 = view_460 = permute_115 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_115: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_115, -1);  baddbmm_115 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_231: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_115, -1)
        expand_230: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_231, [8, 128, 2]);  unsqueeze_231 = None
        gather_115: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_230);  expand_230 = None
        view_461: "f32[2048]" = torch.ops.aten.view.default(gather_115, [-1]);  gather_115 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_1038: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_114, 1, 115)
        copy_115: "i64[8, 128]" = torch.ops.aten.copy.default(select_1038, argmin_115);  select_1038 = argmin_115 = None
        select_scatter_115: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_114, copy_115, 1, 115);  select_scatter_114 = copy_115 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_115: "f32[2048]" = torch.ops.aten.sub.Tensor(select_1035, view_461);  select_1035 = view_461 = None
        div_115: "f32[2048]" = torch.ops.aten.div.Tensor(sub_115, select_1034);  sub_115 = select_1034 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_1040: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 115)
        slice_458: "f32[13]" = torch.ops.aten.slice.Tensor(select_1040, 0, 115, 9223372036854775807);  select_1040 = None
        slice_459: "f32[2048, 13]" = torch.ops.aten.slice.Tensor(slice_scatter_113, 1, 115, 9223372036854775807)
        expand_231: "f32[2048, 13]" = torch.ops.aten.expand.default(slice_459, [2048, 13]);  slice_459 = None
        mul_345: "f32[2048, 13]" = torch.ops.aten.mul.Tensor(expand_231, 1);  expand_231 = None
        view_462: "f32[2048, 1]" = torch.ops.aten.view.default(div_115, [2048, 1]);  div_115 = None
        mul_346: "f32[2048, 13]" = torch.ops.aten.mul.Tensor(view_462, slice_458);  view_462 = slice_458 = None
        mul_347: "f32[2048, 13]" = torch.ops.aten.mul.Tensor(mul_346, -1.0);  mul_346 = None
        add_115: "f32[2048, 13]" = torch.ops.aten.add.Tensor(mul_345, mul_347);  mul_345 = mul_347 = None
        slice_scatter_114: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_113, add_115, 1, 115, 9223372036854775807);  slice_scatter_113 = add_115 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_1042: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 116)
        select_1043: "f32[]" = torch.ops.aten.select.int(select_1042, 0, 116);  select_1042 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_1044: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_114, 1, 116)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_232: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_116: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_1045: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_114, 1, 116)
        view_464: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_1045, [8, 128, 2]);  select_1045 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_116: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_232, view_464, permute_116, alpha = -2.0);  unsqueeze_232 = view_464 = permute_116 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_116: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_116, -1);  baddbmm_116 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_233: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_116, -1)
        expand_232: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_233, [8, 128, 2]);  unsqueeze_233 = None
        gather_116: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_232);  expand_232 = None
        view_465: "f32[2048]" = torch.ops.aten.view.default(gather_116, [-1]);  gather_116 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_1047: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_115, 1, 116)
        copy_116: "i64[8, 128]" = torch.ops.aten.copy.default(select_1047, argmin_116);  select_1047 = argmin_116 = None
        select_scatter_116: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_115, copy_116, 1, 116);  select_scatter_115 = copy_116 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_116: "f32[2048]" = torch.ops.aten.sub.Tensor(select_1044, view_465);  select_1044 = view_465 = None
        div_116: "f32[2048]" = torch.ops.aten.div.Tensor(sub_116, select_1043);  sub_116 = select_1043 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_1049: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 116)
        slice_462: "f32[12]" = torch.ops.aten.slice.Tensor(select_1049, 0, 116, 9223372036854775807);  select_1049 = None
        slice_463: "f32[2048, 12]" = torch.ops.aten.slice.Tensor(slice_scatter_114, 1, 116, 9223372036854775807)
        expand_233: "f32[2048, 12]" = torch.ops.aten.expand.default(slice_463, [2048, 12]);  slice_463 = None
        mul_348: "f32[2048, 12]" = torch.ops.aten.mul.Tensor(expand_233, 1);  expand_233 = None
        view_466: "f32[2048, 1]" = torch.ops.aten.view.default(div_116, [2048, 1]);  div_116 = None
        mul_349: "f32[2048, 12]" = torch.ops.aten.mul.Tensor(view_466, slice_462);  view_466 = slice_462 = None
        mul_350: "f32[2048, 12]" = torch.ops.aten.mul.Tensor(mul_349, -1.0);  mul_349 = None
        add_116: "f32[2048, 12]" = torch.ops.aten.add.Tensor(mul_348, mul_350);  mul_348 = mul_350 = None
        slice_scatter_115: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_114, add_116, 1, 116, 9223372036854775807);  slice_scatter_114 = add_116 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_1051: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 117)
        select_1052: "f32[]" = torch.ops.aten.select.int(select_1051, 0, 117);  select_1051 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_1053: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_115, 1, 117)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_234: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_117: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_1054: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_115, 1, 117)
        view_468: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_1054, [8, 128, 2]);  select_1054 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_117: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_234, view_468, permute_117, alpha = -2.0);  unsqueeze_234 = view_468 = permute_117 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_117: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_117, -1);  baddbmm_117 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_235: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_117, -1)
        expand_234: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_235, [8, 128, 2]);  unsqueeze_235 = None
        gather_117: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_234);  expand_234 = None
        view_469: "f32[2048]" = torch.ops.aten.view.default(gather_117, [-1]);  gather_117 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_1056: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_116, 1, 117)
        copy_117: "i64[8, 128]" = torch.ops.aten.copy.default(select_1056, argmin_117);  select_1056 = argmin_117 = None
        select_scatter_117: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_116, copy_117, 1, 117);  select_scatter_116 = copy_117 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_117: "f32[2048]" = torch.ops.aten.sub.Tensor(select_1053, view_469);  select_1053 = view_469 = None
        div_117: "f32[2048]" = torch.ops.aten.div.Tensor(sub_117, select_1052);  sub_117 = select_1052 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_1058: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 117)
        slice_466: "f32[11]" = torch.ops.aten.slice.Tensor(select_1058, 0, 117, 9223372036854775807);  select_1058 = None
        slice_467: "f32[2048, 11]" = torch.ops.aten.slice.Tensor(slice_scatter_115, 1, 117, 9223372036854775807)
        expand_235: "f32[2048, 11]" = torch.ops.aten.expand.default(slice_467, [2048, 11]);  slice_467 = None
        mul_351: "f32[2048, 11]" = torch.ops.aten.mul.Tensor(expand_235, 1);  expand_235 = None
        view_470: "f32[2048, 1]" = torch.ops.aten.view.default(div_117, [2048, 1]);  div_117 = None
        mul_352: "f32[2048, 11]" = torch.ops.aten.mul.Tensor(view_470, slice_466);  view_470 = slice_466 = None
        mul_353: "f32[2048, 11]" = torch.ops.aten.mul.Tensor(mul_352, -1.0);  mul_352 = None
        add_117: "f32[2048, 11]" = torch.ops.aten.add.Tensor(mul_351, mul_353);  mul_351 = mul_353 = None
        slice_scatter_116: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_115, add_117, 1, 117, 9223372036854775807);  slice_scatter_115 = add_117 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_1060: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 118)
        select_1061: "f32[]" = torch.ops.aten.select.int(select_1060, 0, 118);  select_1060 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_1062: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_116, 1, 118)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_236: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_118: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_1063: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_116, 1, 118)
        view_472: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_1063, [8, 128, 2]);  select_1063 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_118: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_236, view_472, permute_118, alpha = -2.0);  unsqueeze_236 = view_472 = permute_118 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_118: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_118, -1);  baddbmm_118 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_237: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_118, -1)
        expand_236: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_237, [8, 128, 2]);  unsqueeze_237 = None
        gather_118: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_236);  expand_236 = None
        view_473: "f32[2048]" = torch.ops.aten.view.default(gather_118, [-1]);  gather_118 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_1065: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_117, 1, 118)
        copy_118: "i64[8, 128]" = torch.ops.aten.copy.default(select_1065, argmin_118);  select_1065 = argmin_118 = None
        select_scatter_118: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_117, copy_118, 1, 118);  select_scatter_117 = copy_118 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_118: "f32[2048]" = torch.ops.aten.sub.Tensor(select_1062, view_473);  select_1062 = view_473 = None
        div_118: "f32[2048]" = torch.ops.aten.div.Tensor(sub_118, select_1061);  sub_118 = select_1061 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_1067: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 118)
        slice_470: "f32[10]" = torch.ops.aten.slice.Tensor(select_1067, 0, 118, 9223372036854775807);  select_1067 = None
        slice_471: "f32[2048, 10]" = torch.ops.aten.slice.Tensor(slice_scatter_116, 1, 118, 9223372036854775807)
        expand_237: "f32[2048, 10]" = torch.ops.aten.expand.default(slice_471, [2048, 10]);  slice_471 = None
        mul_354: "f32[2048, 10]" = torch.ops.aten.mul.Tensor(expand_237, 1);  expand_237 = None
        view_474: "f32[2048, 1]" = torch.ops.aten.view.default(div_118, [2048, 1]);  div_118 = None
        mul_355: "f32[2048, 10]" = torch.ops.aten.mul.Tensor(view_474, slice_470);  view_474 = slice_470 = None
        mul_356: "f32[2048, 10]" = torch.ops.aten.mul.Tensor(mul_355, -1.0);  mul_355 = None
        add_118: "f32[2048, 10]" = torch.ops.aten.add.Tensor(mul_354, mul_356);  mul_354 = mul_356 = None
        slice_scatter_117: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_116, add_118, 1, 118, 9223372036854775807);  slice_scatter_116 = add_118 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_1069: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 119)
        select_1070: "f32[]" = torch.ops.aten.select.int(select_1069, 0, 119);  select_1069 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_1071: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_117, 1, 119)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_238: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_119: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_1072: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_117, 1, 119)
        view_476: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_1072, [8, 128, 2]);  select_1072 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_119: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_238, view_476, permute_119, alpha = -2.0);  unsqueeze_238 = view_476 = permute_119 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_119: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_119, -1);  baddbmm_119 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_239: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_119, -1)
        expand_238: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_239, [8, 128, 2]);  unsqueeze_239 = None
        gather_119: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_238);  expand_238 = None
        view_477: "f32[2048]" = torch.ops.aten.view.default(gather_119, [-1]);  gather_119 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_1074: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_118, 1, 119)
        copy_119: "i64[8, 128]" = torch.ops.aten.copy.default(select_1074, argmin_119);  select_1074 = argmin_119 = None
        select_scatter_119: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_118, copy_119, 1, 119);  select_scatter_118 = copy_119 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_119: "f32[2048]" = torch.ops.aten.sub.Tensor(select_1071, view_477);  select_1071 = view_477 = None
        div_119: "f32[2048]" = torch.ops.aten.div.Tensor(sub_119, select_1070);  sub_119 = select_1070 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_1076: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 119)
        slice_474: "f32[9]" = torch.ops.aten.slice.Tensor(select_1076, 0, 119, 9223372036854775807);  select_1076 = None
        slice_475: "f32[2048, 9]" = torch.ops.aten.slice.Tensor(slice_scatter_117, 1, 119, 9223372036854775807)
        expand_239: "f32[2048, 9]" = torch.ops.aten.expand.default(slice_475, [2048, 9]);  slice_475 = None
        mul_357: "f32[2048, 9]" = torch.ops.aten.mul.Tensor(expand_239, 1);  expand_239 = None
        view_478: "f32[2048, 1]" = torch.ops.aten.view.default(div_119, [2048, 1]);  div_119 = None
        mul_358: "f32[2048, 9]" = torch.ops.aten.mul.Tensor(view_478, slice_474);  view_478 = slice_474 = None
        mul_359: "f32[2048, 9]" = torch.ops.aten.mul.Tensor(mul_358, -1.0);  mul_358 = None
        add_119: "f32[2048, 9]" = torch.ops.aten.add.Tensor(mul_357, mul_359);  mul_357 = mul_359 = None
        slice_scatter_118: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_117, add_119, 1, 119, 9223372036854775807);  slice_scatter_117 = add_119 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_1078: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 120)
        select_1079: "f32[]" = torch.ops.aten.select.int(select_1078, 0, 120);  select_1078 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_1080: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_118, 1, 120)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_240: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_120: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_1081: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_118, 1, 120)
        view_480: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_1081, [8, 128, 2]);  select_1081 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_120: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_240, view_480, permute_120, alpha = -2.0);  unsqueeze_240 = view_480 = permute_120 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_120: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_120, -1);  baddbmm_120 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_241: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_120, -1)
        expand_240: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_241, [8, 128, 2]);  unsqueeze_241 = None
        gather_120: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_240);  expand_240 = None
        view_481: "f32[2048]" = torch.ops.aten.view.default(gather_120, [-1]);  gather_120 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_1083: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_119, 1, 120)
        copy_120: "i64[8, 128]" = torch.ops.aten.copy.default(select_1083, argmin_120);  select_1083 = argmin_120 = None
        select_scatter_120: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_119, copy_120, 1, 120);  select_scatter_119 = copy_120 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_120: "f32[2048]" = torch.ops.aten.sub.Tensor(select_1080, view_481);  select_1080 = view_481 = None
        div_120: "f32[2048]" = torch.ops.aten.div.Tensor(sub_120, select_1079);  sub_120 = select_1079 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_1085: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 120)
        slice_478: "f32[8]" = torch.ops.aten.slice.Tensor(select_1085, 0, 120, 9223372036854775807);  select_1085 = None
        slice_479: "f32[2048, 8]" = torch.ops.aten.slice.Tensor(slice_scatter_118, 1, 120, 9223372036854775807)
        expand_241: "f32[2048, 8]" = torch.ops.aten.expand.default(slice_479, [2048, 8]);  slice_479 = None
        mul_360: "f32[2048, 8]" = torch.ops.aten.mul.Tensor(expand_241, 1);  expand_241 = None
        view_482: "f32[2048, 1]" = torch.ops.aten.view.default(div_120, [2048, 1]);  div_120 = None
        mul_361: "f32[2048, 8]" = torch.ops.aten.mul.Tensor(view_482, slice_478);  view_482 = slice_478 = None
        mul_362: "f32[2048, 8]" = torch.ops.aten.mul.Tensor(mul_361, -1.0);  mul_361 = None
        add_120: "f32[2048, 8]" = torch.ops.aten.add.Tensor(mul_360, mul_362);  mul_360 = mul_362 = None
        slice_scatter_119: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_118, add_120, 1, 120, 9223372036854775807);  slice_scatter_118 = add_120 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_1087: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 121)
        select_1088: "f32[]" = torch.ops.aten.select.int(select_1087, 0, 121);  select_1087 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_1089: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_119, 1, 121)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_242: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_121: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_1090: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_119, 1, 121)
        view_484: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_1090, [8, 128, 2]);  select_1090 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_121: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_242, view_484, permute_121, alpha = -2.0);  unsqueeze_242 = view_484 = permute_121 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_121: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_121, -1);  baddbmm_121 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_243: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_121, -1)
        expand_242: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_243, [8, 128, 2]);  unsqueeze_243 = None
        gather_121: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_242);  expand_242 = None
        view_485: "f32[2048]" = torch.ops.aten.view.default(gather_121, [-1]);  gather_121 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_1092: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_120, 1, 121)
        copy_121: "i64[8, 128]" = torch.ops.aten.copy.default(select_1092, argmin_121);  select_1092 = argmin_121 = None
        select_scatter_121: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_120, copy_121, 1, 121);  select_scatter_120 = copy_121 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_121: "f32[2048]" = torch.ops.aten.sub.Tensor(select_1089, view_485);  select_1089 = view_485 = None
        div_121: "f32[2048]" = torch.ops.aten.div.Tensor(sub_121, select_1088);  sub_121 = select_1088 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_1094: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 121)
        slice_482: "f32[7]" = torch.ops.aten.slice.Tensor(select_1094, 0, 121, 9223372036854775807);  select_1094 = None
        slice_483: "f32[2048, 7]" = torch.ops.aten.slice.Tensor(slice_scatter_119, 1, 121, 9223372036854775807)
        expand_243: "f32[2048, 7]" = torch.ops.aten.expand.default(slice_483, [2048, 7]);  slice_483 = None
        mul_363: "f32[2048, 7]" = torch.ops.aten.mul.Tensor(expand_243, 1);  expand_243 = None
        view_486: "f32[2048, 1]" = torch.ops.aten.view.default(div_121, [2048, 1]);  div_121 = None
        mul_364: "f32[2048, 7]" = torch.ops.aten.mul.Tensor(view_486, slice_482);  view_486 = slice_482 = None
        mul_365: "f32[2048, 7]" = torch.ops.aten.mul.Tensor(mul_364, -1.0);  mul_364 = None
        add_121: "f32[2048, 7]" = torch.ops.aten.add.Tensor(mul_363, mul_365);  mul_363 = mul_365 = None
        slice_scatter_120: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_119, add_121, 1, 121, 9223372036854775807);  slice_scatter_119 = add_121 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_1096: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 122)
        select_1097: "f32[]" = torch.ops.aten.select.int(select_1096, 0, 122);  select_1096 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_1098: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_120, 1, 122)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_244: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_122: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_1099: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_120, 1, 122)
        view_488: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_1099, [8, 128, 2]);  select_1099 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_122: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_244, view_488, permute_122, alpha = -2.0);  unsqueeze_244 = view_488 = permute_122 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_122: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_122, -1);  baddbmm_122 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_245: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_122, -1)
        expand_244: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_245, [8, 128, 2]);  unsqueeze_245 = None
        gather_122: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_244);  expand_244 = None
        view_489: "f32[2048]" = torch.ops.aten.view.default(gather_122, [-1]);  gather_122 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_1101: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_121, 1, 122)
        copy_122: "i64[8, 128]" = torch.ops.aten.copy.default(select_1101, argmin_122);  select_1101 = argmin_122 = None
        select_scatter_122: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_121, copy_122, 1, 122);  select_scatter_121 = copy_122 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_122: "f32[2048]" = torch.ops.aten.sub.Tensor(select_1098, view_489);  select_1098 = view_489 = None
        div_122: "f32[2048]" = torch.ops.aten.div.Tensor(sub_122, select_1097);  sub_122 = select_1097 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_1103: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 122)
        slice_486: "f32[6]" = torch.ops.aten.slice.Tensor(select_1103, 0, 122, 9223372036854775807);  select_1103 = None
        slice_487: "f32[2048, 6]" = torch.ops.aten.slice.Tensor(slice_scatter_120, 1, 122, 9223372036854775807)
        expand_245: "f32[2048, 6]" = torch.ops.aten.expand.default(slice_487, [2048, 6]);  slice_487 = None
        mul_366: "f32[2048, 6]" = torch.ops.aten.mul.Tensor(expand_245, 1);  expand_245 = None
        view_490: "f32[2048, 1]" = torch.ops.aten.view.default(div_122, [2048, 1]);  div_122 = None
        mul_367: "f32[2048, 6]" = torch.ops.aten.mul.Tensor(view_490, slice_486);  view_490 = slice_486 = None
        mul_368: "f32[2048, 6]" = torch.ops.aten.mul.Tensor(mul_367, -1.0);  mul_367 = None
        add_122: "f32[2048, 6]" = torch.ops.aten.add.Tensor(mul_366, mul_368);  mul_366 = mul_368 = None
        slice_scatter_121: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_120, add_122, 1, 122, 9223372036854775807);  slice_scatter_120 = add_122 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_1105: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 123)
        select_1106: "f32[]" = torch.ops.aten.select.int(select_1105, 0, 123);  select_1105 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_1107: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_121, 1, 123)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_246: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_123: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_1108: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_121, 1, 123)
        view_492: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_1108, [8, 128, 2]);  select_1108 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_123: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_246, view_492, permute_123, alpha = -2.0);  unsqueeze_246 = view_492 = permute_123 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_123: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_123, -1);  baddbmm_123 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_247: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_123, -1)
        expand_246: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_247, [8, 128, 2]);  unsqueeze_247 = None
        gather_123: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_246);  expand_246 = None
        view_493: "f32[2048]" = torch.ops.aten.view.default(gather_123, [-1]);  gather_123 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_1110: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_122, 1, 123)
        copy_123: "i64[8, 128]" = torch.ops.aten.copy.default(select_1110, argmin_123);  select_1110 = argmin_123 = None
        select_scatter_123: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_122, copy_123, 1, 123);  select_scatter_122 = copy_123 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_123: "f32[2048]" = torch.ops.aten.sub.Tensor(select_1107, view_493);  select_1107 = view_493 = None
        div_123: "f32[2048]" = torch.ops.aten.div.Tensor(sub_123, select_1106);  sub_123 = select_1106 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_1112: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 123)
        slice_490: "f32[5]" = torch.ops.aten.slice.Tensor(select_1112, 0, 123, 9223372036854775807);  select_1112 = None
        slice_491: "f32[2048, 5]" = torch.ops.aten.slice.Tensor(slice_scatter_121, 1, 123, 9223372036854775807)
        expand_247: "f32[2048, 5]" = torch.ops.aten.expand.default(slice_491, [2048, 5]);  slice_491 = None
        mul_369: "f32[2048, 5]" = torch.ops.aten.mul.Tensor(expand_247, 1);  expand_247 = None
        view_494: "f32[2048, 1]" = torch.ops.aten.view.default(div_123, [2048, 1]);  div_123 = None
        mul_370: "f32[2048, 5]" = torch.ops.aten.mul.Tensor(view_494, slice_490);  view_494 = slice_490 = None
        mul_371: "f32[2048, 5]" = torch.ops.aten.mul.Tensor(mul_370, -1.0);  mul_370 = None
        add_123: "f32[2048, 5]" = torch.ops.aten.add.Tensor(mul_369, mul_371);  mul_369 = mul_371 = None
        slice_scatter_122: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_121, add_123, 1, 123, 9223372036854775807);  slice_scatter_121 = add_123 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_1114: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 124)
        select_1115: "f32[]" = torch.ops.aten.select.int(select_1114, 0, 124);  select_1114 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_1116: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_122, 1, 124)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_248: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_124: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_1117: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_122, 1, 124)
        view_496: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_1117, [8, 128, 2]);  select_1117 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_124: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_248, view_496, permute_124, alpha = -2.0);  unsqueeze_248 = view_496 = permute_124 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_124: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_124, -1);  baddbmm_124 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_249: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_124, -1)
        expand_248: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_249, [8, 128, 2]);  unsqueeze_249 = None
        gather_124: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_248);  expand_248 = None
        view_497: "f32[2048]" = torch.ops.aten.view.default(gather_124, [-1]);  gather_124 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_1119: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_123, 1, 124)
        copy_124: "i64[8, 128]" = torch.ops.aten.copy.default(select_1119, argmin_124);  select_1119 = argmin_124 = None
        select_scatter_124: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_123, copy_124, 1, 124);  select_scatter_123 = copy_124 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_124: "f32[2048]" = torch.ops.aten.sub.Tensor(select_1116, view_497);  select_1116 = view_497 = None
        div_124: "f32[2048]" = torch.ops.aten.div.Tensor(sub_124, select_1115);  sub_124 = select_1115 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_1121: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 124)
        slice_494: "f32[4]" = torch.ops.aten.slice.Tensor(select_1121, 0, 124, 9223372036854775807);  select_1121 = None
        slice_495: "f32[2048, 4]" = torch.ops.aten.slice.Tensor(slice_scatter_122, 1, 124, 9223372036854775807)
        expand_249: "f32[2048, 4]" = torch.ops.aten.expand.default(slice_495, [2048, 4]);  slice_495 = None
        mul_372: "f32[2048, 4]" = torch.ops.aten.mul.Tensor(expand_249, 1);  expand_249 = None
        view_498: "f32[2048, 1]" = torch.ops.aten.view.default(div_124, [2048, 1]);  div_124 = None
        mul_373: "f32[2048, 4]" = torch.ops.aten.mul.Tensor(view_498, slice_494);  view_498 = slice_494 = None
        mul_374: "f32[2048, 4]" = torch.ops.aten.mul.Tensor(mul_373, -1.0);  mul_373 = None
        add_124: "f32[2048, 4]" = torch.ops.aten.add.Tensor(mul_372, mul_374);  mul_372 = mul_374 = None
        slice_scatter_123: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_122, add_124, 1, 124, 9223372036854775807);  slice_scatter_122 = add_124 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_1123: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 125)
        select_1124: "f32[]" = torch.ops.aten.select.int(select_1123, 0, 125);  select_1123 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_1125: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_123, 1, 125)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_250: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_125: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_1126: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_123, 1, 125)
        view_500: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_1126, [8, 128, 2]);  select_1126 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_125: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_250, view_500, permute_125, alpha = -2.0);  unsqueeze_250 = view_500 = permute_125 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_125: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_125, -1);  baddbmm_125 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_251: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_125, -1)
        expand_250: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_251, [8, 128, 2]);  unsqueeze_251 = None
        gather_125: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_250);  expand_250 = None
        view_501: "f32[2048]" = torch.ops.aten.view.default(gather_125, [-1]);  gather_125 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_1128: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_124, 1, 125)
        copy_125: "i64[8, 128]" = torch.ops.aten.copy.default(select_1128, argmin_125);  select_1128 = argmin_125 = None
        select_scatter_125: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_124, copy_125, 1, 125);  select_scatter_124 = copy_125 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_125: "f32[2048]" = torch.ops.aten.sub.Tensor(select_1125, view_501);  select_1125 = view_501 = None
        div_125: "f32[2048]" = torch.ops.aten.div.Tensor(sub_125, select_1124);  sub_125 = select_1124 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_1130: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 125)
        slice_498: "f32[3]" = torch.ops.aten.slice.Tensor(select_1130, 0, 125, 9223372036854775807);  select_1130 = None
        slice_499: "f32[2048, 3]" = torch.ops.aten.slice.Tensor(slice_scatter_123, 1, 125, 9223372036854775807)
        expand_251: "f32[2048, 3]" = torch.ops.aten.expand.default(slice_499, [2048, 3]);  slice_499 = None
        mul_375: "f32[2048, 3]" = torch.ops.aten.mul.Tensor(expand_251, 1);  expand_251 = None
        view_502: "f32[2048, 1]" = torch.ops.aten.view.default(div_125, [2048, 1]);  div_125 = None
        mul_376: "f32[2048, 3]" = torch.ops.aten.mul.Tensor(view_502, slice_498);  view_502 = slice_498 = None
        mul_377: "f32[2048, 3]" = torch.ops.aten.mul.Tensor(mul_376, -1.0);  mul_376 = None
        add_125: "f32[2048, 3]" = torch.ops.aten.add.Tensor(mul_375, mul_377);  mul_375 = mul_377 = None
        slice_scatter_124: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_123, add_125, 1, 125, 9223372036854775807);  slice_scatter_123 = add_125 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_1132: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 126)
        select_1133: "f32[]" = torch.ops.aten.select.int(select_1132, 0, 126);  select_1132 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_1134: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_124, 1, 126)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_252: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_126: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_1135: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_124, 1, 126)
        view_504: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_1135, [8, 128, 2]);  select_1135 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_126: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_252, view_504, permute_126, alpha = -2.0);  unsqueeze_252 = view_504 = permute_126 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_126: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_126, -1);  baddbmm_126 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_253: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_126, -1)
        expand_252: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_253, [8, 128, 2]);  unsqueeze_253 = None
        gather_126: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_252);  expand_252 = None
        view_505: "f32[2048]" = torch.ops.aten.view.default(gather_126, [-1]);  gather_126 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_1137: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_125, 1, 126)
        copy_126: "i64[8, 128]" = torch.ops.aten.copy.default(select_1137, argmin_126);  select_1137 = argmin_126 = None
        select_scatter_126: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_125, copy_126, 1, 126);  select_scatter_125 = copy_126 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_126: "f32[2048]" = torch.ops.aten.sub.Tensor(select_1134, view_505);  select_1134 = view_505 = None
        div_126: "f32[2048]" = torch.ops.aten.div.Tensor(sub_126, select_1133);  sub_126 = select_1133 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_1139: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 126)
        slice_502: "f32[2]" = torch.ops.aten.slice.Tensor(select_1139, 0, 126, 9223372036854775807);  select_1139 = None
        slice_503: "f32[2048, 2]" = torch.ops.aten.slice.Tensor(slice_scatter_124, 1, 126, 9223372036854775807)
        expand_253: "f32[2048, 2]" = torch.ops.aten.expand.default(slice_503, [2048, 2]);  slice_503 = None
        mul_378: "f32[2048, 2]" = torch.ops.aten.mul.Tensor(expand_253, 1);  expand_253 = None
        view_506: "f32[2048, 1]" = torch.ops.aten.view.default(div_126, [2048, 1]);  div_126 = None
        mul_379: "f32[2048, 2]" = torch.ops.aten.mul.Tensor(view_506, slice_502);  view_506 = slice_502 = None
        mul_380: "f32[2048, 2]" = torch.ops.aten.mul.Tensor(mul_379, -1.0);  mul_379 = None
        add_126: "f32[2048, 2]" = torch.ops.aten.add.Tensor(mul_378, mul_380);  mul_378 = mul_380 = None
        slice_scatter_125: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_124, add_126, 1, 126, 9223372036854775807);  slice_scatter_124 = add_126 = None
        
        # File: /tmp/try_compile.py:15 in block_loop, code: d = Hinv1[j, j]
        select_1141: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 127)
        select_1142: "f32[]" = torch.ops.aten.select.int(select_1141, 0, 127);  select_1141 = None
        
        # File: /tmp/try_compile.py:14 in block_loop, code: w = W1[:, j]
        select_1143: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_125, 1, 127)
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        unsqueeze_254: "f32[8, 1, 16]" = torch.ops.aten.unsqueeze.default(arg2_1, 1);  arg2_1 = None
        permute_127: "f32[8, 2, 16]" = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        
        # File: /tmp/try_compile.py:16 in block_loop, code: vecs = w.view(n_row, nvec, v)
        select_1144: "f32[2048]" = torch.ops.aten.select.int(slice_scatter_125, 1, 127)
        view_508: "f32[8, 128, 2]" = torch.ops.aten.view.default(select_1144, [8, 128, 2]);  select_1144 = None
        
        # File: /tmp/try_compile.py:17 in block_loop, code: d2 = torch.baddbmm(csq.unsqueeze(1), vecs, csb.transpose(1, 2), alpha=-2.0)
        baddbmm_127: "f32[8, 128, 16]" = torch.ops.aten.baddbmm.default(unsqueeze_254, view_508, permute_127, alpha = -2.0);  unsqueeze_254 = view_508 = permute_127 = None
        
        # File: /tmp/try_compile.py:18 in block_loop, code: idx = d2.argmin(-1)
        argmin_127: "i64[8, 128]" = torch.ops.aten.argmin.default(baddbmm_127, -1);  baddbmm_127 = None
        
        # File: /tmp/try_compile.py:19 in block_loop, code: q = torch.gather(csb, 1, idx.unsqueeze(-1).expand(n_row, nvec, v)).reshape(-1)
        unsqueeze_255: "i64[8, 128, 1]" = torch.ops.aten.unsqueeze.default(argmin_127, -1)
        expand_254: "i64[8, 128, 2]" = torch.ops.aten.expand.default(unsqueeze_255, [8, 128, 2]);  unsqueeze_255 = None
        gather_127: "f32[8, 128, 2]" = torch.ops.aten.gather.default(arg3_1, 1, expand_254);  arg3_1 = expand_254 = None
        view_509: "f32[2048]" = torch.ops.aten.view.default(gather_127, [-1]);  gather_127 = None
        
        # File: /tmp/try_compile.py:20 in block_loop, code: idx_out[:, j] = idx
        select_1146: "i64[8, 128]" = torch.ops.aten.select.int(select_scatter_126, 1, 127)
        copy_127: "i64[8, 128]" = torch.ops.aten.copy.default(select_1146, argmin_127);  select_1146 = argmin_127 = None
        select_scatter_127: "i64[8, 128, 128]" = torch.ops.aten.select_scatter.default(select_scatter_126, copy_127, 1, 127);  select_scatter_126 = copy_127 = None
        
        # File: /tmp/try_compile.py:21 in block_loop, code: err = (w - q) / d
        sub_127: "f32[2048]" = torch.ops.aten.sub.Tensor(select_1143, view_509);  select_1143 = view_509 = None
        div_127: "f32[2048]" = torch.ops.aten.div.Tensor(sub_127, select_1142);  sub_127 = select_1142 = None
        
        # File: /tmp/try_compile.py:22 in block_loop, code: W1[:, j:].addr_(err, Hinv1[j, j:], alpha=-1.0)
        select_1148: "f32[128]" = torch.ops.aten.select.int(arg1_1, 0, 127);  arg1_1 = None
        slice_506: "f32[1]" = torch.ops.aten.slice.Tensor(select_1148, 0, 127, 9223372036854775807);  select_1148 = None
        slice_507: "f32[2048, 1]" = torch.ops.aten.slice.Tensor(slice_scatter_125, 1, 127, 9223372036854775807)
        expand_255: "f32[2048, 1]" = torch.ops.aten.expand.default(slice_507, [2048, 1]);  slice_507 = None
        mul_381: "f32[2048, 1]" = torch.ops.aten.mul.Tensor(expand_255, 1);  expand_255 = None
        view_510: "f32[2048, 1]" = torch.ops.aten.view.default(div_127, [2048, 1]);  div_127 = None
        mul_382: "f32[2048, 1]" = torch.ops.aten.mul.Tensor(view_510, slice_506);  view_510 = slice_506 = None
        mul_383: "f32[2048, 1]" = torch.ops.aten.mul.Tensor(mul_382, -1.0);  mul_382 = None
        add_127: "f32[2048, 1]" = torch.ops.aten.add.Tensor(mul_381, mul_383);  mul_381 = mul_383 = None
        slice_scatter_126: "f32[2048, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_125, add_127, 1, 127, 9223372036854775807);  slice_scatter_125 = add_127 = None
        copy_: "f32[2048, 128]" = torch.ops.aten.copy_.default(arg0_1, slice_scatter_126);  arg0_1 = slice_scatter_126 = copy_ = None
        return (select_scatter_127,)
        