
import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_root'
os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'default'
os.environ['TORCHINDUCTOR_COMPREHENSIVE_PADDING'] = '0'
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.automatic_dynamic_shapes = False
torch._inductor.config.allow_buffer_reuse = False
torch._inductor.config.post_grad_fusion_options = {'fav3_partition': {}}
torch._inductor.config.fallback_random = True
torch._inductor.config.compile_threads = 1
torch._inductor.config.comprehensive_padding = False
torch._inductor.config.triton.autotune_at_compile_time = False
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False



isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.10.0+cpu
# torch cuda version: None
# torch git version: 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
        full_default = torch.ops.aten.full.default([8, 128, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='npu', index=0), pin_memory = False)
        select = torch.ops.aten.select.int(arg0_1, 1, 0)
        select_1 = torch.ops.aten.select.int(arg1_1, 0, 0)
        select_2 = torch.ops.aten.select.int(select_1, 0, 0);  select_1 = None
        view = torch.ops.aten.view.default(select, [8, 128, 2])
        unsqueeze = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        baddbmm = torch.ops.aten.baddbmm.default(unsqueeze, view, permute, alpha = -2.0);  unsqueeze = view = permute = None
        argmin = torch.ops.aten.argmin.default(baddbmm, -1);  baddbmm = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(argmin, -1)
        expand = torch.ops.aten.expand.default(unsqueeze_1, [8, 128, 2]);  unsqueeze_1 = None
        gather = torch.ops.aten.gather.default(arg3_1, 1, expand);  expand = None
        view_1 = torch.ops.aten.view.default(gather, [-1]);  gather = None
        select_3 = torch.ops.aten.select.int(full_default, 1, 0)
        copy = torch.ops.aten.copy.default(select_3, argmin);  select_3 = argmin = None
        select_scatter = torch.ops.aten.select_scatter.default(full_default, copy, 1, 0);  full_default = copy = None
        sub = torch.ops.aten.sub.Tensor(select, view_1);  select = view_1 = None
        div = torch.ops.aten.div.Tensor(sub, select_2);  sub = select_2 = None
        select_5 = torch.ops.aten.select.int(arg1_1, 0, 0)
        expand_1 = torch.ops.aten.expand.default(arg0_1, [2048, 128])
        mul = torch.ops.aten.mul.Tensor(expand_1, 1);  expand_1 = None
        view_2 = torch.ops.aten.view.default(div, [2048, 1]);  div = None
        mul_1 = torch.ops.aten.mul.Tensor(view_2, select_5);  view_2 = select_5 = None
        mul_2 = torch.ops.aten.mul.Tensor(mul_1, -1.0);  mul_1 = None
        add = torch.ops.aten.add.Tensor(mul, mul_2);  mul = mul_2 = None
        select_7 = torch.ops.aten.select.int(arg1_1, 0, 1)
        select_8 = torch.ops.aten.select.int(select_7, 0, 1);  select_7 = None
        select_9 = torch.ops.aten.select.int(add, 1, 1)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_1 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_10 = torch.ops.aten.select.int(add, 1, 1)
        view_4 = torch.ops.aten.view.default(select_10, [8, 128, 2]);  select_10 = None
        baddbmm_1 = torch.ops.aten.baddbmm.default(unsqueeze_2, view_4, permute_1, alpha = -2.0);  unsqueeze_2 = view_4 = permute_1 = None
        argmin_1 = torch.ops.aten.argmin.default(baddbmm_1, -1);  baddbmm_1 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(argmin_1, -1)
        expand_2 = torch.ops.aten.expand.default(unsqueeze_3, [8, 128, 2]);  unsqueeze_3 = None
        gather_1 = torch.ops.aten.gather.default(arg3_1, 1, expand_2);  expand_2 = None
        view_5 = torch.ops.aten.view.default(gather_1, [-1]);  gather_1 = None
        select_12 = torch.ops.aten.select.int(select_scatter, 1, 1)
        copy_1 = torch.ops.aten.copy.default(select_12, argmin_1);  select_12 = argmin_1 = None
        select_scatter_1 = torch.ops.aten.select_scatter.default(select_scatter, copy_1, 1, 1);  select_scatter = copy_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(select_9, view_5);  select_9 = view_5 = None
        div_1 = torch.ops.aten.div.Tensor(sub_1, select_8);  sub_1 = select_8 = None
        select_14 = torch.ops.aten.select.int(arg1_1, 0, 1)
        slice_2 = torch.ops.aten.slice.Tensor(select_14, 0, 1, 9223372036854775807);  select_14 = None
        slice_3 = torch.ops.aten.slice.Tensor(add, 1, 1, 9223372036854775807)
        expand_3 = torch.ops.aten.expand.default(slice_3, [2048, 127]);  slice_3 = None
        mul_3 = torch.ops.aten.mul.Tensor(expand_3, 1);  expand_3 = None
        view_6 = torch.ops.aten.view.default(div_1, [2048, 1]);  div_1 = None
        mul_4 = torch.ops.aten.mul.Tensor(view_6, slice_2);  view_6 = slice_2 = None
        mul_5 = torch.ops.aten.mul.Tensor(mul_4, -1.0);  mul_4 = None
        add_1 = torch.ops.aten.add.Tensor(mul_3, mul_5);  mul_3 = mul_5 = None
        slice_scatter = torch.ops.aten.slice_scatter.default(add, add_1, 1, 1, 9223372036854775807);  add = add_1 = None
        select_16 = torch.ops.aten.select.int(arg1_1, 0, 2)
        select_17 = torch.ops.aten.select.int(select_16, 0, 2);  select_16 = None
        select_18 = torch.ops.aten.select.int(slice_scatter, 1, 2)
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_2 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_19 = torch.ops.aten.select.int(slice_scatter, 1, 2)
        view_8 = torch.ops.aten.view.default(select_19, [8, 128, 2]);  select_19 = None
        baddbmm_2 = torch.ops.aten.baddbmm.default(unsqueeze_4, view_8, permute_2, alpha = -2.0);  unsqueeze_4 = view_8 = permute_2 = None
        argmin_2 = torch.ops.aten.argmin.default(baddbmm_2, -1);  baddbmm_2 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(argmin_2, -1)
        expand_4 = torch.ops.aten.expand.default(unsqueeze_5, [8, 128, 2]);  unsqueeze_5 = None
        gather_2 = torch.ops.aten.gather.default(arg3_1, 1, expand_4);  expand_4 = None
        view_9 = torch.ops.aten.view.default(gather_2, [-1]);  gather_2 = None
        select_21 = torch.ops.aten.select.int(select_scatter_1, 1, 2)
        copy_2 = torch.ops.aten.copy.default(select_21, argmin_2);  select_21 = argmin_2 = None
        select_scatter_2 = torch.ops.aten.select_scatter.default(select_scatter_1, copy_2, 1, 2);  select_scatter_1 = copy_2 = None
        sub_2 = torch.ops.aten.sub.Tensor(select_18, view_9);  select_18 = view_9 = None
        div_2 = torch.ops.aten.div.Tensor(sub_2, select_17);  sub_2 = select_17 = None
        select_23 = torch.ops.aten.select.int(arg1_1, 0, 2)
        slice_6 = torch.ops.aten.slice.Tensor(select_23, 0, 2, 9223372036854775807);  select_23 = None
        slice_7 = torch.ops.aten.slice.Tensor(slice_scatter, 1, 2, 9223372036854775807)
        expand_5 = torch.ops.aten.expand.default(slice_7, [2048, 126]);  slice_7 = None
        mul_6 = torch.ops.aten.mul.Tensor(expand_5, 1);  expand_5 = None
        view_10 = torch.ops.aten.view.default(div_2, [2048, 1]);  div_2 = None
        mul_7 = torch.ops.aten.mul.Tensor(view_10, slice_6);  view_10 = slice_6 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, -1.0);  mul_7 = None
        add_2 = torch.ops.aten.add.Tensor(mul_6, mul_8);  mul_6 = mul_8 = None
        slice_scatter_1 = torch.ops.aten.slice_scatter.default(slice_scatter, add_2, 1, 2, 9223372036854775807);  slice_scatter = add_2 = None
        select_25 = torch.ops.aten.select.int(arg1_1, 0, 3)
        select_26 = torch.ops.aten.select.int(select_25, 0, 3);  select_25 = None
        select_27 = torch.ops.aten.select.int(slice_scatter_1, 1, 3)
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_3 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_28 = torch.ops.aten.select.int(slice_scatter_1, 1, 3)
        view_12 = torch.ops.aten.view.default(select_28, [8, 128, 2]);  select_28 = None
        baddbmm_3 = torch.ops.aten.baddbmm.default(unsqueeze_6, view_12, permute_3, alpha = -2.0);  unsqueeze_6 = view_12 = permute_3 = None
        argmin_3 = torch.ops.aten.argmin.default(baddbmm_3, -1);  baddbmm_3 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(argmin_3, -1)
        expand_6 = torch.ops.aten.expand.default(unsqueeze_7, [8, 128, 2]);  unsqueeze_7 = None
        gather_3 = torch.ops.aten.gather.default(arg3_1, 1, expand_6);  expand_6 = None
        view_13 = torch.ops.aten.view.default(gather_3, [-1]);  gather_3 = None
        select_30 = torch.ops.aten.select.int(select_scatter_2, 1, 3)
        copy_3 = torch.ops.aten.copy.default(select_30, argmin_3);  select_30 = argmin_3 = None
        select_scatter_3 = torch.ops.aten.select_scatter.default(select_scatter_2, copy_3, 1, 3);  select_scatter_2 = copy_3 = None
        sub_3 = torch.ops.aten.sub.Tensor(select_27, view_13);  select_27 = view_13 = None
        div_3 = torch.ops.aten.div.Tensor(sub_3, select_26);  sub_3 = select_26 = None
        select_32 = torch.ops.aten.select.int(arg1_1, 0, 3)
        slice_10 = torch.ops.aten.slice.Tensor(select_32, 0, 3, 9223372036854775807);  select_32 = None
        slice_11 = torch.ops.aten.slice.Tensor(slice_scatter_1, 1, 3, 9223372036854775807)
        expand_7 = torch.ops.aten.expand.default(slice_11, [2048, 125]);  slice_11 = None
        mul_9 = torch.ops.aten.mul.Tensor(expand_7, 1);  expand_7 = None
        view_14 = torch.ops.aten.view.default(div_3, [2048, 1]);  div_3 = None
        mul_10 = torch.ops.aten.mul.Tensor(view_14, slice_10);  view_14 = slice_10 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, -1.0);  mul_10 = None
        add_3 = torch.ops.aten.add.Tensor(mul_9, mul_11);  mul_9 = mul_11 = None
        slice_scatter_2 = torch.ops.aten.slice_scatter.default(slice_scatter_1, add_3, 1, 3, 9223372036854775807);  slice_scatter_1 = add_3 = None
        select_34 = torch.ops.aten.select.int(arg1_1, 0, 4)
        select_35 = torch.ops.aten.select.int(select_34, 0, 4);  select_34 = None
        select_36 = torch.ops.aten.select.int(slice_scatter_2, 1, 4)
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_4 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_37 = torch.ops.aten.select.int(slice_scatter_2, 1, 4)
        view_16 = torch.ops.aten.view.default(select_37, [8, 128, 2]);  select_37 = None
        baddbmm_4 = torch.ops.aten.baddbmm.default(unsqueeze_8, view_16, permute_4, alpha = -2.0);  unsqueeze_8 = view_16 = permute_4 = None
        argmin_4 = torch.ops.aten.argmin.default(baddbmm_4, -1);  baddbmm_4 = None
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(argmin_4, -1)
        expand_8 = torch.ops.aten.expand.default(unsqueeze_9, [8, 128, 2]);  unsqueeze_9 = None
        gather_4 = torch.ops.aten.gather.default(arg3_1, 1, expand_8);  expand_8 = None
        view_17 = torch.ops.aten.view.default(gather_4, [-1]);  gather_4 = None
        select_39 = torch.ops.aten.select.int(select_scatter_3, 1, 4)
        copy_4 = torch.ops.aten.copy.default(select_39, argmin_4);  select_39 = argmin_4 = None
        select_scatter_4 = torch.ops.aten.select_scatter.default(select_scatter_3, copy_4, 1, 4);  select_scatter_3 = copy_4 = None
        sub_4 = torch.ops.aten.sub.Tensor(select_36, view_17);  select_36 = view_17 = None
        div_4 = torch.ops.aten.div.Tensor(sub_4, select_35);  sub_4 = select_35 = None
        select_41 = torch.ops.aten.select.int(arg1_1, 0, 4)
        slice_14 = torch.ops.aten.slice.Tensor(select_41, 0, 4, 9223372036854775807);  select_41 = None
        slice_15 = torch.ops.aten.slice.Tensor(slice_scatter_2, 1, 4, 9223372036854775807)
        expand_9 = torch.ops.aten.expand.default(slice_15, [2048, 124]);  slice_15 = None
        mul_12 = torch.ops.aten.mul.Tensor(expand_9, 1);  expand_9 = None
        view_18 = torch.ops.aten.view.default(div_4, [2048, 1]);  div_4 = None
        mul_13 = torch.ops.aten.mul.Tensor(view_18, slice_14);  view_18 = slice_14 = None
        mul_14 = torch.ops.aten.mul.Tensor(mul_13, -1.0);  mul_13 = None
        add_4 = torch.ops.aten.add.Tensor(mul_12, mul_14);  mul_12 = mul_14 = None
        slice_scatter_3 = torch.ops.aten.slice_scatter.default(slice_scatter_2, add_4, 1, 4, 9223372036854775807);  slice_scatter_2 = add_4 = None
        select_43 = torch.ops.aten.select.int(arg1_1, 0, 5)
        select_44 = torch.ops.aten.select.int(select_43, 0, 5);  select_43 = None
        select_45 = torch.ops.aten.select.int(slice_scatter_3, 1, 5)
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_5 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_46 = torch.ops.aten.select.int(slice_scatter_3, 1, 5)
        view_20 = torch.ops.aten.view.default(select_46, [8, 128, 2]);  select_46 = None
        baddbmm_5 = torch.ops.aten.baddbmm.default(unsqueeze_10, view_20, permute_5, alpha = -2.0);  unsqueeze_10 = view_20 = permute_5 = None
        argmin_5 = torch.ops.aten.argmin.default(baddbmm_5, -1);  baddbmm_5 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(argmin_5, -1)
        expand_10 = torch.ops.aten.expand.default(unsqueeze_11, [8, 128, 2]);  unsqueeze_11 = None
        gather_5 = torch.ops.aten.gather.default(arg3_1, 1, expand_10);  expand_10 = None
        view_21 = torch.ops.aten.view.default(gather_5, [-1]);  gather_5 = None
        select_48 = torch.ops.aten.select.int(select_scatter_4, 1, 5)
        copy_5 = torch.ops.aten.copy.default(select_48, argmin_5);  select_48 = argmin_5 = None
        select_scatter_5 = torch.ops.aten.select_scatter.default(select_scatter_4, copy_5, 1, 5);  select_scatter_4 = copy_5 = None
        sub_5 = torch.ops.aten.sub.Tensor(select_45, view_21);  select_45 = view_21 = None
        div_5 = torch.ops.aten.div.Tensor(sub_5, select_44);  sub_5 = select_44 = None
        select_50 = torch.ops.aten.select.int(arg1_1, 0, 5)
        slice_18 = torch.ops.aten.slice.Tensor(select_50, 0, 5, 9223372036854775807);  select_50 = None
        slice_19 = torch.ops.aten.slice.Tensor(slice_scatter_3, 1, 5, 9223372036854775807)
        expand_11 = torch.ops.aten.expand.default(slice_19, [2048, 123]);  slice_19 = None
        mul_15 = torch.ops.aten.mul.Tensor(expand_11, 1);  expand_11 = None
        view_22 = torch.ops.aten.view.default(div_5, [2048, 1]);  div_5 = None
        mul_16 = torch.ops.aten.mul.Tensor(view_22, slice_18);  view_22 = slice_18 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, -1.0);  mul_16 = None
        add_5 = torch.ops.aten.add.Tensor(mul_15, mul_17);  mul_15 = mul_17 = None
        slice_scatter_4 = torch.ops.aten.slice_scatter.default(slice_scatter_3, add_5, 1, 5, 9223372036854775807);  slice_scatter_3 = add_5 = None
        select_52 = torch.ops.aten.select.int(arg1_1, 0, 6)
        select_53 = torch.ops.aten.select.int(select_52, 0, 6);  select_52 = None
        select_54 = torch.ops.aten.select.int(slice_scatter_4, 1, 6)
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_6 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_55 = torch.ops.aten.select.int(slice_scatter_4, 1, 6)
        view_24 = torch.ops.aten.view.default(select_55, [8, 128, 2]);  select_55 = None
        baddbmm_6 = torch.ops.aten.baddbmm.default(unsqueeze_12, view_24, permute_6, alpha = -2.0);  unsqueeze_12 = view_24 = permute_6 = None
        argmin_6 = torch.ops.aten.argmin.default(baddbmm_6, -1);  baddbmm_6 = None
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(argmin_6, -1)
        expand_12 = torch.ops.aten.expand.default(unsqueeze_13, [8, 128, 2]);  unsqueeze_13 = None
        gather_6 = torch.ops.aten.gather.default(arg3_1, 1, expand_12);  expand_12 = None
        view_25 = torch.ops.aten.view.default(gather_6, [-1]);  gather_6 = None
        select_57 = torch.ops.aten.select.int(select_scatter_5, 1, 6)
        copy_6 = torch.ops.aten.copy.default(select_57, argmin_6);  select_57 = argmin_6 = None
        select_scatter_6 = torch.ops.aten.select_scatter.default(select_scatter_5, copy_6, 1, 6);  select_scatter_5 = copy_6 = None
        sub_6 = torch.ops.aten.sub.Tensor(select_54, view_25);  select_54 = view_25 = None
        div_6 = torch.ops.aten.div.Tensor(sub_6, select_53);  sub_6 = select_53 = None
        select_59 = torch.ops.aten.select.int(arg1_1, 0, 6)
        slice_22 = torch.ops.aten.slice.Tensor(select_59, 0, 6, 9223372036854775807);  select_59 = None
        slice_23 = torch.ops.aten.slice.Tensor(slice_scatter_4, 1, 6, 9223372036854775807)
        expand_13 = torch.ops.aten.expand.default(slice_23, [2048, 122]);  slice_23 = None
        mul_18 = torch.ops.aten.mul.Tensor(expand_13, 1);  expand_13 = None
        view_26 = torch.ops.aten.view.default(div_6, [2048, 1]);  div_6 = None
        mul_19 = torch.ops.aten.mul.Tensor(view_26, slice_22);  view_26 = slice_22 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_19, -1.0);  mul_19 = None
        add_6 = torch.ops.aten.add.Tensor(mul_18, mul_20);  mul_18 = mul_20 = None
        slice_scatter_5 = torch.ops.aten.slice_scatter.default(slice_scatter_4, add_6, 1, 6, 9223372036854775807);  slice_scatter_4 = add_6 = None
        select_61 = torch.ops.aten.select.int(arg1_1, 0, 7)
        select_62 = torch.ops.aten.select.int(select_61, 0, 7);  select_61 = None
        select_63 = torch.ops.aten.select.int(slice_scatter_5, 1, 7)
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_7 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_64 = torch.ops.aten.select.int(slice_scatter_5, 1, 7)
        view_28 = torch.ops.aten.view.default(select_64, [8, 128, 2]);  select_64 = None
        baddbmm_7 = torch.ops.aten.baddbmm.default(unsqueeze_14, view_28, permute_7, alpha = -2.0);  unsqueeze_14 = view_28 = permute_7 = None
        argmin_7 = torch.ops.aten.argmin.default(baddbmm_7, -1);  baddbmm_7 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(argmin_7, -1)
        expand_14 = torch.ops.aten.expand.default(unsqueeze_15, [8, 128, 2]);  unsqueeze_15 = None
        gather_7 = torch.ops.aten.gather.default(arg3_1, 1, expand_14);  expand_14 = None
        view_29 = torch.ops.aten.view.default(gather_7, [-1]);  gather_7 = None
        select_66 = torch.ops.aten.select.int(select_scatter_6, 1, 7)
        copy_7 = torch.ops.aten.copy.default(select_66, argmin_7);  select_66 = argmin_7 = None
        select_scatter_7 = torch.ops.aten.select_scatter.default(select_scatter_6, copy_7, 1, 7);  select_scatter_6 = copy_7 = None
        sub_7 = torch.ops.aten.sub.Tensor(select_63, view_29);  select_63 = view_29 = None
        div_7 = torch.ops.aten.div.Tensor(sub_7, select_62);  sub_7 = select_62 = None
        select_68 = torch.ops.aten.select.int(arg1_1, 0, 7)
        slice_26 = torch.ops.aten.slice.Tensor(select_68, 0, 7, 9223372036854775807);  select_68 = None
        slice_27 = torch.ops.aten.slice.Tensor(slice_scatter_5, 1, 7, 9223372036854775807)
        expand_15 = torch.ops.aten.expand.default(slice_27, [2048, 121]);  slice_27 = None
        mul_21 = torch.ops.aten.mul.Tensor(expand_15, 1);  expand_15 = None
        view_30 = torch.ops.aten.view.default(div_7, [2048, 1]);  div_7 = None
        mul_22 = torch.ops.aten.mul.Tensor(view_30, slice_26);  view_30 = slice_26 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_22, -1.0);  mul_22 = None
        add_7 = torch.ops.aten.add.Tensor(mul_21, mul_23);  mul_21 = mul_23 = None
        slice_scatter_6 = torch.ops.aten.slice_scatter.default(slice_scatter_5, add_7, 1, 7, 9223372036854775807);  slice_scatter_5 = add_7 = None
        select_70 = torch.ops.aten.select.int(arg1_1, 0, 8)
        select_71 = torch.ops.aten.select.int(select_70, 0, 8);  select_70 = None
        select_72 = torch.ops.aten.select.int(slice_scatter_6, 1, 8)
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_8 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_73 = torch.ops.aten.select.int(slice_scatter_6, 1, 8)
        view_32 = torch.ops.aten.view.default(select_73, [8, 128, 2]);  select_73 = None
        baddbmm_8 = torch.ops.aten.baddbmm.default(unsqueeze_16, view_32, permute_8, alpha = -2.0);  unsqueeze_16 = view_32 = permute_8 = None
        argmin_8 = torch.ops.aten.argmin.default(baddbmm_8, -1);  baddbmm_8 = None
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(argmin_8, -1)
        expand_16 = torch.ops.aten.expand.default(unsqueeze_17, [8, 128, 2]);  unsqueeze_17 = None
        gather_8 = torch.ops.aten.gather.default(arg3_1, 1, expand_16);  expand_16 = None
        view_33 = torch.ops.aten.view.default(gather_8, [-1]);  gather_8 = None
        select_75 = torch.ops.aten.select.int(select_scatter_7, 1, 8)
        copy_8 = torch.ops.aten.copy.default(select_75, argmin_8);  select_75 = argmin_8 = None
        select_scatter_8 = torch.ops.aten.select_scatter.default(select_scatter_7, copy_8, 1, 8);  select_scatter_7 = copy_8 = None
        sub_8 = torch.ops.aten.sub.Tensor(select_72, view_33);  select_72 = view_33 = None
        div_8 = torch.ops.aten.div.Tensor(sub_8, select_71);  sub_8 = select_71 = None
        select_77 = torch.ops.aten.select.int(arg1_1, 0, 8)
        slice_30 = torch.ops.aten.slice.Tensor(select_77, 0, 8, 9223372036854775807);  select_77 = None
        slice_31 = torch.ops.aten.slice.Tensor(slice_scatter_6, 1, 8, 9223372036854775807)
        expand_17 = torch.ops.aten.expand.default(slice_31, [2048, 120]);  slice_31 = None
        mul_24 = torch.ops.aten.mul.Tensor(expand_17, 1);  expand_17 = None
        view_34 = torch.ops.aten.view.default(div_8, [2048, 1]);  div_8 = None
        mul_25 = torch.ops.aten.mul.Tensor(view_34, slice_30);  view_34 = slice_30 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_25, -1.0);  mul_25 = None
        add_8 = torch.ops.aten.add.Tensor(mul_24, mul_26);  mul_24 = mul_26 = None
        slice_scatter_7 = torch.ops.aten.slice_scatter.default(slice_scatter_6, add_8, 1, 8, 9223372036854775807);  slice_scatter_6 = add_8 = None
        select_79 = torch.ops.aten.select.int(arg1_1, 0, 9)
        select_80 = torch.ops.aten.select.int(select_79, 0, 9);  select_79 = None
        select_81 = torch.ops.aten.select.int(slice_scatter_7, 1, 9)
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_9 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_82 = torch.ops.aten.select.int(slice_scatter_7, 1, 9)
        view_36 = torch.ops.aten.view.default(select_82, [8, 128, 2]);  select_82 = None
        baddbmm_9 = torch.ops.aten.baddbmm.default(unsqueeze_18, view_36, permute_9, alpha = -2.0);  unsqueeze_18 = view_36 = permute_9 = None
        argmin_9 = torch.ops.aten.argmin.default(baddbmm_9, -1);  baddbmm_9 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(argmin_9, -1)
        expand_18 = torch.ops.aten.expand.default(unsqueeze_19, [8, 128, 2]);  unsqueeze_19 = None
        gather_9 = torch.ops.aten.gather.default(arg3_1, 1, expand_18);  expand_18 = None
        view_37 = torch.ops.aten.view.default(gather_9, [-1]);  gather_9 = None
        select_84 = torch.ops.aten.select.int(select_scatter_8, 1, 9)
        copy_9 = torch.ops.aten.copy.default(select_84, argmin_9);  select_84 = argmin_9 = None
        select_scatter_9 = torch.ops.aten.select_scatter.default(select_scatter_8, copy_9, 1, 9);  select_scatter_8 = copy_9 = None
        sub_9 = torch.ops.aten.sub.Tensor(select_81, view_37);  select_81 = view_37 = None
        div_9 = torch.ops.aten.div.Tensor(sub_9, select_80);  sub_9 = select_80 = None
        select_86 = torch.ops.aten.select.int(arg1_1, 0, 9)
        slice_34 = torch.ops.aten.slice.Tensor(select_86, 0, 9, 9223372036854775807);  select_86 = None
        slice_35 = torch.ops.aten.slice.Tensor(slice_scatter_7, 1, 9, 9223372036854775807)
        expand_19 = torch.ops.aten.expand.default(slice_35, [2048, 119]);  slice_35 = None
        mul_27 = torch.ops.aten.mul.Tensor(expand_19, 1);  expand_19 = None
        view_38 = torch.ops.aten.view.default(div_9, [2048, 1]);  div_9 = None
        mul_28 = torch.ops.aten.mul.Tensor(view_38, slice_34);  view_38 = slice_34 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, -1.0);  mul_28 = None
        add_9 = torch.ops.aten.add.Tensor(mul_27, mul_29);  mul_27 = mul_29 = None
        slice_scatter_8 = torch.ops.aten.slice_scatter.default(slice_scatter_7, add_9, 1, 9, 9223372036854775807);  slice_scatter_7 = add_9 = None
        select_88 = torch.ops.aten.select.int(arg1_1, 0, 10)
        select_89 = torch.ops.aten.select.int(select_88, 0, 10);  select_88 = None
        select_90 = torch.ops.aten.select.int(slice_scatter_8, 1, 10)
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_10 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_91 = torch.ops.aten.select.int(slice_scatter_8, 1, 10)
        view_40 = torch.ops.aten.view.default(select_91, [8, 128, 2]);  select_91 = None
        baddbmm_10 = torch.ops.aten.baddbmm.default(unsqueeze_20, view_40, permute_10, alpha = -2.0);  unsqueeze_20 = view_40 = permute_10 = None
        argmin_10 = torch.ops.aten.argmin.default(baddbmm_10, -1);  baddbmm_10 = None
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(argmin_10, -1)
        expand_20 = torch.ops.aten.expand.default(unsqueeze_21, [8, 128, 2]);  unsqueeze_21 = None
        gather_10 = torch.ops.aten.gather.default(arg3_1, 1, expand_20);  expand_20 = None
        view_41 = torch.ops.aten.view.default(gather_10, [-1]);  gather_10 = None
        select_93 = torch.ops.aten.select.int(select_scatter_9, 1, 10)
        copy_10 = torch.ops.aten.copy.default(select_93, argmin_10);  select_93 = argmin_10 = None
        select_scatter_10 = torch.ops.aten.select_scatter.default(select_scatter_9, copy_10, 1, 10);  select_scatter_9 = copy_10 = None
        sub_10 = torch.ops.aten.sub.Tensor(select_90, view_41);  select_90 = view_41 = None
        div_10 = torch.ops.aten.div.Tensor(sub_10, select_89);  sub_10 = select_89 = None
        select_95 = torch.ops.aten.select.int(arg1_1, 0, 10)
        slice_38 = torch.ops.aten.slice.Tensor(select_95, 0, 10, 9223372036854775807);  select_95 = None
        slice_39 = torch.ops.aten.slice.Tensor(slice_scatter_8, 1, 10, 9223372036854775807)
        expand_21 = torch.ops.aten.expand.default(slice_39, [2048, 118]);  slice_39 = None
        mul_30 = torch.ops.aten.mul.Tensor(expand_21, 1);  expand_21 = None
        view_42 = torch.ops.aten.view.default(div_10, [2048, 1]);  div_10 = None
        mul_31 = torch.ops.aten.mul.Tensor(view_42, slice_38);  view_42 = slice_38 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_31, -1.0);  mul_31 = None
        add_10 = torch.ops.aten.add.Tensor(mul_30, mul_32);  mul_30 = mul_32 = None
        slice_scatter_9 = torch.ops.aten.slice_scatter.default(slice_scatter_8, add_10, 1, 10, 9223372036854775807);  slice_scatter_8 = add_10 = None
        select_97 = torch.ops.aten.select.int(arg1_1, 0, 11)
        select_98 = torch.ops.aten.select.int(select_97, 0, 11);  select_97 = None
        select_99 = torch.ops.aten.select.int(slice_scatter_9, 1, 11)
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_11 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_100 = torch.ops.aten.select.int(slice_scatter_9, 1, 11)
        view_44 = torch.ops.aten.view.default(select_100, [8, 128, 2]);  select_100 = None
        baddbmm_11 = torch.ops.aten.baddbmm.default(unsqueeze_22, view_44, permute_11, alpha = -2.0);  unsqueeze_22 = view_44 = permute_11 = None
        argmin_11 = torch.ops.aten.argmin.default(baddbmm_11, -1);  baddbmm_11 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(argmin_11, -1)
        expand_22 = torch.ops.aten.expand.default(unsqueeze_23, [8, 128, 2]);  unsqueeze_23 = None
        gather_11 = torch.ops.aten.gather.default(arg3_1, 1, expand_22);  expand_22 = None
        view_45 = torch.ops.aten.view.default(gather_11, [-1]);  gather_11 = None
        select_102 = torch.ops.aten.select.int(select_scatter_10, 1, 11)
        copy_11 = torch.ops.aten.copy.default(select_102, argmin_11);  select_102 = argmin_11 = None
        select_scatter_11 = torch.ops.aten.select_scatter.default(select_scatter_10, copy_11, 1, 11);  select_scatter_10 = copy_11 = None
        sub_11 = torch.ops.aten.sub.Tensor(select_99, view_45);  select_99 = view_45 = None
        div_11 = torch.ops.aten.div.Tensor(sub_11, select_98);  sub_11 = select_98 = None
        select_104 = torch.ops.aten.select.int(arg1_1, 0, 11)
        slice_42 = torch.ops.aten.slice.Tensor(select_104, 0, 11, 9223372036854775807);  select_104 = None
        slice_43 = torch.ops.aten.slice.Tensor(slice_scatter_9, 1, 11, 9223372036854775807)
        expand_23 = torch.ops.aten.expand.default(slice_43, [2048, 117]);  slice_43 = None
        mul_33 = torch.ops.aten.mul.Tensor(expand_23, 1);  expand_23 = None
        view_46 = torch.ops.aten.view.default(div_11, [2048, 1]);  div_11 = None
        mul_34 = torch.ops.aten.mul.Tensor(view_46, slice_42);  view_46 = slice_42 = None
        mul_35 = torch.ops.aten.mul.Tensor(mul_34, -1.0);  mul_34 = None
        add_11 = torch.ops.aten.add.Tensor(mul_33, mul_35);  mul_33 = mul_35 = None
        slice_scatter_10 = torch.ops.aten.slice_scatter.default(slice_scatter_9, add_11, 1, 11, 9223372036854775807);  slice_scatter_9 = add_11 = None
        select_106 = torch.ops.aten.select.int(arg1_1, 0, 12)
        select_107 = torch.ops.aten.select.int(select_106, 0, 12);  select_106 = None
        select_108 = torch.ops.aten.select.int(slice_scatter_10, 1, 12)
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_12 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_109 = torch.ops.aten.select.int(slice_scatter_10, 1, 12)
        view_48 = torch.ops.aten.view.default(select_109, [8, 128, 2]);  select_109 = None
        baddbmm_12 = torch.ops.aten.baddbmm.default(unsqueeze_24, view_48, permute_12, alpha = -2.0);  unsqueeze_24 = view_48 = permute_12 = None
        argmin_12 = torch.ops.aten.argmin.default(baddbmm_12, -1);  baddbmm_12 = None
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(argmin_12, -1)
        expand_24 = torch.ops.aten.expand.default(unsqueeze_25, [8, 128, 2]);  unsqueeze_25 = None
        gather_12 = torch.ops.aten.gather.default(arg3_1, 1, expand_24);  expand_24 = None
        view_49 = torch.ops.aten.view.default(gather_12, [-1]);  gather_12 = None
        select_111 = torch.ops.aten.select.int(select_scatter_11, 1, 12)
        copy_12 = torch.ops.aten.copy.default(select_111, argmin_12);  select_111 = argmin_12 = None
        select_scatter_12 = torch.ops.aten.select_scatter.default(select_scatter_11, copy_12, 1, 12);  select_scatter_11 = copy_12 = None
        sub_12 = torch.ops.aten.sub.Tensor(select_108, view_49);  select_108 = view_49 = None
        div_12 = torch.ops.aten.div.Tensor(sub_12, select_107);  sub_12 = select_107 = None
        select_113 = torch.ops.aten.select.int(arg1_1, 0, 12)
        slice_46 = torch.ops.aten.slice.Tensor(select_113, 0, 12, 9223372036854775807);  select_113 = None
        slice_47 = torch.ops.aten.slice.Tensor(slice_scatter_10, 1, 12, 9223372036854775807)
        expand_25 = torch.ops.aten.expand.default(slice_47, [2048, 116]);  slice_47 = None
        mul_36 = torch.ops.aten.mul.Tensor(expand_25, 1);  expand_25 = None
        view_50 = torch.ops.aten.view.default(div_12, [2048, 1]);  div_12 = None
        mul_37 = torch.ops.aten.mul.Tensor(view_50, slice_46);  view_50 = slice_46 = None
        mul_38 = torch.ops.aten.mul.Tensor(mul_37, -1.0);  mul_37 = None
        add_12 = torch.ops.aten.add.Tensor(mul_36, mul_38);  mul_36 = mul_38 = None
        slice_scatter_11 = torch.ops.aten.slice_scatter.default(slice_scatter_10, add_12, 1, 12, 9223372036854775807);  slice_scatter_10 = add_12 = None
        select_115 = torch.ops.aten.select.int(arg1_1, 0, 13)
        select_116 = torch.ops.aten.select.int(select_115, 0, 13);  select_115 = None
        select_117 = torch.ops.aten.select.int(slice_scatter_11, 1, 13)
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_13 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_118 = torch.ops.aten.select.int(slice_scatter_11, 1, 13)
        view_52 = torch.ops.aten.view.default(select_118, [8, 128, 2]);  select_118 = None
        baddbmm_13 = torch.ops.aten.baddbmm.default(unsqueeze_26, view_52, permute_13, alpha = -2.0);  unsqueeze_26 = view_52 = permute_13 = None
        argmin_13 = torch.ops.aten.argmin.default(baddbmm_13, -1);  baddbmm_13 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(argmin_13, -1)
        expand_26 = torch.ops.aten.expand.default(unsqueeze_27, [8, 128, 2]);  unsqueeze_27 = None
        gather_13 = torch.ops.aten.gather.default(arg3_1, 1, expand_26);  expand_26 = None
        view_53 = torch.ops.aten.view.default(gather_13, [-1]);  gather_13 = None
        select_120 = torch.ops.aten.select.int(select_scatter_12, 1, 13)
        copy_13 = torch.ops.aten.copy.default(select_120, argmin_13);  select_120 = argmin_13 = None
        select_scatter_13 = torch.ops.aten.select_scatter.default(select_scatter_12, copy_13, 1, 13);  select_scatter_12 = copy_13 = None
        sub_13 = torch.ops.aten.sub.Tensor(select_117, view_53);  select_117 = view_53 = None
        div_13 = torch.ops.aten.div.Tensor(sub_13, select_116);  sub_13 = select_116 = None
        select_122 = torch.ops.aten.select.int(arg1_1, 0, 13)
        slice_50 = torch.ops.aten.slice.Tensor(select_122, 0, 13, 9223372036854775807);  select_122 = None
        slice_51 = torch.ops.aten.slice.Tensor(slice_scatter_11, 1, 13, 9223372036854775807)
        expand_27 = torch.ops.aten.expand.default(slice_51, [2048, 115]);  slice_51 = None
        mul_39 = torch.ops.aten.mul.Tensor(expand_27, 1);  expand_27 = None
        view_54 = torch.ops.aten.view.default(div_13, [2048, 1]);  div_13 = None
        mul_40 = torch.ops.aten.mul.Tensor(view_54, slice_50);  view_54 = slice_50 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, -1.0);  mul_40 = None
        add_13 = torch.ops.aten.add.Tensor(mul_39, mul_41);  mul_39 = mul_41 = None
        slice_scatter_12 = torch.ops.aten.slice_scatter.default(slice_scatter_11, add_13, 1, 13, 9223372036854775807);  slice_scatter_11 = add_13 = None
        select_124 = torch.ops.aten.select.int(arg1_1, 0, 14)
        select_125 = torch.ops.aten.select.int(select_124, 0, 14);  select_124 = None
        select_126 = torch.ops.aten.select.int(slice_scatter_12, 1, 14)
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_14 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_127 = torch.ops.aten.select.int(slice_scatter_12, 1, 14)
        view_56 = torch.ops.aten.view.default(select_127, [8, 128, 2]);  select_127 = None
        baddbmm_14 = torch.ops.aten.baddbmm.default(unsqueeze_28, view_56, permute_14, alpha = -2.0);  unsqueeze_28 = view_56 = permute_14 = None
        argmin_14 = torch.ops.aten.argmin.default(baddbmm_14, -1);  baddbmm_14 = None
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(argmin_14, -1)
        expand_28 = torch.ops.aten.expand.default(unsqueeze_29, [8, 128, 2]);  unsqueeze_29 = None
        gather_14 = torch.ops.aten.gather.default(arg3_1, 1, expand_28);  expand_28 = None
        view_57 = torch.ops.aten.view.default(gather_14, [-1]);  gather_14 = None
        select_129 = torch.ops.aten.select.int(select_scatter_13, 1, 14)
        copy_14 = torch.ops.aten.copy.default(select_129, argmin_14);  select_129 = argmin_14 = None
        select_scatter_14 = torch.ops.aten.select_scatter.default(select_scatter_13, copy_14, 1, 14);  select_scatter_13 = copy_14 = None
        sub_14 = torch.ops.aten.sub.Tensor(select_126, view_57);  select_126 = view_57 = None
        div_14 = torch.ops.aten.div.Tensor(sub_14, select_125);  sub_14 = select_125 = None
        select_131 = torch.ops.aten.select.int(arg1_1, 0, 14)
        slice_54 = torch.ops.aten.slice.Tensor(select_131, 0, 14, 9223372036854775807);  select_131 = None
        slice_55 = torch.ops.aten.slice.Tensor(slice_scatter_12, 1, 14, 9223372036854775807)
        expand_29 = torch.ops.aten.expand.default(slice_55, [2048, 114]);  slice_55 = None
        mul_42 = torch.ops.aten.mul.Tensor(expand_29, 1);  expand_29 = None
        view_58 = torch.ops.aten.view.default(div_14, [2048, 1]);  div_14 = None
        mul_43 = torch.ops.aten.mul.Tensor(view_58, slice_54);  view_58 = slice_54 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_43, -1.0);  mul_43 = None
        add_14 = torch.ops.aten.add.Tensor(mul_42, mul_44);  mul_42 = mul_44 = None
        slice_scatter_13 = torch.ops.aten.slice_scatter.default(slice_scatter_12, add_14, 1, 14, 9223372036854775807);  slice_scatter_12 = add_14 = None
        select_133 = torch.ops.aten.select.int(arg1_1, 0, 15)
        select_134 = torch.ops.aten.select.int(select_133, 0, 15);  select_133 = None
        select_135 = torch.ops.aten.select.int(slice_scatter_13, 1, 15)
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_15 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_136 = torch.ops.aten.select.int(slice_scatter_13, 1, 15)
        view_60 = torch.ops.aten.view.default(select_136, [8, 128, 2]);  select_136 = None
        baddbmm_15 = torch.ops.aten.baddbmm.default(unsqueeze_30, view_60, permute_15, alpha = -2.0);  unsqueeze_30 = view_60 = permute_15 = None
        argmin_15 = torch.ops.aten.argmin.default(baddbmm_15, -1);  baddbmm_15 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(argmin_15, -1)
        expand_30 = torch.ops.aten.expand.default(unsqueeze_31, [8, 128, 2]);  unsqueeze_31 = None
        gather_15 = torch.ops.aten.gather.default(arg3_1, 1, expand_30);  expand_30 = None
        view_61 = torch.ops.aten.view.default(gather_15, [-1]);  gather_15 = None
        select_138 = torch.ops.aten.select.int(select_scatter_14, 1, 15)
        copy_15 = torch.ops.aten.copy.default(select_138, argmin_15);  select_138 = argmin_15 = None
        select_scatter_15 = torch.ops.aten.select_scatter.default(select_scatter_14, copy_15, 1, 15);  select_scatter_14 = copy_15 = None
        sub_15 = torch.ops.aten.sub.Tensor(select_135, view_61);  select_135 = view_61 = None
        div_15 = torch.ops.aten.div.Tensor(sub_15, select_134);  sub_15 = select_134 = None
        select_140 = torch.ops.aten.select.int(arg1_1, 0, 15)
        slice_58 = torch.ops.aten.slice.Tensor(select_140, 0, 15, 9223372036854775807);  select_140 = None
        slice_59 = torch.ops.aten.slice.Tensor(slice_scatter_13, 1, 15, 9223372036854775807)
        expand_31 = torch.ops.aten.expand.default(slice_59, [2048, 113]);  slice_59 = None
        mul_45 = torch.ops.aten.mul.Tensor(expand_31, 1);  expand_31 = None
        view_62 = torch.ops.aten.view.default(div_15, [2048, 1]);  div_15 = None
        mul_46 = torch.ops.aten.mul.Tensor(view_62, slice_58);  view_62 = slice_58 = None
        mul_47 = torch.ops.aten.mul.Tensor(mul_46, -1.0);  mul_46 = None
        add_15 = torch.ops.aten.add.Tensor(mul_45, mul_47);  mul_45 = mul_47 = None
        slice_scatter_14 = torch.ops.aten.slice_scatter.default(slice_scatter_13, add_15, 1, 15, 9223372036854775807);  slice_scatter_13 = add_15 = None
        select_142 = torch.ops.aten.select.int(arg1_1, 0, 16)
        select_143 = torch.ops.aten.select.int(select_142, 0, 16);  select_142 = None
        select_144 = torch.ops.aten.select.int(slice_scatter_14, 1, 16)
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_16 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_145 = torch.ops.aten.select.int(slice_scatter_14, 1, 16)
        view_64 = torch.ops.aten.view.default(select_145, [8, 128, 2]);  select_145 = None
        baddbmm_16 = torch.ops.aten.baddbmm.default(unsqueeze_32, view_64, permute_16, alpha = -2.0);  unsqueeze_32 = view_64 = permute_16 = None
        argmin_16 = torch.ops.aten.argmin.default(baddbmm_16, -1);  baddbmm_16 = None
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(argmin_16, -1)
        expand_32 = torch.ops.aten.expand.default(unsqueeze_33, [8, 128, 2]);  unsqueeze_33 = None
        gather_16 = torch.ops.aten.gather.default(arg3_1, 1, expand_32);  expand_32 = None
        view_65 = torch.ops.aten.view.default(gather_16, [-1]);  gather_16 = None
        select_147 = torch.ops.aten.select.int(select_scatter_15, 1, 16)
        copy_16 = torch.ops.aten.copy.default(select_147, argmin_16);  select_147 = argmin_16 = None
        select_scatter_16 = torch.ops.aten.select_scatter.default(select_scatter_15, copy_16, 1, 16);  select_scatter_15 = copy_16 = None
        sub_16 = torch.ops.aten.sub.Tensor(select_144, view_65);  select_144 = view_65 = None
        div_16 = torch.ops.aten.div.Tensor(sub_16, select_143);  sub_16 = select_143 = None
        select_149 = torch.ops.aten.select.int(arg1_1, 0, 16)
        slice_62 = torch.ops.aten.slice.Tensor(select_149, 0, 16, 9223372036854775807);  select_149 = None
        slice_63 = torch.ops.aten.slice.Tensor(slice_scatter_14, 1, 16, 9223372036854775807)
        expand_33 = torch.ops.aten.expand.default(slice_63, [2048, 112]);  slice_63 = None
        mul_48 = torch.ops.aten.mul.Tensor(expand_33, 1);  expand_33 = None
        view_66 = torch.ops.aten.view.default(div_16, [2048, 1]);  div_16 = None
        mul_49 = torch.ops.aten.mul.Tensor(view_66, slice_62);  view_66 = slice_62 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_49, -1.0);  mul_49 = None
        add_16 = torch.ops.aten.add.Tensor(mul_48, mul_50);  mul_48 = mul_50 = None
        slice_scatter_15 = torch.ops.aten.slice_scatter.default(slice_scatter_14, add_16, 1, 16, 9223372036854775807);  slice_scatter_14 = add_16 = None
        select_151 = torch.ops.aten.select.int(arg1_1, 0, 17)
        select_152 = torch.ops.aten.select.int(select_151, 0, 17);  select_151 = None
        select_153 = torch.ops.aten.select.int(slice_scatter_15, 1, 17)
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_17 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_154 = torch.ops.aten.select.int(slice_scatter_15, 1, 17)
        view_68 = torch.ops.aten.view.default(select_154, [8, 128, 2]);  select_154 = None
        baddbmm_17 = torch.ops.aten.baddbmm.default(unsqueeze_34, view_68, permute_17, alpha = -2.0);  unsqueeze_34 = view_68 = permute_17 = None
        argmin_17 = torch.ops.aten.argmin.default(baddbmm_17, -1);  baddbmm_17 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(argmin_17, -1)
        expand_34 = torch.ops.aten.expand.default(unsqueeze_35, [8, 128, 2]);  unsqueeze_35 = None
        gather_17 = torch.ops.aten.gather.default(arg3_1, 1, expand_34);  expand_34 = None
        view_69 = torch.ops.aten.view.default(gather_17, [-1]);  gather_17 = None
        select_156 = torch.ops.aten.select.int(select_scatter_16, 1, 17)
        copy_17 = torch.ops.aten.copy.default(select_156, argmin_17);  select_156 = argmin_17 = None
        select_scatter_17 = torch.ops.aten.select_scatter.default(select_scatter_16, copy_17, 1, 17);  select_scatter_16 = copy_17 = None
        sub_17 = torch.ops.aten.sub.Tensor(select_153, view_69);  select_153 = view_69 = None
        div_17 = torch.ops.aten.div.Tensor(sub_17, select_152);  sub_17 = select_152 = None
        select_158 = torch.ops.aten.select.int(arg1_1, 0, 17)
        slice_66 = torch.ops.aten.slice.Tensor(select_158, 0, 17, 9223372036854775807);  select_158 = None
        slice_67 = torch.ops.aten.slice.Tensor(slice_scatter_15, 1, 17, 9223372036854775807)
        expand_35 = torch.ops.aten.expand.default(slice_67, [2048, 111]);  slice_67 = None
        mul_51 = torch.ops.aten.mul.Tensor(expand_35, 1);  expand_35 = None
        view_70 = torch.ops.aten.view.default(div_17, [2048, 1]);  div_17 = None
        mul_52 = torch.ops.aten.mul.Tensor(view_70, slice_66);  view_70 = slice_66 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, -1.0);  mul_52 = None
        add_17 = torch.ops.aten.add.Tensor(mul_51, mul_53);  mul_51 = mul_53 = None
        slice_scatter_16 = torch.ops.aten.slice_scatter.default(slice_scatter_15, add_17, 1, 17, 9223372036854775807);  slice_scatter_15 = add_17 = None
        select_160 = torch.ops.aten.select.int(arg1_1, 0, 18)
        select_161 = torch.ops.aten.select.int(select_160, 0, 18);  select_160 = None
        select_162 = torch.ops.aten.select.int(slice_scatter_16, 1, 18)
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_18 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_163 = torch.ops.aten.select.int(slice_scatter_16, 1, 18)
        view_72 = torch.ops.aten.view.default(select_163, [8, 128, 2]);  select_163 = None
        baddbmm_18 = torch.ops.aten.baddbmm.default(unsqueeze_36, view_72, permute_18, alpha = -2.0);  unsqueeze_36 = view_72 = permute_18 = None
        argmin_18 = torch.ops.aten.argmin.default(baddbmm_18, -1);  baddbmm_18 = None
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(argmin_18, -1)
        expand_36 = torch.ops.aten.expand.default(unsqueeze_37, [8, 128, 2]);  unsqueeze_37 = None
        gather_18 = torch.ops.aten.gather.default(arg3_1, 1, expand_36);  expand_36 = None
        view_73 = torch.ops.aten.view.default(gather_18, [-1]);  gather_18 = None
        select_165 = torch.ops.aten.select.int(select_scatter_17, 1, 18)
        copy_18 = torch.ops.aten.copy.default(select_165, argmin_18);  select_165 = argmin_18 = None
        select_scatter_18 = torch.ops.aten.select_scatter.default(select_scatter_17, copy_18, 1, 18);  select_scatter_17 = copy_18 = None
        sub_18 = torch.ops.aten.sub.Tensor(select_162, view_73);  select_162 = view_73 = None
        div_18 = torch.ops.aten.div.Tensor(sub_18, select_161);  sub_18 = select_161 = None
        select_167 = torch.ops.aten.select.int(arg1_1, 0, 18)
        slice_70 = torch.ops.aten.slice.Tensor(select_167, 0, 18, 9223372036854775807);  select_167 = None
        slice_71 = torch.ops.aten.slice.Tensor(slice_scatter_16, 1, 18, 9223372036854775807)
        expand_37 = torch.ops.aten.expand.default(slice_71, [2048, 110]);  slice_71 = None
        mul_54 = torch.ops.aten.mul.Tensor(expand_37, 1);  expand_37 = None
        view_74 = torch.ops.aten.view.default(div_18, [2048, 1]);  div_18 = None
        mul_55 = torch.ops.aten.mul.Tensor(view_74, slice_70);  view_74 = slice_70 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_55, -1.0);  mul_55 = None
        add_18 = torch.ops.aten.add.Tensor(mul_54, mul_56);  mul_54 = mul_56 = None
        slice_scatter_17 = torch.ops.aten.slice_scatter.default(slice_scatter_16, add_18, 1, 18, 9223372036854775807);  slice_scatter_16 = add_18 = None
        select_169 = torch.ops.aten.select.int(arg1_1, 0, 19)
        select_170 = torch.ops.aten.select.int(select_169, 0, 19);  select_169 = None
        select_171 = torch.ops.aten.select.int(slice_scatter_17, 1, 19)
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_19 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_172 = torch.ops.aten.select.int(slice_scatter_17, 1, 19)
        view_76 = torch.ops.aten.view.default(select_172, [8, 128, 2]);  select_172 = None
        baddbmm_19 = torch.ops.aten.baddbmm.default(unsqueeze_38, view_76, permute_19, alpha = -2.0);  unsqueeze_38 = view_76 = permute_19 = None
        argmin_19 = torch.ops.aten.argmin.default(baddbmm_19, -1);  baddbmm_19 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(argmin_19, -1)
        expand_38 = torch.ops.aten.expand.default(unsqueeze_39, [8, 128, 2]);  unsqueeze_39 = None
        gather_19 = torch.ops.aten.gather.default(arg3_1, 1, expand_38);  expand_38 = None
        view_77 = torch.ops.aten.view.default(gather_19, [-1]);  gather_19 = None
        select_174 = torch.ops.aten.select.int(select_scatter_18, 1, 19)
        copy_19 = torch.ops.aten.copy.default(select_174, argmin_19);  select_174 = argmin_19 = None
        select_scatter_19 = torch.ops.aten.select_scatter.default(select_scatter_18, copy_19, 1, 19);  select_scatter_18 = copy_19 = None
        sub_19 = torch.ops.aten.sub.Tensor(select_171, view_77);  select_171 = view_77 = None
        div_19 = torch.ops.aten.div.Tensor(sub_19, select_170);  sub_19 = select_170 = None
        select_176 = torch.ops.aten.select.int(arg1_1, 0, 19)
        slice_74 = torch.ops.aten.slice.Tensor(select_176, 0, 19, 9223372036854775807);  select_176 = None
        slice_75 = torch.ops.aten.slice.Tensor(slice_scatter_17, 1, 19, 9223372036854775807)
        expand_39 = torch.ops.aten.expand.default(slice_75, [2048, 109]);  slice_75 = None
        mul_57 = torch.ops.aten.mul.Tensor(expand_39, 1);  expand_39 = None
        view_78 = torch.ops.aten.view.default(div_19, [2048, 1]);  div_19 = None
        mul_58 = torch.ops.aten.mul.Tensor(view_78, slice_74);  view_78 = slice_74 = None
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, -1.0);  mul_58 = None
        add_19 = torch.ops.aten.add.Tensor(mul_57, mul_59);  mul_57 = mul_59 = None
        slice_scatter_18 = torch.ops.aten.slice_scatter.default(slice_scatter_17, add_19, 1, 19, 9223372036854775807);  slice_scatter_17 = add_19 = None
        select_178 = torch.ops.aten.select.int(arg1_1, 0, 20)
        select_179 = torch.ops.aten.select.int(select_178, 0, 20);  select_178 = None
        select_180 = torch.ops.aten.select.int(slice_scatter_18, 1, 20)
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_20 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_181 = torch.ops.aten.select.int(slice_scatter_18, 1, 20)
        view_80 = torch.ops.aten.view.default(select_181, [8, 128, 2]);  select_181 = None
        baddbmm_20 = torch.ops.aten.baddbmm.default(unsqueeze_40, view_80, permute_20, alpha = -2.0);  unsqueeze_40 = view_80 = permute_20 = None
        argmin_20 = torch.ops.aten.argmin.default(baddbmm_20, -1);  baddbmm_20 = None
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(argmin_20, -1)
        expand_40 = torch.ops.aten.expand.default(unsqueeze_41, [8, 128, 2]);  unsqueeze_41 = None
        gather_20 = torch.ops.aten.gather.default(arg3_1, 1, expand_40);  expand_40 = None
        view_81 = torch.ops.aten.view.default(gather_20, [-1]);  gather_20 = None
        select_183 = torch.ops.aten.select.int(select_scatter_19, 1, 20)
        copy_20 = torch.ops.aten.copy.default(select_183, argmin_20);  select_183 = argmin_20 = None
        select_scatter_20 = torch.ops.aten.select_scatter.default(select_scatter_19, copy_20, 1, 20);  select_scatter_19 = copy_20 = None
        sub_20 = torch.ops.aten.sub.Tensor(select_180, view_81);  select_180 = view_81 = None
        div_20 = torch.ops.aten.div.Tensor(sub_20, select_179);  sub_20 = select_179 = None
        select_185 = torch.ops.aten.select.int(arg1_1, 0, 20)
        slice_78 = torch.ops.aten.slice.Tensor(select_185, 0, 20, 9223372036854775807);  select_185 = None
        slice_79 = torch.ops.aten.slice.Tensor(slice_scatter_18, 1, 20, 9223372036854775807)
        expand_41 = torch.ops.aten.expand.default(slice_79, [2048, 108]);  slice_79 = None
        mul_60 = torch.ops.aten.mul.Tensor(expand_41, 1);  expand_41 = None
        view_82 = torch.ops.aten.view.default(div_20, [2048, 1]);  div_20 = None
        mul_61 = torch.ops.aten.mul.Tensor(view_82, slice_78);  view_82 = slice_78 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul_61, -1.0);  mul_61 = None
        add_20 = torch.ops.aten.add.Tensor(mul_60, mul_62);  mul_60 = mul_62 = None
        slice_scatter_19 = torch.ops.aten.slice_scatter.default(slice_scatter_18, add_20, 1, 20, 9223372036854775807);  slice_scatter_18 = add_20 = None
        select_187 = torch.ops.aten.select.int(arg1_1, 0, 21)
        select_188 = torch.ops.aten.select.int(select_187, 0, 21);  select_187 = None
        select_189 = torch.ops.aten.select.int(slice_scatter_19, 1, 21)
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_21 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_190 = torch.ops.aten.select.int(slice_scatter_19, 1, 21)
        view_84 = torch.ops.aten.view.default(select_190, [8, 128, 2]);  select_190 = None
        baddbmm_21 = torch.ops.aten.baddbmm.default(unsqueeze_42, view_84, permute_21, alpha = -2.0);  unsqueeze_42 = view_84 = permute_21 = None
        argmin_21 = torch.ops.aten.argmin.default(baddbmm_21, -1);  baddbmm_21 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(argmin_21, -1)
        expand_42 = torch.ops.aten.expand.default(unsqueeze_43, [8, 128, 2]);  unsqueeze_43 = None
        gather_21 = torch.ops.aten.gather.default(arg3_1, 1, expand_42);  expand_42 = None
        view_85 = torch.ops.aten.view.default(gather_21, [-1]);  gather_21 = None
        select_192 = torch.ops.aten.select.int(select_scatter_20, 1, 21)
        copy_21 = torch.ops.aten.copy.default(select_192, argmin_21);  select_192 = argmin_21 = None
        select_scatter_21 = torch.ops.aten.select_scatter.default(select_scatter_20, copy_21, 1, 21);  select_scatter_20 = copy_21 = None
        sub_21 = torch.ops.aten.sub.Tensor(select_189, view_85);  select_189 = view_85 = None
        div_21 = torch.ops.aten.div.Tensor(sub_21, select_188);  sub_21 = select_188 = None
        select_194 = torch.ops.aten.select.int(arg1_1, 0, 21)
        slice_82 = torch.ops.aten.slice.Tensor(select_194, 0, 21, 9223372036854775807);  select_194 = None
        slice_83 = torch.ops.aten.slice.Tensor(slice_scatter_19, 1, 21, 9223372036854775807)
        expand_43 = torch.ops.aten.expand.default(slice_83, [2048, 107]);  slice_83 = None
        mul_63 = torch.ops.aten.mul.Tensor(expand_43, 1);  expand_43 = None
        view_86 = torch.ops.aten.view.default(div_21, [2048, 1]);  div_21 = None
        mul_64 = torch.ops.aten.mul.Tensor(view_86, slice_82);  view_86 = slice_82 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_64, -1.0);  mul_64 = None
        add_21 = torch.ops.aten.add.Tensor(mul_63, mul_65);  mul_63 = mul_65 = None
        slice_scatter_20 = torch.ops.aten.slice_scatter.default(slice_scatter_19, add_21, 1, 21, 9223372036854775807);  slice_scatter_19 = add_21 = None
        select_196 = torch.ops.aten.select.int(arg1_1, 0, 22)
        select_197 = torch.ops.aten.select.int(select_196, 0, 22);  select_196 = None
        select_198 = torch.ops.aten.select.int(slice_scatter_20, 1, 22)
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_22 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_199 = torch.ops.aten.select.int(slice_scatter_20, 1, 22)
        view_88 = torch.ops.aten.view.default(select_199, [8, 128, 2]);  select_199 = None
        baddbmm_22 = torch.ops.aten.baddbmm.default(unsqueeze_44, view_88, permute_22, alpha = -2.0);  unsqueeze_44 = view_88 = permute_22 = None
        argmin_22 = torch.ops.aten.argmin.default(baddbmm_22, -1);  baddbmm_22 = None
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(argmin_22, -1)
        expand_44 = torch.ops.aten.expand.default(unsqueeze_45, [8, 128, 2]);  unsqueeze_45 = None
        gather_22 = torch.ops.aten.gather.default(arg3_1, 1, expand_44);  expand_44 = None
        view_89 = torch.ops.aten.view.default(gather_22, [-1]);  gather_22 = None
        select_201 = torch.ops.aten.select.int(select_scatter_21, 1, 22)
        copy_22 = torch.ops.aten.copy.default(select_201, argmin_22);  select_201 = argmin_22 = None
        select_scatter_22 = torch.ops.aten.select_scatter.default(select_scatter_21, copy_22, 1, 22);  select_scatter_21 = copy_22 = None
        sub_22 = torch.ops.aten.sub.Tensor(select_198, view_89);  select_198 = view_89 = None
        div_22 = torch.ops.aten.div.Tensor(sub_22, select_197);  sub_22 = select_197 = None
        select_203 = torch.ops.aten.select.int(arg1_1, 0, 22)
        slice_86 = torch.ops.aten.slice.Tensor(select_203, 0, 22, 9223372036854775807);  select_203 = None
        slice_87 = torch.ops.aten.slice.Tensor(slice_scatter_20, 1, 22, 9223372036854775807)
        expand_45 = torch.ops.aten.expand.default(slice_87, [2048, 106]);  slice_87 = None
        mul_66 = torch.ops.aten.mul.Tensor(expand_45, 1);  expand_45 = None
        view_90 = torch.ops.aten.view.default(div_22, [2048, 1]);  div_22 = None
        mul_67 = torch.ops.aten.mul.Tensor(view_90, slice_86);  view_90 = slice_86 = None
        mul_68 = torch.ops.aten.mul.Tensor(mul_67, -1.0);  mul_67 = None
        add_22 = torch.ops.aten.add.Tensor(mul_66, mul_68);  mul_66 = mul_68 = None
        slice_scatter_21 = torch.ops.aten.slice_scatter.default(slice_scatter_20, add_22, 1, 22, 9223372036854775807);  slice_scatter_20 = add_22 = None
        select_205 = torch.ops.aten.select.int(arg1_1, 0, 23)
        select_206 = torch.ops.aten.select.int(select_205, 0, 23);  select_205 = None
        select_207 = torch.ops.aten.select.int(slice_scatter_21, 1, 23)
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_23 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_208 = torch.ops.aten.select.int(slice_scatter_21, 1, 23)
        view_92 = torch.ops.aten.view.default(select_208, [8, 128, 2]);  select_208 = None
        baddbmm_23 = torch.ops.aten.baddbmm.default(unsqueeze_46, view_92, permute_23, alpha = -2.0);  unsqueeze_46 = view_92 = permute_23 = None
        argmin_23 = torch.ops.aten.argmin.default(baddbmm_23, -1);  baddbmm_23 = None
        unsqueeze_47 = torch.ops.aten.unsqueeze.default(argmin_23, -1)
        expand_46 = torch.ops.aten.expand.default(unsqueeze_47, [8, 128, 2]);  unsqueeze_47 = None
        gather_23 = torch.ops.aten.gather.default(arg3_1, 1, expand_46);  expand_46 = None
        view_93 = torch.ops.aten.view.default(gather_23, [-1]);  gather_23 = None
        select_210 = torch.ops.aten.select.int(select_scatter_22, 1, 23)
        copy_23 = torch.ops.aten.copy.default(select_210, argmin_23);  select_210 = argmin_23 = None
        select_scatter_23 = torch.ops.aten.select_scatter.default(select_scatter_22, copy_23, 1, 23);  select_scatter_22 = copy_23 = None
        sub_23 = torch.ops.aten.sub.Tensor(select_207, view_93);  select_207 = view_93 = None
        div_23 = torch.ops.aten.div.Tensor(sub_23, select_206);  sub_23 = select_206 = None
        select_212 = torch.ops.aten.select.int(arg1_1, 0, 23)
        slice_90 = torch.ops.aten.slice.Tensor(select_212, 0, 23, 9223372036854775807);  select_212 = None
        slice_91 = torch.ops.aten.slice.Tensor(slice_scatter_21, 1, 23, 9223372036854775807)
        expand_47 = torch.ops.aten.expand.default(slice_91, [2048, 105]);  slice_91 = None
        mul_69 = torch.ops.aten.mul.Tensor(expand_47, 1);  expand_47 = None
        view_94 = torch.ops.aten.view.default(div_23, [2048, 1]);  div_23 = None
        mul_70 = torch.ops.aten.mul.Tensor(view_94, slice_90);  view_94 = slice_90 = None
        mul_71 = torch.ops.aten.mul.Tensor(mul_70, -1.0);  mul_70 = None
        add_23 = torch.ops.aten.add.Tensor(mul_69, mul_71);  mul_69 = mul_71 = None
        slice_scatter_22 = torch.ops.aten.slice_scatter.default(slice_scatter_21, add_23, 1, 23, 9223372036854775807);  slice_scatter_21 = add_23 = None
        select_214 = torch.ops.aten.select.int(arg1_1, 0, 24)
        select_215 = torch.ops.aten.select.int(select_214, 0, 24);  select_214 = None
        select_216 = torch.ops.aten.select.int(slice_scatter_22, 1, 24)
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_24 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_217 = torch.ops.aten.select.int(slice_scatter_22, 1, 24)
        view_96 = torch.ops.aten.view.default(select_217, [8, 128, 2]);  select_217 = None
        baddbmm_24 = torch.ops.aten.baddbmm.default(unsqueeze_48, view_96, permute_24, alpha = -2.0);  unsqueeze_48 = view_96 = permute_24 = None
        argmin_24 = torch.ops.aten.argmin.default(baddbmm_24, -1);  baddbmm_24 = None
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(argmin_24, -1)
        expand_48 = torch.ops.aten.expand.default(unsqueeze_49, [8, 128, 2]);  unsqueeze_49 = None
        gather_24 = torch.ops.aten.gather.default(arg3_1, 1, expand_48);  expand_48 = None
        view_97 = torch.ops.aten.view.default(gather_24, [-1]);  gather_24 = None
        select_219 = torch.ops.aten.select.int(select_scatter_23, 1, 24)
        copy_24 = torch.ops.aten.copy.default(select_219, argmin_24);  select_219 = argmin_24 = None
        select_scatter_24 = torch.ops.aten.select_scatter.default(select_scatter_23, copy_24, 1, 24);  select_scatter_23 = copy_24 = None
        sub_24 = torch.ops.aten.sub.Tensor(select_216, view_97);  select_216 = view_97 = None
        div_24 = torch.ops.aten.div.Tensor(sub_24, select_215);  sub_24 = select_215 = None
        select_221 = torch.ops.aten.select.int(arg1_1, 0, 24)
        slice_94 = torch.ops.aten.slice.Tensor(select_221, 0, 24, 9223372036854775807);  select_221 = None
        slice_95 = torch.ops.aten.slice.Tensor(slice_scatter_22, 1, 24, 9223372036854775807)
        expand_49 = torch.ops.aten.expand.default(slice_95, [2048, 104]);  slice_95 = None
        mul_72 = torch.ops.aten.mul.Tensor(expand_49, 1);  expand_49 = None
        view_98 = torch.ops.aten.view.default(div_24, [2048, 1]);  div_24 = None
        mul_73 = torch.ops.aten.mul.Tensor(view_98, slice_94);  view_98 = slice_94 = None
        mul_74 = torch.ops.aten.mul.Tensor(mul_73, -1.0);  mul_73 = None
        add_24 = torch.ops.aten.add.Tensor(mul_72, mul_74);  mul_72 = mul_74 = None
        slice_scatter_23 = torch.ops.aten.slice_scatter.default(slice_scatter_22, add_24, 1, 24, 9223372036854775807);  slice_scatter_22 = add_24 = None
        select_223 = torch.ops.aten.select.int(arg1_1, 0, 25)
        select_224 = torch.ops.aten.select.int(select_223, 0, 25);  select_223 = None
        select_225 = torch.ops.aten.select.int(slice_scatter_23, 1, 25)
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_25 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_226 = torch.ops.aten.select.int(slice_scatter_23, 1, 25)
        view_100 = torch.ops.aten.view.default(select_226, [8, 128, 2]);  select_226 = None
        baddbmm_25 = torch.ops.aten.baddbmm.default(unsqueeze_50, view_100, permute_25, alpha = -2.0);  unsqueeze_50 = view_100 = permute_25 = None
        argmin_25 = torch.ops.aten.argmin.default(baddbmm_25, -1);  baddbmm_25 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(argmin_25, -1)
        expand_50 = torch.ops.aten.expand.default(unsqueeze_51, [8, 128, 2]);  unsqueeze_51 = None
        gather_25 = torch.ops.aten.gather.default(arg3_1, 1, expand_50);  expand_50 = None
        view_101 = torch.ops.aten.view.default(gather_25, [-1]);  gather_25 = None
        select_228 = torch.ops.aten.select.int(select_scatter_24, 1, 25)
        copy_25 = torch.ops.aten.copy.default(select_228, argmin_25);  select_228 = argmin_25 = None
        select_scatter_25 = torch.ops.aten.select_scatter.default(select_scatter_24, copy_25, 1, 25);  select_scatter_24 = copy_25 = None
        sub_25 = torch.ops.aten.sub.Tensor(select_225, view_101);  select_225 = view_101 = None
        div_25 = torch.ops.aten.div.Tensor(sub_25, select_224);  sub_25 = select_224 = None
        select_230 = torch.ops.aten.select.int(arg1_1, 0, 25)
        slice_98 = torch.ops.aten.slice.Tensor(select_230, 0, 25, 9223372036854775807);  select_230 = None
        slice_99 = torch.ops.aten.slice.Tensor(slice_scatter_23, 1, 25, 9223372036854775807)
        expand_51 = torch.ops.aten.expand.default(slice_99, [2048, 103]);  slice_99 = None
        mul_75 = torch.ops.aten.mul.Tensor(expand_51, 1);  expand_51 = None
        view_102 = torch.ops.aten.view.default(div_25, [2048, 1]);  div_25 = None
        mul_76 = torch.ops.aten.mul.Tensor(view_102, slice_98);  view_102 = slice_98 = None
        mul_77 = torch.ops.aten.mul.Tensor(mul_76, -1.0);  mul_76 = None
        add_25 = torch.ops.aten.add.Tensor(mul_75, mul_77);  mul_75 = mul_77 = None
        slice_scatter_24 = torch.ops.aten.slice_scatter.default(slice_scatter_23, add_25, 1, 25, 9223372036854775807);  slice_scatter_23 = add_25 = None
        select_232 = torch.ops.aten.select.int(arg1_1, 0, 26)
        select_233 = torch.ops.aten.select.int(select_232, 0, 26);  select_232 = None
        select_234 = torch.ops.aten.select.int(slice_scatter_24, 1, 26)
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_26 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_235 = torch.ops.aten.select.int(slice_scatter_24, 1, 26)
        view_104 = torch.ops.aten.view.default(select_235, [8, 128, 2]);  select_235 = None
        baddbmm_26 = torch.ops.aten.baddbmm.default(unsqueeze_52, view_104, permute_26, alpha = -2.0);  unsqueeze_52 = view_104 = permute_26 = None
        argmin_26 = torch.ops.aten.argmin.default(baddbmm_26, -1);  baddbmm_26 = None
        unsqueeze_53 = torch.ops.aten.unsqueeze.default(argmin_26, -1)
        expand_52 = torch.ops.aten.expand.default(unsqueeze_53, [8, 128, 2]);  unsqueeze_53 = None
        gather_26 = torch.ops.aten.gather.default(arg3_1, 1, expand_52);  expand_52 = None
        view_105 = torch.ops.aten.view.default(gather_26, [-1]);  gather_26 = None
        select_237 = torch.ops.aten.select.int(select_scatter_25, 1, 26)
        copy_26 = torch.ops.aten.copy.default(select_237, argmin_26);  select_237 = argmin_26 = None
        select_scatter_26 = torch.ops.aten.select_scatter.default(select_scatter_25, copy_26, 1, 26);  select_scatter_25 = copy_26 = None
        sub_26 = torch.ops.aten.sub.Tensor(select_234, view_105);  select_234 = view_105 = None
        div_26 = torch.ops.aten.div.Tensor(sub_26, select_233);  sub_26 = select_233 = None
        select_239 = torch.ops.aten.select.int(arg1_1, 0, 26)
        slice_102 = torch.ops.aten.slice.Tensor(select_239, 0, 26, 9223372036854775807);  select_239 = None
        slice_103 = torch.ops.aten.slice.Tensor(slice_scatter_24, 1, 26, 9223372036854775807)
        expand_53 = torch.ops.aten.expand.default(slice_103, [2048, 102]);  slice_103 = None
        mul_78 = torch.ops.aten.mul.Tensor(expand_53, 1);  expand_53 = None
        view_106 = torch.ops.aten.view.default(div_26, [2048, 1]);  div_26 = None
        mul_79 = torch.ops.aten.mul.Tensor(view_106, slice_102);  view_106 = slice_102 = None
        mul_80 = torch.ops.aten.mul.Tensor(mul_79, -1.0);  mul_79 = None
        add_26 = torch.ops.aten.add.Tensor(mul_78, mul_80);  mul_78 = mul_80 = None
        slice_scatter_25 = torch.ops.aten.slice_scatter.default(slice_scatter_24, add_26, 1, 26, 9223372036854775807);  slice_scatter_24 = add_26 = None
        select_241 = torch.ops.aten.select.int(arg1_1, 0, 27)
        select_242 = torch.ops.aten.select.int(select_241, 0, 27);  select_241 = None
        select_243 = torch.ops.aten.select.int(slice_scatter_25, 1, 27)
        unsqueeze_54 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_27 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_244 = torch.ops.aten.select.int(slice_scatter_25, 1, 27)
        view_108 = torch.ops.aten.view.default(select_244, [8, 128, 2]);  select_244 = None
        baddbmm_27 = torch.ops.aten.baddbmm.default(unsqueeze_54, view_108, permute_27, alpha = -2.0);  unsqueeze_54 = view_108 = permute_27 = None
        argmin_27 = torch.ops.aten.argmin.default(baddbmm_27, -1);  baddbmm_27 = None
        unsqueeze_55 = torch.ops.aten.unsqueeze.default(argmin_27, -1)
        expand_54 = torch.ops.aten.expand.default(unsqueeze_55, [8, 128, 2]);  unsqueeze_55 = None
        gather_27 = torch.ops.aten.gather.default(arg3_1, 1, expand_54);  expand_54 = None
        view_109 = torch.ops.aten.view.default(gather_27, [-1]);  gather_27 = None
        select_246 = torch.ops.aten.select.int(select_scatter_26, 1, 27)
        copy_27 = torch.ops.aten.copy.default(select_246, argmin_27);  select_246 = argmin_27 = None
        select_scatter_27 = torch.ops.aten.select_scatter.default(select_scatter_26, copy_27, 1, 27);  select_scatter_26 = copy_27 = None
        sub_27 = torch.ops.aten.sub.Tensor(select_243, view_109);  select_243 = view_109 = None
        div_27 = torch.ops.aten.div.Tensor(sub_27, select_242);  sub_27 = select_242 = None
        select_248 = torch.ops.aten.select.int(arg1_1, 0, 27)
        slice_106 = torch.ops.aten.slice.Tensor(select_248, 0, 27, 9223372036854775807);  select_248 = None
        slice_107 = torch.ops.aten.slice.Tensor(slice_scatter_25, 1, 27, 9223372036854775807)
        expand_55 = torch.ops.aten.expand.default(slice_107, [2048, 101]);  slice_107 = None
        mul_81 = torch.ops.aten.mul.Tensor(expand_55, 1);  expand_55 = None
        view_110 = torch.ops.aten.view.default(div_27, [2048, 1]);  div_27 = None
        mul_82 = torch.ops.aten.mul.Tensor(view_110, slice_106);  view_110 = slice_106 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_82, -1.0);  mul_82 = None
        add_27 = torch.ops.aten.add.Tensor(mul_81, mul_83);  mul_81 = mul_83 = None
        slice_scatter_26 = torch.ops.aten.slice_scatter.default(slice_scatter_25, add_27, 1, 27, 9223372036854775807);  slice_scatter_25 = add_27 = None
        select_250 = torch.ops.aten.select.int(arg1_1, 0, 28)
        select_251 = torch.ops.aten.select.int(select_250, 0, 28);  select_250 = None
        select_252 = torch.ops.aten.select.int(slice_scatter_26, 1, 28)
        unsqueeze_56 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_28 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_253 = torch.ops.aten.select.int(slice_scatter_26, 1, 28)
        view_112 = torch.ops.aten.view.default(select_253, [8, 128, 2]);  select_253 = None
        baddbmm_28 = torch.ops.aten.baddbmm.default(unsqueeze_56, view_112, permute_28, alpha = -2.0);  unsqueeze_56 = view_112 = permute_28 = None
        argmin_28 = torch.ops.aten.argmin.default(baddbmm_28, -1);  baddbmm_28 = None
        unsqueeze_57 = torch.ops.aten.unsqueeze.default(argmin_28, -1)
        expand_56 = torch.ops.aten.expand.default(unsqueeze_57, [8, 128, 2]);  unsqueeze_57 = None
        gather_28 = torch.ops.aten.gather.default(arg3_1, 1, expand_56);  expand_56 = None
        view_113 = torch.ops.aten.view.default(gather_28, [-1]);  gather_28 = None
        select_255 = torch.ops.aten.select.int(select_scatter_27, 1, 28)
        copy_28 = torch.ops.aten.copy.default(select_255, argmin_28);  select_255 = argmin_28 = None
        select_scatter_28 = torch.ops.aten.select_scatter.default(select_scatter_27, copy_28, 1, 28);  select_scatter_27 = copy_28 = None
        sub_28 = torch.ops.aten.sub.Tensor(select_252, view_113);  select_252 = view_113 = None
        div_28 = torch.ops.aten.div.Tensor(sub_28, select_251);  sub_28 = select_251 = None
        select_257 = torch.ops.aten.select.int(arg1_1, 0, 28)
        slice_110 = torch.ops.aten.slice.Tensor(select_257, 0, 28, 9223372036854775807);  select_257 = None
        slice_111 = torch.ops.aten.slice.Tensor(slice_scatter_26, 1, 28, 9223372036854775807)
        expand_57 = torch.ops.aten.expand.default(slice_111, [2048, 100]);  slice_111 = None
        mul_84 = torch.ops.aten.mul.Tensor(expand_57, 1);  expand_57 = None
        view_114 = torch.ops.aten.view.default(div_28, [2048, 1]);  div_28 = None
        mul_85 = torch.ops.aten.mul.Tensor(view_114, slice_110);  view_114 = slice_110 = None
        mul_86 = torch.ops.aten.mul.Tensor(mul_85, -1.0);  mul_85 = None
        add_28 = torch.ops.aten.add.Tensor(mul_84, mul_86);  mul_84 = mul_86 = None
        slice_scatter_27 = torch.ops.aten.slice_scatter.default(slice_scatter_26, add_28, 1, 28, 9223372036854775807);  slice_scatter_26 = add_28 = None
        select_259 = torch.ops.aten.select.int(arg1_1, 0, 29)
        select_260 = torch.ops.aten.select.int(select_259, 0, 29);  select_259 = None
        select_261 = torch.ops.aten.select.int(slice_scatter_27, 1, 29)
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_29 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_262 = torch.ops.aten.select.int(slice_scatter_27, 1, 29)
        view_116 = torch.ops.aten.view.default(select_262, [8, 128, 2]);  select_262 = None
        baddbmm_29 = torch.ops.aten.baddbmm.default(unsqueeze_58, view_116, permute_29, alpha = -2.0);  unsqueeze_58 = view_116 = permute_29 = None
        argmin_29 = torch.ops.aten.argmin.default(baddbmm_29, -1);  baddbmm_29 = None
        unsqueeze_59 = torch.ops.aten.unsqueeze.default(argmin_29, -1)
        expand_58 = torch.ops.aten.expand.default(unsqueeze_59, [8, 128, 2]);  unsqueeze_59 = None
        gather_29 = torch.ops.aten.gather.default(arg3_1, 1, expand_58);  expand_58 = None
        view_117 = torch.ops.aten.view.default(gather_29, [-1]);  gather_29 = None
        select_264 = torch.ops.aten.select.int(select_scatter_28, 1, 29)
        copy_29 = torch.ops.aten.copy.default(select_264, argmin_29);  select_264 = argmin_29 = None
        select_scatter_29 = torch.ops.aten.select_scatter.default(select_scatter_28, copy_29, 1, 29);  select_scatter_28 = copy_29 = None
        sub_29 = torch.ops.aten.sub.Tensor(select_261, view_117);  select_261 = view_117 = None
        div_29 = torch.ops.aten.div.Tensor(sub_29, select_260);  sub_29 = select_260 = None
        select_266 = torch.ops.aten.select.int(arg1_1, 0, 29)
        slice_114 = torch.ops.aten.slice.Tensor(select_266, 0, 29, 9223372036854775807);  select_266 = None
        slice_115 = torch.ops.aten.slice.Tensor(slice_scatter_27, 1, 29, 9223372036854775807)
        expand_59 = torch.ops.aten.expand.default(slice_115, [2048, 99]);  slice_115 = None
        mul_87 = torch.ops.aten.mul.Tensor(expand_59, 1);  expand_59 = None
        view_118 = torch.ops.aten.view.default(div_29, [2048, 1]);  div_29 = None
        mul_88 = torch.ops.aten.mul.Tensor(view_118, slice_114);  view_118 = slice_114 = None
        mul_89 = torch.ops.aten.mul.Tensor(mul_88, -1.0);  mul_88 = None
        add_29 = torch.ops.aten.add.Tensor(mul_87, mul_89);  mul_87 = mul_89 = None
        slice_scatter_28 = torch.ops.aten.slice_scatter.default(slice_scatter_27, add_29, 1, 29, 9223372036854775807);  slice_scatter_27 = add_29 = None
        select_268 = torch.ops.aten.select.int(arg1_1, 0, 30)
        select_269 = torch.ops.aten.select.int(select_268, 0, 30);  select_268 = None
        select_270 = torch.ops.aten.select.int(slice_scatter_28, 1, 30)
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_30 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_271 = torch.ops.aten.select.int(slice_scatter_28, 1, 30)
        view_120 = torch.ops.aten.view.default(select_271, [8, 128, 2]);  select_271 = None
        baddbmm_30 = torch.ops.aten.baddbmm.default(unsqueeze_60, view_120, permute_30, alpha = -2.0);  unsqueeze_60 = view_120 = permute_30 = None
        argmin_30 = torch.ops.aten.argmin.default(baddbmm_30, -1);  baddbmm_30 = None
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(argmin_30, -1)
        expand_60 = torch.ops.aten.expand.default(unsqueeze_61, [8, 128, 2]);  unsqueeze_61 = None
        gather_30 = torch.ops.aten.gather.default(arg3_1, 1, expand_60);  expand_60 = None
        view_121 = torch.ops.aten.view.default(gather_30, [-1]);  gather_30 = None
        select_273 = torch.ops.aten.select.int(select_scatter_29, 1, 30)
        copy_30 = torch.ops.aten.copy.default(select_273, argmin_30);  select_273 = argmin_30 = None
        select_scatter_30 = torch.ops.aten.select_scatter.default(select_scatter_29, copy_30, 1, 30);  select_scatter_29 = copy_30 = None
        sub_30 = torch.ops.aten.sub.Tensor(select_270, view_121);  select_270 = view_121 = None
        div_30 = torch.ops.aten.div.Tensor(sub_30, select_269);  sub_30 = select_269 = None
        select_275 = torch.ops.aten.select.int(arg1_1, 0, 30)
        slice_118 = torch.ops.aten.slice.Tensor(select_275, 0, 30, 9223372036854775807);  select_275 = None
        slice_119 = torch.ops.aten.slice.Tensor(slice_scatter_28, 1, 30, 9223372036854775807)
        expand_61 = torch.ops.aten.expand.default(slice_119, [2048, 98]);  slice_119 = None
        mul_90 = torch.ops.aten.mul.Tensor(expand_61, 1);  expand_61 = None
        view_122 = torch.ops.aten.view.default(div_30, [2048, 1]);  div_30 = None
        mul_91 = torch.ops.aten.mul.Tensor(view_122, slice_118);  view_122 = slice_118 = None
        mul_92 = torch.ops.aten.mul.Tensor(mul_91, -1.0);  mul_91 = None
        add_30 = torch.ops.aten.add.Tensor(mul_90, mul_92);  mul_90 = mul_92 = None
        slice_scatter_29 = torch.ops.aten.slice_scatter.default(slice_scatter_28, add_30, 1, 30, 9223372036854775807);  slice_scatter_28 = add_30 = None
        select_277 = torch.ops.aten.select.int(arg1_1, 0, 31)
        select_278 = torch.ops.aten.select.int(select_277, 0, 31);  select_277 = None
        select_279 = torch.ops.aten.select.int(slice_scatter_29, 1, 31)
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_31 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_280 = torch.ops.aten.select.int(slice_scatter_29, 1, 31)
        view_124 = torch.ops.aten.view.default(select_280, [8, 128, 2]);  select_280 = None
        baddbmm_31 = torch.ops.aten.baddbmm.default(unsqueeze_62, view_124, permute_31, alpha = -2.0);  unsqueeze_62 = view_124 = permute_31 = None
        argmin_31 = torch.ops.aten.argmin.default(baddbmm_31, -1);  baddbmm_31 = None
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(argmin_31, -1)
        expand_62 = torch.ops.aten.expand.default(unsqueeze_63, [8, 128, 2]);  unsqueeze_63 = None
        gather_31 = torch.ops.aten.gather.default(arg3_1, 1, expand_62);  expand_62 = None
        view_125 = torch.ops.aten.view.default(gather_31, [-1]);  gather_31 = None
        select_282 = torch.ops.aten.select.int(select_scatter_30, 1, 31)
        copy_31 = torch.ops.aten.copy.default(select_282, argmin_31);  select_282 = argmin_31 = None
        select_scatter_31 = torch.ops.aten.select_scatter.default(select_scatter_30, copy_31, 1, 31);  select_scatter_30 = copy_31 = None
        sub_31 = torch.ops.aten.sub.Tensor(select_279, view_125);  select_279 = view_125 = None
        div_31 = torch.ops.aten.div.Tensor(sub_31, select_278);  sub_31 = select_278 = None
        select_284 = torch.ops.aten.select.int(arg1_1, 0, 31)
        slice_122 = torch.ops.aten.slice.Tensor(select_284, 0, 31, 9223372036854775807);  select_284 = None
        slice_123 = torch.ops.aten.slice.Tensor(slice_scatter_29, 1, 31, 9223372036854775807)
        expand_63 = torch.ops.aten.expand.default(slice_123, [2048, 97]);  slice_123 = None
        mul_93 = torch.ops.aten.mul.Tensor(expand_63, 1);  expand_63 = None
        view_126 = torch.ops.aten.view.default(div_31, [2048, 1]);  div_31 = None
        mul_94 = torch.ops.aten.mul.Tensor(view_126, slice_122);  view_126 = slice_122 = None
        mul_95 = torch.ops.aten.mul.Tensor(mul_94, -1.0);  mul_94 = None
        add_31 = torch.ops.aten.add.Tensor(mul_93, mul_95);  mul_93 = mul_95 = None
        slice_scatter_30 = torch.ops.aten.slice_scatter.default(slice_scatter_29, add_31, 1, 31, 9223372036854775807);  slice_scatter_29 = add_31 = None
        select_286 = torch.ops.aten.select.int(arg1_1, 0, 32)
        select_287 = torch.ops.aten.select.int(select_286, 0, 32);  select_286 = None
        select_288 = torch.ops.aten.select.int(slice_scatter_30, 1, 32)
        unsqueeze_64 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_32 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_289 = torch.ops.aten.select.int(slice_scatter_30, 1, 32)
        view_128 = torch.ops.aten.view.default(select_289, [8, 128, 2]);  select_289 = None
        baddbmm_32 = torch.ops.aten.baddbmm.default(unsqueeze_64, view_128, permute_32, alpha = -2.0);  unsqueeze_64 = view_128 = permute_32 = None
        argmin_32 = torch.ops.aten.argmin.default(baddbmm_32, -1);  baddbmm_32 = None
        unsqueeze_65 = torch.ops.aten.unsqueeze.default(argmin_32, -1)
        expand_64 = torch.ops.aten.expand.default(unsqueeze_65, [8, 128, 2]);  unsqueeze_65 = None
        gather_32 = torch.ops.aten.gather.default(arg3_1, 1, expand_64);  expand_64 = None
        view_129 = torch.ops.aten.view.default(gather_32, [-1]);  gather_32 = None
        select_291 = torch.ops.aten.select.int(select_scatter_31, 1, 32)
        copy_32 = torch.ops.aten.copy.default(select_291, argmin_32);  select_291 = argmin_32 = None
        select_scatter_32 = torch.ops.aten.select_scatter.default(select_scatter_31, copy_32, 1, 32);  select_scatter_31 = copy_32 = None
        sub_32 = torch.ops.aten.sub.Tensor(select_288, view_129);  select_288 = view_129 = None
        div_32 = torch.ops.aten.div.Tensor(sub_32, select_287);  sub_32 = select_287 = None
        select_293 = torch.ops.aten.select.int(arg1_1, 0, 32)
        slice_126 = torch.ops.aten.slice.Tensor(select_293, 0, 32, 9223372036854775807);  select_293 = None
        slice_127 = torch.ops.aten.slice.Tensor(slice_scatter_30, 1, 32, 9223372036854775807)
        expand_65 = torch.ops.aten.expand.default(slice_127, [2048, 96]);  slice_127 = None
        mul_96 = torch.ops.aten.mul.Tensor(expand_65, 1);  expand_65 = None
        view_130 = torch.ops.aten.view.default(div_32, [2048, 1]);  div_32 = None
        mul_97 = torch.ops.aten.mul.Tensor(view_130, slice_126);  view_130 = slice_126 = None
        mul_98 = torch.ops.aten.mul.Tensor(mul_97, -1.0);  mul_97 = None
        add_32 = torch.ops.aten.add.Tensor(mul_96, mul_98);  mul_96 = mul_98 = None
        slice_scatter_31 = torch.ops.aten.slice_scatter.default(slice_scatter_30, add_32, 1, 32, 9223372036854775807);  slice_scatter_30 = add_32 = None
        select_295 = torch.ops.aten.select.int(arg1_1, 0, 33)
        select_296 = torch.ops.aten.select.int(select_295, 0, 33);  select_295 = None
        select_297 = torch.ops.aten.select.int(slice_scatter_31, 1, 33)
        unsqueeze_66 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_33 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_298 = torch.ops.aten.select.int(slice_scatter_31, 1, 33)
        view_132 = torch.ops.aten.view.default(select_298, [8, 128, 2]);  select_298 = None
        baddbmm_33 = torch.ops.aten.baddbmm.default(unsqueeze_66, view_132, permute_33, alpha = -2.0);  unsqueeze_66 = view_132 = permute_33 = None
        argmin_33 = torch.ops.aten.argmin.default(baddbmm_33, -1);  baddbmm_33 = None
        unsqueeze_67 = torch.ops.aten.unsqueeze.default(argmin_33, -1)
        expand_66 = torch.ops.aten.expand.default(unsqueeze_67, [8, 128, 2]);  unsqueeze_67 = None
        gather_33 = torch.ops.aten.gather.default(arg3_1, 1, expand_66);  expand_66 = None
        view_133 = torch.ops.aten.view.default(gather_33, [-1]);  gather_33 = None
        select_300 = torch.ops.aten.select.int(select_scatter_32, 1, 33)
        copy_33 = torch.ops.aten.copy.default(select_300, argmin_33);  select_300 = argmin_33 = None
        select_scatter_33 = torch.ops.aten.select_scatter.default(select_scatter_32, copy_33, 1, 33);  select_scatter_32 = copy_33 = None
        sub_33 = torch.ops.aten.sub.Tensor(select_297, view_133);  select_297 = view_133 = None
        div_33 = torch.ops.aten.div.Tensor(sub_33, select_296);  sub_33 = select_296 = None
        select_302 = torch.ops.aten.select.int(arg1_1, 0, 33)
        slice_130 = torch.ops.aten.slice.Tensor(select_302, 0, 33, 9223372036854775807);  select_302 = None
        slice_131 = torch.ops.aten.slice.Tensor(slice_scatter_31, 1, 33, 9223372036854775807)
        expand_67 = torch.ops.aten.expand.default(slice_131, [2048, 95]);  slice_131 = None
        mul_99 = torch.ops.aten.mul.Tensor(expand_67, 1);  expand_67 = None
        view_134 = torch.ops.aten.view.default(div_33, [2048, 1]);  div_33 = None
        mul_100 = torch.ops.aten.mul.Tensor(view_134, slice_130);  view_134 = slice_130 = None
        mul_101 = torch.ops.aten.mul.Tensor(mul_100, -1.0);  mul_100 = None
        add_33 = torch.ops.aten.add.Tensor(mul_99, mul_101);  mul_99 = mul_101 = None
        slice_scatter_32 = torch.ops.aten.slice_scatter.default(slice_scatter_31, add_33, 1, 33, 9223372036854775807);  slice_scatter_31 = add_33 = None
        select_304 = torch.ops.aten.select.int(arg1_1, 0, 34)
        select_305 = torch.ops.aten.select.int(select_304, 0, 34);  select_304 = None
        select_306 = torch.ops.aten.select.int(slice_scatter_32, 1, 34)
        unsqueeze_68 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_34 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_307 = torch.ops.aten.select.int(slice_scatter_32, 1, 34)
        view_136 = torch.ops.aten.view.default(select_307, [8, 128, 2]);  select_307 = None
        baddbmm_34 = torch.ops.aten.baddbmm.default(unsqueeze_68, view_136, permute_34, alpha = -2.0);  unsqueeze_68 = view_136 = permute_34 = None
        argmin_34 = torch.ops.aten.argmin.default(baddbmm_34, -1);  baddbmm_34 = None
        unsqueeze_69 = torch.ops.aten.unsqueeze.default(argmin_34, -1)
        expand_68 = torch.ops.aten.expand.default(unsqueeze_69, [8, 128, 2]);  unsqueeze_69 = None
        gather_34 = torch.ops.aten.gather.default(arg3_1, 1, expand_68);  expand_68 = None
        view_137 = torch.ops.aten.view.default(gather_34, [-1]);  gather_34 = None
        select_309 = torch.ops.aten.select.int(select_scatter_33, 1, 34)
        copy_34 = torch.ops.aten.copy.default(select_309, argmin_34);  select_309 = argmin_34 = None
        select_scatter_34 = torch.ops.aten.select_scatter.default(select_scatter_33, copy_34, 1, 34);  select_scatter_33 = copy_34 = None
        sub_34 = torch.ops.aten.sub.Tensor(select_306, view_137);  select_306 = view_137 = None
        div_34 = torch.ops.aten.div.Tensor(sub_34, select_305);  sub_34 = select_305 = None
        select_311 = torch.ops.aten.select.int(arg1_1, 0, 34)
        slice_134 = torch.ops.aten.slice.Tensor(select_311, 0, 34, 9223372036854775807);  select_311 = None
        slice_135 = torch.ops.aten.slice.Tensor(slice_scatter_32, 1, 34, 9223372036854775807)
        expand_69 = torch.ops.aten.expand.default(slice_135, [2048, 94]);  slice_135 = None
        mul_102 = torch.ops.aten.mul.Tensor(expand_69, 1);  expand_69 = None
        view_138 = torch.ops.aten.view.default(div_34, [2048, 1]);  div_34 = None
        mul_103 = torch.ops.aten.mul.Tensor(view_138, slice_134);  view_138 = slice_134 = None
        mul_104 = torch.ops.aten.mul.Tensor(mul_103, -1.0);  mul_103 = None
        add_34 = torch.ops.aten.add.Tensor(mul_102, mul_104);  mul_102 = mul_104 = None
        slice_scatter_33 = torch.ops.aten.slice_scatter.default(slice_scatter_32, add_34, 1, 34, 9223372036854775807);  slice_scatter_32 = add_34 = None
        select_313 = torch.ops.aten.select.int(arg1_1, 0, 35)
        select_314 = torch.ops.aten.select.int(select_313, 0, 35);  select_313 = None
        select_315 = torch.ops.aten.select.int(slice_scatter_33, 1, 35)
        unsqueeze_70 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_35 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_316 = torch.ops.aten.select.int(slice_scatter_33, 1, 35)
        view_140 = torch.ops.aten.view.default(select_316, [8, 128, 2]);  select_316 = None
        baddbmm_35 = torch.ops.aten.baddbmm.default(unsqueeze_70, view_140, permute_35, alpha = -2.0);  unsqueeze_70 = view_140 = permute_35 = None
        argmin_35 = torch.ops.aten.argmin.default(baddbmm_35, -1);  baddbmm_35 = None
        unsqueeze_71 = torch.ops.aten.unsqueeze.default(argmin_35, -1)
        expand_70 = torch.ops.aten.expand.default(unsqueeze_71, [8, 128, 2]);  unsqueeze_71 = None
        gather_35 = torch.ops.aten.gather.default(arg3_1, 1, expand_70);  expand_70 = None
        view_141 = torch.ops.aten.view.default(gather_35, [-1]);  gather_35 = None
        select_318 = torch.ops.aten.select.int(select_scatter_34, 1, 35)
        copy_35 = torch.ops.aten.copy.default(select_318, argmin_35);  select_318 = argmin_35 = None
        select_scatter_35 = torch.ops.aten.select_scatter.default(select_scatter_34, copy_35, 1, 35);  select_scatter_34 = copy_35 = None
        sub_35 = torch.ops.aten.sub.Tensor(select_315, view_141);  select_315 = view_141 = None
        div_35 = torch.ops.aten.div.Tensor(sub_35, select_314);  sub_35 = select_314 = None
        select_320 = torch.ops.aten.select.int(arg1_1, 0, 35)
        slice_138 = torch.ops.aten.slice.Tensor(select_320, 0, 35, 9223372036854775807);  select_320 = None
        slice_139 = torch.ops.aten.slice.Tensor(slice_scatter_33, 1, 35, 9223372036854775807)
        expand_71 = torch.ops.aten.expand.default(slice_139, [2048, 93]);  slice_139 = None
        mul_105 = torch.ops.aten.mul.Tensor(expand_71, 1);  expand_71 = None
        view_142 = torch.ops.aten.view.default(div_35, [2048, 1]);  div_35 = None
        mul_106 = torch.ops.aten.mul.Tensor(view_142, slice_138);  view_142 = slice_138 = None
        mul_107 = torch.ops.aten.mul.Tensor(mul_106, -1.0);  mul_106 = None
        add_35 = torch.ops.aten.add.Tensor(mul_105, mul_107);  mul_105 = mul_107 = None
        slice_scatter_34 = torch.ops.aten.slice_scatter.default(slice_scatter_33, add_35, 1, 35, 9223372036854775807);  slice_scatter_33 = add_35 = None
        select_322 = torch.ops.aten.select.int(arg1_1, 0, 36)
        select_323 = torch.ops.aten.select.int(select_322, 0, 36);  select_322 = None
        select_324 = torch.ops.aten.select.int(slice_scatter_34, 1, 36)
        unsqueeze_72 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_36 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_325 = torch.ops.aten.select.int(slice_scatter_34, 1, 36)
        view_144 = torch.ops.aten.view.default(select_325, [8, 128, 2]);  select_325 = None
        baddbmm_36 = torch.ops.aten.baddbmm.default(unsqueeze_72, view_144, permute_36, alpha = -2.0);  unsqueeze_72 = view_144 = permute_36 = None
        argmin_36 = torch.ops.aten.argmin.default(baddbmm_36, -1);  baddbmm_36 = None
        unsqueeze_73 = torch.ops.aten.unsqueeze.default(argmin_36, -1)
        expand_72 = torch.ops.aten.expand.default(unsqueeze_73, [8, 128, 2]);  unsqueeze_73 = None
        gather_36 = torch.ops.aten.gather.default(arg3_1, 1, expand_72);  expand_72 = None
        view_145 = torch.ops.aten.view.default(gather_36, [-1]);  gather_36 = None
        select_327 = torch.ops.aten.select.int(select_scatter_35, 1, 36)
        copy_36 = torch.ops.aten.copy.default(select_327, argmin_36);  select_327 = argmin_36 = None
        select_scatter_36 = torch.ops.aten.select_scatter.default(select_scatter_35, copy_36, 1, 36);  select_scatter_35 = copy_36 = None
        sub_36 = torch.ops.aten.sub.Tensor(select_324, view_145);  select_324 = view_145 = None
        div_36 = torch.ops.aten.div.Tensor(sub_36, select_323);  sub_36 = select_323 = None
        select_329 = torch.ops.aten.select.int(arg1_1, 0, 36)
        slice_142 = torch.ops.aten.slice.Tensor(select_329, 0, 36, 9223372036854775807);  select_329 = None
        slice_143 = torch.ops.aten.slice.Tensor(slice_scatter_34, 1, 36, 9223372036854775807)
        expand_73 = torch.ops.aten.expand.default(slice_143, [2048, 92]);  slice_143 = None
        mul_108 = torch.ops.aten.mul.Tensor(expand_73, 1);  expand_73 = None
        view_146 = torch.ops.aten.view.default(div_36, [2048, 1]);  div_36 = None
        mul_109 = torch.ops.aten.mul.Tensor(view_146, slice_142);  view_146 = slice_142 = None
        mul_110 = torch.ops.aten.mul.Tensor(mul_109, -1.0);  mul_109 = None
        add_36 = torch.ops.aten.add.Tensor(mul_108, mul_110);  mul_108 = mul_110 = None
        slice_scatter_35 = torch.ops.aten.slice_scatter.default(slice_scatter_34, add_36, 1, 36, 9223372036854775807);  slice_scatter_34 = add_36 = None
        select_331 = torch.ops.aten.select.int(arg1_1, 0, 37)
        select_332 = torch.ops.aten.select.int(select_331, 0, 37);  select_331 = None
        select_333 = torch.ops.aten.select.int(slice_scatter_35, 1, 37)
        unsqueeze_74 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_37 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_334 = torch.ops.aten.select.int(slice_scatter_35, 1, 37)
        view_148 = torch.ops.aten.view.default(select_334, [8, 128, 2]);  select_334 = None
        baddbmm_37 = torch.ops.aten.baddbmm.default(unsqueeze_74, view_148, permute_37, alpha = -2.0);  unsqueeze_74 = view_148 = permute_37 = None
        argmin_37 = torch.ops.aten.argmin.default(baddbmm_37, -1);  baddbmm_37 = None
        unsqueeze_75 = torch.ops.aten.unsqueeze.default(argmin_37, -1)
        expand_74 = torch.ops.aten.expand.default(unsqueeze_75, [8, 128, 2]);  unsqueeze_75 = None
        gather_37 = torch.ops.aten.gather.default(arg3_1, 1, expand_74);  expand_74 = None
        view_149 = torch.ops.aten.view.default(gather_37, [-1]);  gather_37 = None
        select_336 = torch.ops.aten.select.int(select_scatter_36, 1, 37)
        copy_37 = torch.ops.aten.copy.default(select_336, argmin_37);  select_336 = argmin_37 = None
        select_scatter_37 = torch.ops.aten.select_scatter.default(select_scatter_36, copy_37, 1, 37);  select_scatter_36 = copy_37 = None
        sub_37 = torch.ops.aten.sub.Tensor(select_333, view_149);  select_333 = view_149 = None
        div_37 = torch.ops.aten.div.Tensor(sub_37, select_332);  sub_37 = select_332 = None
        select_338 = torch.ops.aten.select.int(arg1_1, 0, 37)
        slice_146 = torch.ops.aten.slice.Tensor(select_338, 0, 37, 9223372036854775807);  select_338 = None
        slice_147 = torch.ops.aten.slice.Tensor(slice_scatter_35, 1, 37, 9223372036854775807)
        expand_75 = torch.ops.aten.expand.default(slice_147, [2048, 91]);  slice_147 = None
        mul_111 = torch.ops.aten.mul.Tensor(expand_75, 1);  expand_75 = None
        view_150 = torch.ops.aten.view.default(div_37, [2048, 1]);  div_37 = None
        mul_112 = torch.ops.aten.mul.Tensor(view_150, slice_146);  view_150 = slice_146 = None
        mul_113 = torch.ops.aten.mul.Tensor(mul_112, -1.0);  mul_112 = None
        add_37 = torch.ops.aten.add.Tensor(mul_111, mul_113);  mul_111 = mul_113 = None
        slice_scatter_36 = torch.ops.aten.slice_scatter.default(slice_scatter_35, add_37, 1, 37, 9223372036854775807);  slice_scatter_35 = add_37 = None
        select_340 = torch.ops.aten.select.int(arg1_1, 0, 38)
        select_341 = torch.ops.aten.select.int(select_340, 0, 38);  select_340 = None
        select_342 = torch.ops.aten.select.int(slice_scatter_36, 1, 38)
        unsqueeze_76 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_38 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_343 = torch.ops.aten.select.int(slice_scatter_36, 1, 38)
        view_152 = torch.ops.aten.view.default(select_343, [8, 128, 2]);  select_343 = None
        baddbmm_38 = torch.ops.aten.baddbmm.default(unsqueeze_76, view_152, permute_38, alpha = -2.0);  unsqueeze_76 = view_152 = permute_38 = None
        argmin_38 = torch.ops.aten.argmin.default(baddbmm_38, -1);  baddbmm_38 = None
        unsqueeze_77 = torch.ops.aten.unsqueeze.default(argmin_38, -1)
        expand_76 = torch.ops.aten.expand.default(unsqueeze_77, [8, 128, 2]);  unsqueeze_77 = None
        gather_38 = torch.ops.aten.gather.default(arg3_1, 1, expand_76);  expand_76 = None
        view_153 = torch.ops.aten.view.default(gather_38, [-1]);  gather_38 = None
        select_345 = torch.ops.aten.select.int(select_scatter_37, 1, 38)
        copy_38 = torch.ops.aten.copy.default(select_345, argmin_38);  select_345 = argmin_38 = None
        select_scatter_38 = torch.ops.aten.select_scatter.default(select_scatter_37, copy_38, 1, 38);  select_scatter_37 = copy_38 = None
        sub_38 = torch.ops.aten.sub.Tensor(select_342, view_153);  select_342 = view_153 = None
        div_38 = torch.ops.aten.div.Tensor(sub_38, select_341);  sub_38 = select_341 = None
        select_347 = torch.ops.aten.select.int(arg1_1, 0, 38)
        slice_150 = torch.ops.aten.slice.Tensor(select_347, 0, 38, 9223372036854775807);  select_347 = None
        slice_151 = torch.ops.aten.slice.Tensor(slice_scatter_36, 1, 38, 9223372036854775807)
        expand_77 = torch.ops.aten.expand.default(slice_151, [2048, 90]);  slice_151 = None
        mul_114 = torch.ops.aten.mul.Tensor(expand_77, 1);  expand_77 = None
        view_154 = torch.ops.aten.view.default(div_38, [2048, 1]);  div_38 = None
        mul_115 = torch.ops.aten.mul.Tensor(view_154, slice_150);  view_154 = slice_150 = None
        mul_116 = torch.ops.aten.mul.Tensor(mul_115, -1.0);  mul_115 = None
        add_38 = torch.ops.aten.add.Tensor(mul_114, mul_116);  mul_114 = mul_116 = None
        slice_scatter_37 = torch.ops.aten.slice_scatter.default(slice_scatter_36, add_38, 1, 38, 9223372036854775807);  slice_scatter_36 = add_38 = None
        select_349 = torch.ops.aten.select.int(arg1_1, 0, 39)
        select_350 = torch.ops.aten.select.int(select_349, 0, 39);  select_349 = None
        select_351 = torch.ops.aten.select.int(slice_scatter_37, 1, 39)
        unsqueeze_78 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_39 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_352 = torch.ops.aten.select.int(slice_scatter_37, 1, 39)
        view_156 = torch.ops.aten.view.default(select_352, [8, 128, 2]);  select_352 = None
        baddbmm_39 = torch.ops.aten.baddbmm.default(unsqueeze_78, view_156, permute_39, alpha = -2.0);  unsqueeze_78 = view_156 = permute_39 = None
        argmin_39 = torch.ops.aten.argmin.default(baddbmm_39, -1);  baddbmm_39 = None
        unsqueeze_79 = torch.ops.aten.unsqueeze.default(argmin_39, -1)
        expand_78 = torch.ops.aten.expand.default(unsqueeze_79, [8, 128, 2]);  unsqueeze_79 = None
        gather_39 = torch.ops.aten.gather.default(arg3_1, 1, expand_78);  expand_78 = None
        view_157 = torch.ops.aten.view.default(gather_39, [-1]);  gather_39 = None
        select_354 = torch.ops.aten.select.int(select_scatter_38, 1, 39)
        copy_39 = torch.ops.aten.copy.default(select_354, argmin_39);  select_354 = argmin_39 = None
        select_scatter_39 = torch.ops.aten.select_scatter.default(select_scatter_38, copy_39, 1, 39);  select_scatter_38 = copy_39 = None
        sub_39 = torch.ops.aten.sub.Tensor(select_351, view_157);  select_351 = view_157 = None
        div_39 = torch.ops.aten.div.Tensor(sub_39, select_350);  sub_39 = select_350 = None
        select_356 = torch.ops.aten.select.int(arg1_1, 0, 39)
        slice_154 = torch.ops.aten.slice.Tensor(select_356, 0, 39, 9223372036854775807);  select_356 = None
        slice_155 = torch.ops.aten.slice.Tensor(slice_scatter_37, 1, 39, 9223372036854775807)
        expand_79 = torch.ops.aten.expand.default(slice_155, [2048, 89]);  slice_155 = None
        mul_117 = torch.ops.aten.mul.Tensor(expand_79, 1);  expand_79 = None
        view_158 = torch.ops.aten.view.default(div_39, [2048, 1]);  div_39 = None
        mul_118 = torch.ops.aten.mul.Tensor(view_158, slice_154);  view_158 = slice_154 = None
        mul_119 = torch.ops.aten.mul.Tensor(mul_118, -1.0);  mul_118 = None
        add_39 = torch.ops.aten.add.Tensor(mul_117, mul_119);  mul_117 = mul_119 = None
        slice_scatter_38 = torch.ops.aten.slice_scatter.default(slice_scatter_37, add_39, 1, 39, 9223372036854775807);  slice_scatter_37 = add_39 = None
        select_358 = torch.ops.aten.select.int(arg1_1, 0, 40)
        select_359 = torch.ops.aten.select.int(select_358, 0, 40);  select_358 = None
        select_360 = torch.ops.aten.select.int(slice_scatter_38, 1, 40)
        unsqueeze_80 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_40 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_361 = torch.ops.aten.select.int(slice_scatter_38, 1, 40)
        view_160 = torch.ops.aten.view.default(select_361, [8, 128, 2]);  select_361 = None
        baddbmm_40 = torch.ops.aten.baddbmm.default(unsqueeze_80, view_160, permute_40, alpha = -2.0);  unsqueeze_80 = view_160 = permute_40 = None
        argmin_40 = torch.ops.aten.argmin.default(baddbmm_40, -1);  baddbmm_40 = None
        unsqueeze_81 = torch.ops.aten.unsqueeze.default(argmin_40, -1)
        expand_80 = torch.ops.aten.expand.default(unsqueeze_81, [8, 128, 2]);  unsqueeze_81 = None
        gather_40 = torch.ops.aten.gather.default(arg3_1, 1, expand_80);  expand_80 = None
        view_161 = torch.ops.aten.view.default(gather_40, [-1]);  gather_40 = None
        select_363 = torch.ops.aten.select.int(select_scatter_39, 1, 40)
        copy_40 = torch.ops.aten.copy.default(select_363, argmin_40);  select_363 = argmin_40 = None
        select_scatter_40 = torch.ops.aten.select_scatter.default(select_scatter_39, copy_40, 1, 40);  select_scatter_39 = copy_40 = None
        sub_40 = torch.ops.aten.sub.Tensor(select_360, view_161);  select_360 = view_161 = None
        div_40 = torch.ops.aten.div.Tensor(sub_40, select_359);  sub_40 = select_359 = None
        select_365 = torch.ops.aten.select.int(arg1_1, 0, 40)
        slice_158 = torch.ops.aten.slice.Tensor(select_365, 0, 40, 9223372036854775807);  select_365 = None
        slice_159 = torch.ops.aten.slice.Tensor(slice_scatter_38, 1, 40, 9223372036854775807)
        expand_81 = torch.ops.aten.expand.default(slice_159, [2048, 88]);  slice_159 = None
        mul_120 = torch.ops.aten.mul.Tensor(expand_81, 1);  expand_81 = None
        view_162 = torch.ops.aten.view.default(div_40, [2048, 1]);  div_40 = None
        mul_121 = torch.ops.aten.mul.Tensor(view_162, slice_158);  view_162 = slice_158 = None
        mul_122 = torch.ops.aten.mul.Tensor(mul_121, -1.0);  mul_121 = None
        add_40 = torch.ops.aten.add.Tensor(mul_120, mul_122);  mul_120 = mul_122 = None
        slice_scatter_39 = torch.ops.aten.slice_scatter.default(slice_scatter_38, add_40, 1, 40, 9223372036854775807);  slice_scatter_38 = add_40 = None
        select_367 = torch.ops.aten.select.int(arg1_1, 0, 41)
        select_368 = torch.ops.aten.select.int(select_367, 0, 41);  select_367 = None
        select_369 = torch.ops.aten.select.int(slice_scatter_39, 1, 41)
        unsqueeze_82 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_41 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_370 = torch.ops.aten.select.int(slice_scatter_39, 1, 41)
        view_164 = torch.ops.aten.view.default(select_370, [8, 128, 2]);  select_370 = None
        baddbmm_41 = torch.ops.aten.baddbmm.default(unsqueeze_82, view_164, permute_41, alpha = -2.0);  unsqueeze_82 = view_164 = permute_41 = None
        argmin_41 = torch.ops.aten.argmin.default(baddbmm_41, -1);  baddbmm_41 = None
        unsqueeze_83 = torch.ops.aten.unsqueeze.default(argmin_41, -1)
        expand_82 = torch.ops.aten.expand.default(unsqueeze_83, [8, 128, 2]);  unsqueeze_83 = None
        gather_41 = torch.ops.aten.gather.default(arg3_1, 1, expand_82);  expand_82 = None
        view_165 = torch.ops.aten.view.default(gather_41, [-1]);  gather_41 = None
        select_372 = torch.ops.aten.select.int(select_scatter_40, 1, 41)
        copy_41 = torch.ops.aten.copy.default(select_372, argmin_41);  select_372 = argmin_41 = None
        select_scatter_41 = torch.ops.aten.select_scatter.default(select_scatter_40, copy_41, 1, 41);  select_scatter_40 = copy_41 = None
        sub_41 = torch.ops.aten.sub.Tensor(select_369, view_165);  select_369 = view_165 = None
        div_41 = torch.ops.aten.div.Tensor(sub_41, select_368);  sub_41 = select_368 = None
        select_374 = torch.ops.aten.select.int(arg1_1, 0, 41)
        slice_162 = torch.ops.aten.slice.Tensor(select_374, 0, 41, 9223372036854775807);  select_374 = None
        slice_163 = torch.ops.aten.slice.Tensor(slice_scatter_39, 1, 41, 9223372036854775807)
        expand_83 = torch.ops.aten.expand.default(slice_163, [2048, 87]);  slice_163 = None
        mul_123 = torch.ops.aten.mul.Tensor(expand_83, 1);  expand_83 = None
        view_166 = torch.ops.aten.view.default(div_41, [2048, 1]);  div_41 = None
        mul_124 = torch.ops.aten.mul.Tensor(view_166, slice_162);  view_166 = slice_162 = None
        mul_125 = torch.ops.aten.mul.Tensor(mul_124, -1.0);  mul_124 = None
        add_41 = torch.ops.aten.add.Tensor(mul_123, mul_125);  mul_123 = mul_125 = None
        slice_scatter_40 = torch.ops.aten.slice_scatter.default(slice_scatter_39, add_41, 1, 41, 9223372036854775807);  slice_scatter_39 = add_41 = None
        select_376 = torch.ops.aten.select.int(arg1_1, 0, 42)
        select_377 = torch.ops.aten.select.int(select_376, 0, 42);  select_376 = None
        select_378 = torch.ops.aten.select.int(slice_scatter_40, 1, 42)
        unsqueeze_84 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_42 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_379 = torch.ops.aten.select.int(slice_scatter_40, 1, 42)
        view_168 = torch.ops.aten.view.default(select_379, [8, 128, 2]);  select_379 = None
        baddbmm_42 = torch.ops.aten.baddbmm.default(unsqueeze_84, view_168, permute_42, alpha = -2.0);  unsqueeze_84 = view_168 = permute_42 = None
        argmin_42 = torch.ops.aten.argmin.default(baddbmm_42, -1);  baddbmm_42 = None
        unsqueeze_85 = torch.ops.aten.unsqueeze.default(argmin_42, -1)
        expand_84 = torch.ops.aten.expand.default(unsqueeze_85, [8, 128, 2]);  unsqueeze_85 = None
        gather_42 = torch.ops.aten.gather.default(arg3_1, 1, expand_84);  expand_84 = None
        view_169 = torch.ops.aten.view.default(gather_42, [-1]);  gather_42 = None
        select_381 = torch.ops.aten.select.int(select_scatter_41, 1, 42)
        copy_42 = torch.ops.aten.copy.default(select_381, argmin_42);  select_381 = argmin_42 = None
        select_scatter_42 = torch.ops.aten.select_scatter.default(select_scatter_41, copy_42, 1, 42);  select_scatter_41 = copy_42 = None
        sub_42 = torch.ops.aten.sub.Tensor(select_378, view_169);  select_378 = view_169 = None
        div_42 = torch.ops.aten.div.Tensor(sub_42, select_377);  sub_42 = select_377 = None
        select_383 = torch.ops.aten.select.int(arg1_1, 0, 42)
        slice_166 = torch.ops.aten.slice.Tensor(select_383, 0, 42, 9223372036854775807);  select_383 = None
        slice_167 = torch.ops.aten.slice.Tensor(slice_scatter_40, 1, 42, 9223372036854775807)
        expand_85 = torch.ops.aten.expand.default(slice_167, [2048, 86]);  slice_167 = None
        mul_126 = torch.ops.aten.mul.Tensor(expand_85, 1);  expand_85 = None
        view_170 = torch.ops.aten.view.default(div_42, [2048, 1]);  div_42 = None
        mul_127 = torch.ops.aten.mul.Tensor(view_170, slice_166);  view_170 = slice_166 = None
        mul_128 = torch.ops.aten.mul.Tensor(mul_127, -1.0);  mul_127 = None
        add_42 = torch.ops.aten.add.Tensor(mul_126, mul_128);  mul_126 = mul_128 = None
        slice_scatter_41 = torch.ops.aten.slice_scatter.default(slice_scatter_40, add_42, 1, 42, 9223372036854775807);  slice_scatter_40 = add_42 = None
        select_385 = torch.ops.aten.select.int(arg1_1, 0, 43)
        select_386 = torch.ops.aten.select.int(select_385, 0, 43);  select_385 = None
        select_387 = torch.ops.aten.select.int(slice_scatter_41, 1, 43)
        unsqueeze_86 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_43 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_388 = torch.ops.aten.select.int(slice_scatter_41, 1, 43)
        view_172 = torch.ops.aten.view.default(select_388, [8, 128, 2]);  select_388 = None
        baddbmm_43 = torch.ops.aten.baddbmm.default(unsqueeze_86, view_172, permute_43, alpha = -2.0);  unsqueeze_86 = view_172 = permute_43 = None
        argmin_43 = torch.ops.aten.argmin.default(baddbmm_43, -1);  baddbmm_43 = None
        unsqueeze_87 = torch.ops.aten.unsqueeze.default(argmin_43, -1)
        expand_86 = torch.ops.aten.expand.default(unsqueeze_87, [8, 128, 2]);  unsqueeze_87 = None
        gather_43 = torch.ops.aten.gather.default(arg3_1, 1, expand_86);  expand_86 = None
        view_173 = torch.ops.aten.view.default(gather_43, [-1]);  gather_43 = None
        select_390 = torch.ops.aten.select.int(select_scatter_42, 1, 43)
        copy_43 = torch.ops.aten.copy.default(select_390, argmin_43);  select_390 = argmin_43 = None
        select_scatter_43 = torch.ops.aten.select_scatter.default(select_scatter_42, copy_43, 1, 43);  select_scatter_42 = copy_43 = None
        sub_43 = torch.ops.aten.sub.Tensor(select_387, view_173);  select_387 = view_173 = None
        div_43 = torch.ops.aten.div.Tensor(sub_43, select_386);  sub_43 = select_386 = None
        select_392 = torch.ops.aten.select.int(arg1_1, 0, 43)
        slice_170 = torch.ops.aten.slice.Tensor(select_392, 0, 43, 9223372036854775807);  select_392 = None
        slice_171 = torch.ops.aten.slice.Tensor(slice_scatter_41, 1, 43, 9223372036854775807)
        expand_87 = torch.ops.aten.expand.default(slice_171, [2048, 85]);  slice_171 = None
        mul_129 = torch.ops.aten.mul.Tensor(expand_87, 1);  expand_87 = None
        view_174 = torch.ops.aten.view.default(div_43, [2048, 1]);  div_43 = None
        mul_130 = torch.ops.aten.mul.Tensor(view_174, slice_170);  view_174 = slice_170 = None
        mul_131 = torch.ops.aten.mul.Tensor(mul_130, -1.0);  mul_130 = None
        add_43 = torch.ops.aten.add.Tensor(mul_129, mul_131);  mul_129 = mul_131 = None
        slice_scatter_42 = torch.ops.aten.slice_scatter.default(slice_scatter_41, add_43, 1, 43, 9223372036854775807);  slice_scatter_41 = add_43 = None
        select_394 = torch.ops.aten.select.int(arg1_1, 0, 44)
        select_395 = torch.ops.aten.select.int(select_394, 0, 44);  select_394 = None
        select_396 = torch.ops.aten.select.int(slice_scatter_42, 1, 44)
        unsqueeze_88 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_44 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_397 = torch.ops.aten.select.int(slice_scatter_42, 1, 44)
        view_176 = torch.ops.aten.view.default(select_397, [8, 128, 2]);  select_397 = None
        baddbmm_44 = torch.ops.aten.baddbmm.default(unsqueeze_88, view_176, permute_44, alpha = -2.0);  unsqueeze_88 = view_176 = permute_44 = None
        argmin_44 = torch.ops.aten.argmin.default(baddbmm_44, -1);  baddbmm_44 = None
        unsqueeze_89 = torch.ops.aten.unsqueeze.default(argmin_44, -1)
        expand_88 = torch.ops.aten.expand.default(unsqueeze_89, [8, 128, 2]);  unsqueeze_89 = None
        gather_44 = torch.ops.aten.gather.default(arg3_1, 1, expand_88);  expand_88 = None
        view_177 = torch.ops.aten.view.default(gather_44, [-1]);  gather_44 = None
        select_399 = torch.ops.aten.select.int(select_scatter_43, 1, 44)
        copy_44 = torch.ops.aten.copy.default(select_399, argmin_44);  select_399 = argmin_44 = None
        select_scatter_44 = torch.ops.aten.select_scatter.default(select_scatter_43, copy_44, 1, 44);  select_scatter_43 = copy_44 = None
        sub_44 = torch.ops.aten.sub.Tensor(select_396, view_177);  select_396 = view_177 = None
        div_44 = torch.ops.aten.div.Tensor(sub_44, select_395);  sub_44 = select_395 = None
        select_401 = torch.ops.aten.select.int(arg1_1, 0, 44)
        slice_174 = torch.ops.aten.slice.Tensor(select_401, 0, 44, 9223372036854775807);  select_401 = None
        slice_175 = torch.ops.aten.slice.Tensor(slice_scatter_42, 1, 44, 9223372036854775807)
        expand_89 = torch.ops.aten.expand.default(slice_175, [2048, 84]);  slice_175 = None
        mul_132 = torch.ops.aten.mul.Tensor(expand_89, 1);  expand_89 = None
        view_178 = torch.ops.aten.view.default(div_44, [2048, 1]);  div_44 = None
        mul_133 = torch.ops.aten.mul.Tensor(view_178, slice_174);  view_178 = slice_174 = None
        mul_134 = torch.ops.aten.mul.Tensor(mul_133, -1.0);  mul_133 = None
        add_44 = torch.ops.aten.add.Tensor(mul_132, mul_134);  mul_132 = mul_134 = None
        slice_scatter_43 = torch.ops.aten.slice_scatter.default(slice_scatter_42, add_44, 1, 44, 9223372036854775807);  slice_scatter_42 = add_44 = None
        select_403 = torch.ops.aten.select.int(arg1_1, 0, 45)
        select_404 = torch.ops.aten.select.int(select_403, 0, 45);  select_403 = None
        select_405 = torch.ops.aten.select.int(slice_scatter_43, 1, 45)
        unsqueeze_90 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_45 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_406 = torch.ops.aten.select.int(slice_scatter_43, 1, 45)
        view_180 = torch.ops.aten.view.default(select_406, [8, 128, 2]);  select_406 = None
        baddbmm_45 = torch.ops.aten.baddbmm.default(unsqueeze_90, view_180, permute_45, alpha = -2.0);  unsqueeze_90 = view_180 = permute_45 = None
        argmin_45 = torch.ops.aten.argmin.default(baddbmm_45, -1);  baddbmm_45 = None
        unsqueeze_91 = torch.ops.aten.unsqueeze.default(argmin_45, -1)
        expand_90 = torch.ops.aten.expand.default(unsqueeze_91, [8, 128, 2]);  unsqueeze_91 = None
        gather_45 = torch.ops.aten.gather.default(arg3_1, 1, expand_90);  expand_90 = None
        view_181 = torch.ops.aten.view.default(gather_45, [-1]);  gather_45 = None
        select_408 = torch.ops.aten.select.int(select_scatter_44, 1, 45)
        copy_45 = torch.ops.aten.copy.default(select_408, argmin_45);  select_408 = argmin_45 = None
        select_scatter_45 = torch.ops.aten.select_scatter.default(select_scatter_44, copy_45, 1, 45);  select_scatter_44 = copy_45 = None
        sub_45 = torch.ops.aten.sub.Tensor(select_405, view_181);  select_405 = view_181 = None
        div_45 = torch.ops.aten.div.Tensor(sub_45, select_404);  sub_45 = select_404 = None
        select_410 = torch.ops.aten.select.int(arg1_1, 0, 45)
        slice_178 = torch.ops.aten.slice.Tensor(select_410, 0, 45, 9223372036854775807);  select_410 = None
        slice_179 = torch.ops.aten.slice.Tensor(slice_scatter_43, 1, 45, 9223372036854775807)
        expand_91 = torch.ops.aten.expand.default(slice_179, [2048, 83]);  slice_179 = None
        mul_135 = torch.ops.aten.mul.Tensor(expand_91, 1);  expand_91 = None
        view_182 = torch.ops.aten.view.default(div_45, [2048, 1]);  div_45 = None
        mul_136 = torch.ops.aten.mul.Tensor(view_182, slice_178);  view_182 = slice_178 = None
        mul_137 = torch.ops.aten.mul.Tensor(mul_136, -1.0);  mul_136 = None
        add_45 = torch.ops.aten.add.Tensor(mul_135, mul_137);  mul_135 = mul_137 = None
        slice_scatter_44 = torch.ops.aten.slice_scatter.default(slice_scatter_43, add_45, 1, 45, 9223372036854775807);  slice_scatter_43 = add_45 = None
        select_412 = torch.ops.aten.select.int(arg1_1, 0, 46)
        select_413 = torch.ops.aten.select.int(select_412, 0, 46);  select_412 = None
        select_414 = torch.ops.aten.select.int(slice_scatter_44, 1, 46)
        unsqueeze_92 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_46 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_415 = torch.ops.aten.select.int(slice_scatter_44, 1, 46)
        view_184 = torch.ops.aten.view.default(select_415, [8, 128, 2]);  select_415 = None
        baddbmm_46 = torch.ops.aten.baddbmm.default(unsqueeze_92, view_184, permute_46, alpha = -2.0);  unsqueeze_92 = view_184 = permute_46 = None
        argmin_46 = torch.ops.aten.argmin.default(baddbmm_46, -1);  baddbmm_46 = None
        unsqueeze_93 = torch.ops.aten.unsqueeze.default(argmin_46, -1)
        expand_92 = torch.ops.aten.expand.default(unsqueeze_93, [8, 128, 2]);  unsqueeze_93 = None
        gather_46 = torch.ops.aten.gather.default(arg3_1, 1, expand_92);  expand_92 = None
        view_185 = torch.ops.aten.view.default(gather_46, [-1]);  gather_46 = None
        select_417 = torch.ops.aten.select.int(select_scatter_45, 1, 46)
        copy_46 = torch.ops.aten.copy.default(select_417, argmin_46);  select_417 = argmin_46 = None
        select_scatter_46 = torch.ops.aten.select_scatter.default(select_scatter_45, copy_46, 1, 46);  select_scatter_45 = copy_46 = None
        sub_46 = torch.ops.aten.sub.Tensor(select_414, view_185);  select_414 = view_185 = None
        div_46 = torch.ops.aten.div.Tensor(sub_46, select_413);  sub_46 = select_413 = None
        select_419 = torch.ops.aten.select.int(arg1_1, 0, 46)
        slice_182 = torch.ops.aten.slice.Tensor(select_419, 0, 46, 9223372036854775807);  select_419 = None
        slice_183 = torch.ops.aten.slice.Tensor(slice_scatter_44, 1, 46, 9223372036854775807)
        expand_93 = torch.ops.aten.expand.default(slice_183, [2048, 82]);  slice_183 = None
        mul_138 = torch.ops.aten.mul.Tensor(expand_93, 1);  expand_93 = None
        view_186 = torch.ops.aten.view.default(div_46, [2048, 1]);  div_46 = None
        mul_139 = torch.ops.aten.mul.Tensor(view_186, slice_182);  view_186 = slice_182 = None
        mul_140 = torch.ops.aten.mul.Tensor(mul_139, -1.0);  mul_139 = None
        add_46 = torch.ops.aten.add.Tensor(mul_138, mul_140);  mul_138 = mul_140 = None
        slice_scatter_45 = torch.ops.aten.slice_scatter.default(slice_scatter_44, add_46, 1, 46, 9223372036854775807);  slice_scatter_44 = add_46 = None
        select_421 = torch.ops.aten.select.int(arg1_1, 0, 47)
        select_422 = torch.ops.aten.select.int(select_421, 0, 47);  select_421 = None
        select_423 = torch.ops.aten.select.int(slice_scatter_45, 1, 47)
        unsqueeze_94 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_47 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_424 = torch.ops.aten.select.int(slice_scatter_45, 1, 47)
        view_188 = torch.ops.aten.view.default(select_424, [8, 128, 2]);  select_424 = None
        baddbmm_47 = torch.ops.aten.baddbmm.default(unsqueeze_94, view_188, permute_47, alpha = -2.0);  unsqueeze_94 = view_188 = permute_47 = None
        argmin_47 = torch.ops.aten.argmin.default(baddbmm_47, -1);  baddbmm_47 = None
        unsqueeze_95 = torch.ops.aten.unsqueeze.default(argmin_47, -1)
        expand_94 = torch.ops.aten.expand.default(unsqueeze_95, [8, 128, 2]);  unsqueeze_95 = None
        gather_47 = torch.ops.aten.gather.default(arg3_1, 1, expand_94);  expand_94 = None
        view_189 = torch.ops.aten.view.default(gather_47, [-1]);  gather_47 = None
        select_426 = torch.ops.aten.select.int(select_scatter_46, 1, 47)
        copy_47 = torch.ops.aten.copy.default(select_426, argmin_47);  select_426 = argmin_47 = None
        select_scatter_47 = torch.ops.aten.select_scatter.default(select_scatter_46, copy_47, 1, 47);  select_scatter_46 = copy_47 = None
        sub_47 = torch.ops.aten.sub.Tensor(select_423, view_189);  select_423 = view_189 = None
        div_47 = torch.ops.aten.div.Tensor(sub_47, select_422);  sub_47 = select_422 = None
        select_428 = torch.ops.aten.select.int(arg1_1, 0, 47)
        slice_186 = torch.ops.aten.slice.Tensor(select_428, 0, 47, 9223372036854775807);  select_428 = None
        slice_187 = torch.ops.aten.slice.Tensor(slice_scatter_45, 1, 47, 9223372036854775807)
        expand_95 = torch.ops.aten.expand.default(slice_187, [2048, 81]);  slice_187 = None
        mul_141 = torch.ops.aten.mul.Tensor(expand_95, 1);  expand_95 = None
        view_190 = torch.ops.aten.view.default(div_47, [2048, 1]);  div_47 = None
        mul_142 = torch.ops.aten.mul.Tensor(view_190, slice_186);  view_190 = slice_186 = None
        mul_143 = torch.ops.aten.mul.Tensor(mul_142, -1.0);  mul_142 = None
        add_47 = torch.ops.aten.add.Tensor(mul_141, mul_143);  mul_141 = mul_143 = None
        slice_scatter_46 = torch.ops.aten.slice_scatter.default(slice_scatter_45, add_47, 1, 47, 9223372036854775807);  slice_scatter_45 = add_47 = None
        select_430 = torch.ops.aten.select.int(arg1_1, 0, 48)
        select_431 = torch.ops.aten.select.int(select_430, 0, 48);  select_430 = None
        select_432 = torch.ops.aten.select.int(slice_scatter_46, 1, 48)
        unsqueeze_96 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_48 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_433 = torch.ops.aten.select.int(slice_scatter_46, 1, 48)
        view_192 = torch.ops.aten.view.default(select_433, [8, 128, 2]);  select_433 = None
        baddbmm_48 = torch.ops.aten.baddbmm.default(unsqueeze_96, view_192, permute_48, alpha = -2.0);  unsqueeze_96 = view_192 = permute_48 = None
        argmin_48 = torch.ops.aten.argmin.default(baddbmm_48, -1);  baddbmm_48 = None
        unsqueeze_97 = torch.ops.aten.unsqueeze.default(argmin_48, -1)
        expand_96 = torch.ops.aten.expand.default(unsqueeze_97, [8, 128, 2]);  unsqueeze_97 = None
        gather_48 = torch.ops.aten.gather.default(arg3_1, 1, expand_96);  expand_96 = None
        view_193 = torch.ops.aten.view.default(gather_48, [-1]);  gather_48 = None
        select_435 = torch.ops.aten.select.int(select_scatter_47, 1, 48)
        copy_48 = torch.ops.aten.copy.default(select_435, argmin_48);  select_435 = argmin_48 = None
        select_scatter_48 = torch.ops.aten.select_scatter.default(select_scatter_47, copy_48, 1, 48);  select_scatter_47 = copy_48 = None
        sub_48 = torch.ops.aten.sub.Tensor(select_432, view_193);  select_432 = view_193 = None
        div_48 = torch.ops.aten.div.Tensor(sub_48, select_431);  sub_48 = select_431 = None
        select_437 = torch.ops.aten.select.int(arg1_1, 0, 48)
        slice_190 = torch.ops.aten.slice.Tensor(select_437, 0, 48, 9223372036854775807);  select_437 = None
        slice_191 = torch.ops.aten.slice.Tensor(slice_scatter_46, 1, 48, 9223372036854775807)
        expand_97 = torch.ops.aten.expand.default(slice_191, [2048, 80]);  slice_191 = None
        mul_144 = torch.ops.aten.mul.Tensor(expand_97, 1);  expand_97 = None
        view_194 = torch.ops.aten.view.default(div_48, [2048, 1]);  div_48 = None
        mul_145 = torch.ops.aten.mul.Tensor(view_194, slice_190);  view_194 = slice_190 = None
        mul_146 = torch.ops.aten.mul.Tensor(mul_145, -1.0);  mul_145 = None
        add_48 = torch.ops.aten.add.Tensor(mul_144, mul_146);  mul_144 = mul_146 = None
        slice_scatter_47 = torch.ops.aten.slice_scatter.default(slice_scatter_46, add_48, 1, 48, 9223372036854775807);  slice_scatter_46 = add_48 = None
        select_439 = torch.ops.aten.select.int(arg1_1, 0, 49)
        select_440 = torch.ops.aten.select.int(select_439, 0, 49);  select_439 = None
        select_441 = torch.ops.aten.select.int(slice_scatter_47, 1, 49)
        unsqueeze_98 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_49 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_442 = torch.ops.aten.select.int(slice_scatter_47, 1, 49)
        view_196 = torch.ops.aten.view.default(select_442, [8, 128, 2]);  select_442 = None
        baddbmm_49 = torch.ops.aten.baddbmm.default(unsqueeze_98, view_196, permute_49, alpha = -2.0);  unsqueeze_98 = view_196 = permute_49 = None
        argmin_49 = torch.ops.aten.argmin.default(baddbmm_49, -1);  baddbmm_49 = None
        unsqueeze_99 = torch.ops.aten.unsqueeze.default(argmin_49, -1)
        expand_98 = torch.ops.aten.expand.default(unsqueeze_99, [8, 128, 2]);  unsqueeze_99 = None
        gather_49 = torch.ops.aten.gather.default(arg3_1, 1, expand_98);  expand_98 = None
        view_197 = torch.ops.aten.view.default(gather_49, [-1]);  gather_49 = None
        select_444 = torch.ops.aten.select.int(select_scatter_48, 1, 49)
        copy_49 = torch.ops.aten.copy.default(select_444, argmin_49);  select_444 = argmin_49 = None
        select_scatter_49 = torch.ops.aten.select_scatter.default(select_scatter_48, copy_49, 1, 49);  select_scatter_48 = copy_49 = None
        sub_49 = torch.ops.aten.sub.Tensor(select_441, view_197);  select_441 = view_197 = None
        div_49 = torch.ops.aten.div.Tensor(sub_49, select_440);  sub_49 = select_440 = None
        select_446 = torch.ops.aten.select.int(arg1_1, 0, 49)
        slice_194 = torch.ops.aten.slice.Tensor(select_446, 0, 49, 9223372036854775807);  select_446 = None
        slice_195 = torch.ops.aten.slice.Tensor(slice_scatter_47, 1, 49, 9223372036854775807)
        expand_99 = torch.ops.aten.expand.default(slice_195, [2048, 79]);  slice_195 = None
        mul_147 = torch.ops.aten.mul.Tensor(expand_99, 1);  expand_99 = None
        view_198 = torch.ops.aten.view.default(div_49, [2048, 1]);  div_49 = None
        mul_148 = torch.ops.aten.mul.Tensor(view_198, slice_194);  view_198 = slice_194 = None
        mul_149 = torch.ops.aten.mul.Tensor(mul_148, -1.0);  mul_148 = None
        add_49 = torch.ops.aten.add.Tensor(mul_147, mul_149);  mul_147 = mul_149 = None
        slice_scatter_48 = torch.ops.aten.slice_scatter.default(slice_scatter_47, add_49, 1, 49, 9223372036854775807);  slice_scatter_47 = add_49 = None
        select_448 = torch.ops.aten.select.int(arg1_1, 0, 50)
        select_449 = torch.ops.aten.select.int(select_448, 0, 50);  select_448 = None
        select_450 = torch.ops.aten.select.int(slice_scatter_48, 1, 50)
        unsqueeze_100 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_50 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_451 = torch.ops.aten.select.int(slice_scatter_48, 1, 50)
        view_200 = torch.ops.aten.view.default(select_451, [8, 128, 2]);  select_451 = None
        baddbmm_50 = torch.ops.aten.baddbmm.default(unsqueeze_100, view_200, permute_50, alpha = -2.0);  unsqueeze_100 = view_200 = permute_50 = None
        argmin_50 = torch.ops.aten.argmin.default(baddbmm_50, -1);  baddbmm_50 = None
        unsqueeze_101 = torch.ops.aten.unsqueeze.default(argmin_50, -1)
        expand_100 = torch.ops.aten.expand.default(unsqueeze_101, [8, 128, 2]);  unsqueeze_101 = None
        gather_50 = torch.ops.aten.gather.default(arg3_1, 1, expand_100);  expand_100 = None
        view_201 = torch.ops.aten.view.default(gather_50, [-1]);  gather_50 = None
        select_453 = torch.ops.aten.select.int(select_scatter_49, 1, 50)
        copy_50 = torch.ops.aten.copy.default(select_453, argmin_50);  select_453 = argmin_50 = None
        select_scatter_50 = torch.ops.aten.select_scatter.default(select_scatter_49, copy_50, 1, 50);  select_scatter_49 = copy_50 = None
        sub_50 = torch.ops.aten.sub.Tensor(select_450, view_201);  select_450 = view_201 = None
        div_50 = torch.ops.aten.div.Tensor(sub_50, select_449);  sub_50 = select_449 = None
        select_455 = torch.ops.aten.select.int(arg1_1, 0, 50)
        slice_198 = torch.ops.aten.slice.Tensor(select_455, 0, 50, 9223372036854775807);  select_455 = None
        slice_199 = torch.ops.aten.slice.Tensor(slice_scatter_48, 1, 50, 9223372036854775807)
        expand_101 = torch.ops.aten.expand.default(slice_199, [2048, 78]);  slice_199 = None
        mul_150 = torch.ops.aten.mul.Tensor(expand_101, 1);  expand_101 = None
        view_202 = torch.ops.aten.view.default(div_50, [2048, 1]);  div_50 = None
        mul_151 = torch.ops.aten.mul.Tensor(view_202, slice_198);  view_202 = slice_198 = None
        mul_152 = torch.ops.aten.mul.Tensor(mul_151, -1.0);  mul_151 = None
        add_50 = torch.ops.aten.add.Tensor(mul_150, mul_152);  mul_150 = mul_152 = None
        slice_scatter_49 = torch.ops.aten.slice_scatter.default(slice_scatter_48, add_50, 1, 50, 9223372036854775807);  slice_scatter_48 = add_50 = None
        select_457 = torch.ops.aten.select.int(arg1_1, 0, 51)
        select_458 = torch.ops.aten.select.int(select_457, 0, 51);  select_457 = None
        select_459 = torch.ops.aten.select.int(slice_scatter_49, 1, 51)
        unsqueeze_102 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_51 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_460 = torch.ops.aten.select.int(slice_scatter_49, 1, 51)
        view_204 = torch.ops.aten.view.default(select_460, [8, 128, 2]);  select_460 = None
        baddbmm_51 = torch.ops.aten.baddbmm.default(unsqueeze_102, view_204, permute_51, alpha = -2.0);  unsqueeze_102 = view_204 = permute_51 = None
        argmin_51 = torch.ops.aten.argmin.default(baddbmm_51, -1);  baddbmm_51 = None
        unsqueeze_103 = torch.ops.aten.unsqueeze.default(argmin_51, -1)
        expand_102 = torch.ops.aten.expand.default(unsqueeze_103, [8, 128, 2]);  unsqueeze_103 = None
        gather_51 = torch.ops.aten.gather.default(arg3_1, 1, expand_102);  expand_102 = None
        view_205 = torch.ops.aten.view.default(gather_51, [-1]);  gather_51 = None
        select_462 = torch.ops.aten.select.int(select_scatter_50, 1, 51)
        copy_51 = torch.ops.aten.copy.default(select_462, argmin_51);  select_462 = argmin_51 = None
        select_scatter_51 = torch.ops.aten.select_scatter.default(select_scatter_50, copy_51, 1, 51);  select_scatter_50 = copy_51 = None
        sub_51 = torch.ops.aten.sub.Tensor(select_459, view_205);  select_459 = view_205 = None
        div_51 = torch.ops.aten.div.Tensor(sub_51, select_458);  sub_51 = select_458 = None
        select_464 = torch.ops.aten.select.int(arg1_1, 0, 51)
        slice_202 = torch.ops.aten.slice.Tensor(select_464, 0, 51, 9223372036854775807);  select_464 = None
        slice_203 = torch.ops.aten.slice.Tensor(slice_scatter_49, 1, 51, 9223372036854775807)
        expand_103 = torch.ops.aten.expand.default(slice_203, [2048, 77]);  slice_203 = None
        mul_153 = torch.ops.aten.mul.Tensor(expand_103, 1);  expand_103 = None
        view_206 = torch.ops.aten.view.default(div_51, [2048, 1]);  div_51 = None
        mul_154 = torch.ops.aten.mul.Tensor(view_206, slice_202);  view_206 = slice_202 = None
        mul_155 = torch.ops.aten.mul.Tensor(mul_154, -1.0);  mul_154 = None
        add_51 = torch.ops.aten.add.Tensor(mul_153, mul_155);  mul_153 = mul_155 = None
        slice_scatter_50 = torch.ops.aten.slice_scatter.default(slice_scatter_49, add_51, 1, 51, 9223372036854775807);  slice_scatter_49 = add_51 = None
        select_466 = torch.ops.aten.select.int(arg1_1, 0, 52)
        select_467 = torch.ops.aten.select.int(select_466, 0, 52);  select_466 = None
        select_468 = torch.ops.aten.select.int(slice_scatter_50, 1, 52)
        unsqueeze_104 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_52 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_469 = torch.ops.aten.select.int(slice_scatter_50, 1, 52)
        view_208 = torch.ops.aten.view.default(select_469, [8, 128, 2]);  select_469 = None
        baddbmm_52 = torch.ops.aten.baddbmm.default(unsqueeze_104, view_208, permute_52, alpha = -2.0);  unsqueeze_104 = view_208 = permute_52 = None
        argmin_52 = torch.ops.aten.argmin.default(baddbmm_52, -1);  baddbmm_52 = None
        unsqueeze_105 = torch.ops.aten.unsqueeze.default(argmin_52, -1)
        expand_104 = torch.ops.aten.expand.default(unsqueeze_105, [8, 128, 2]);  unsqueeze_105 = None
        gather_52 = torch.ops.aten.gather.default(arg3_1, 1, expand_104);  expand_104 = None
        view_209 = torch.ops.aten.view.default(gather_52, [-1]);  gather_52 = None
        select_471 = torch.ops.aten.select.int(select_scatter_51, 1, 52)
        copy_52 = torch.ops.aten.copy.default(select_471, argmin_52);  select_471 = argmin_52 = None
        select_scatter_52 = torch.ops.aten.select_scatter.default(select_scatter_51, copy_52, 1, 52);  select_scatter_51 = copy_52 = None
        sub_52 = torch.ops.aten.sub.Tensor(select_468, view_209);  select_468 = view_209 = None
        div_52 = torch.ops.aten.div.Tensor(sub_52, select_467);  sub_52 = select_467 = None
        select_473 = torch.ops.aten.select.int(arg1_1, 0, 52)
        slice_206 = torch.ops.aten.slice.Tensor(select_473, 0, 52, 9223372036854775807);  select_473 = None
        slice_207 = torch.ops.aten.slice.Tensor(slice_scatter_50, 1, 52, 9223372036854775807)
        expand_105 = torch.ops.aten.expand.default(slice_207, [2048, 76]);  slice_207 = None
        mul_156 = torch.ops.aten.mul.Tensor(expand_105, 1);  expand_105 = None
        view_210 = torch.ops.aten.view.default(div_52, [2048, 1]);  div_52 = None
        mul_157 = torch.ops.aten.mul.Tensor(view_210, slice_206);  view_210 = slice_206 = None
        mul_158 = torch.ops.aten.mul.Tensor(mul_157, -1.0);  mul_157 = None
        add_52 = torch.ops.aten.add.Tensor(mul_156, mul_158);  mul_156 = mul_158 = None
        slice_scatter_51 = torch.ops.aten.slice_scatter.default(slice_scatter_50, add_52, 1, 52, 9223372036854775807);  slice_scatter_50 = add_52 = None
        select_475 = torch.ops.aten.select.int(arg1_1, 0, 53)
        select_476 = torch.ops.aten.select.int(select_475, 0, 53);  select_475 = None
        select_477 = torch.ops.aten.select.int(slice_scatter_51, 1, 53)
        unsqueeze_106 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_53 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_478 = torch.ops.aten.select.int(slice_scatter_51, 1, 53)
        view_212 = torch.ops.aten.view.default(select_478, [8, 128, 2]);  select_478 = None
        baddbmm_53 = torch.ops.aten.baddbmm.default(unsqueeze_106, view_212, permute_53, alpha = -2.0);  unsqueeze_106 = view_212 = permute_53 = None
        argmin_53 = torch.ops.aten.argmin.default(baddbmm_53, -1);  baddbmm_53 = None
        unsqueeze_107 = torch.ops.aten.unsqueeze.default(argmin_53, -1)
        expand_106 = torch.ops.aten.expand.default(unsqueeze_107, [8, 128, 2]);  unsqueeze_107 = None
        gather_53 = torch.ops.aten.gather.default(arg3_1, 1, expand_106);  expand_106 = None
        view_213 = torch.ops.aten.view.default(gather_53, [-1]);  gather_53 = None
        select_480 = torch.ops.aten.select.int(select_scatter_52, 1, 53)
        copy_53 = torch.ops.aten.copy.default(select_480, argmin_53);  select_480 = argmin_53 = None
        select_scatter_53 = torch.ops.aten.select_scatter.default(select_scatter_52, copy_53, 1, 53);  select_scatter_52 = copy_53 = None
        sub_53 = torch.ops.aten.sub.Tensor(select_477, view_213);  select_477 = view_213 = None
        div_53 = torch.ops.aten.div.Tensor(sub_53, select_476);  sub_53 = select_476 = None
        select_482 = torch.ops.aten.select.int(arg1_1, 0, 53)
        slice_210 = torch.ops.aten.slice.Tensor(select_482, 0, 53, 9223372036854775807);  select_482 = None
        slice_211 = torch.ops.aten.slice.Tensor(slice_scatter_51, 1, 53, 9223372036854775807)
        expand_107 = torch.ops.aten.expand.default(slice_211, [2048, 75]);  slice_211 = None
        mul_159 = torch.ops.aten.mul.Tensor(expand_107, 1);  expand_107 = None
        view_214 = torch.ops.aten.view.default(div_53, [2048, 1]);  div_53 = None
        mul_160 = torch.ops.aten.mul.Tensor(view_214, slice_210);  view_214 = slice_210 = None
        mul_161 = torch.ops.aten.mul.Tensor(mul_160, -1.0);  mul_160 = None
        add_53 = torch.ops.aten.add.Tensor(mul_159, mul_161);  mul_159 = mul_161 = None
        slice_scatter_52 = torch.ops.aten.slice_scatter.default(slice_scatter_51, add_53, 1, 53, 9223372036854775807);  slice_scatter_51 = add_53 = None
        select_484 = torch.ops.aten.select.int(arg1_1, 0, 54)
        select_485 = torch.ops.aten.select.int(select_484, 0, 54);  select_484 = None
        select_486 = torch.ops.aten.select.int(slice_scatter_52, 1, 54)
        unsqueeze_108 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_54 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_487 = torch.ops.aten.select.int(slice_scatter_52, 1, 54)
        view_216 = torch.ops.aten.view.default(select_487, [8, 128, 2]);  select_487 = None
        baddbmm_54 = torch.ops.aten.baddbmm.default(unsqueeze_108, view_216, permute_54, alpha = -2.0);  unsqueeze_108 = view_216 = permute_54 = None
        argmin_54 = torch.ops.aten.argmin.default(baddbmm_54, -1);  baddbmm_54 = None
        unsqueeze_109 = torch.ops.aten.unsqueeze.default(argmin_54, -1)
        expand_108 = torch.ops.aten.expand.default(unsqueeze_109, [8, 128, 2]);  unsqueeze_109 = None
        gather_54 = torch.ops.aten.gather.default(arg3_1, 1, expand_108);  expand_108 = None
        view_217 = torch.ops.aten.view.default(gather_54, [-1]);  gather_54 = None
        select_489 = torch.ops.aten.select.int(select_scatter_53, 1, 54)
        copy_54 = torch.ops.aten.copy.default(select_489, argmin_54);  select_489 = argmin_54 = None
        select_scatter_54 = torch.ops.aten.select_scatter.default(select_scatter_53, copy_54, 1, 54);  select_scatter_53 = copy_54 = None
        sub_54 = torch.ops.aten.sub.Tensor(select_486, view_217);  select_486 = view_217 = None
        div_54 = torch.ops.aten.div.Tensor(sub_54, select_485);  sub_54 = select_485 = None
        select_491 = torch.ops.aten.select.int(arg1_1, 0, 54)
        slice_214 = torch.ops.aten.slice.Tensor(select_491, 0, 54, 9223372036854775807);  select_491 = None
        slice_215 = torch.ops.aten.slice.Tensor(slice_scatter_52, 1, 54, 9223372036854775807)
        expand_109 = torch.ops.aten.expand.default(slice_215, [2048, 74]);  slice_215 = None
        mul_162 = torch.ops.aten.mul.Tensor(expand_109, 1);  expand_109 = None
        view_218 = torch.ops.aten.view.default(div_54, [2048, 1]);  div_54 = None
        mul_163 = torch.ops.aten.mul.Tensor(view_218, slice_214);  view_218 = slice_214 = None
        mul_164 = torch.ops.aten.mul.Tensor(mul_163, -1.0);  mul_163 = None
        add_54 = torch.ops.aten.add.Tensor(mul_162, mul_164);  mul_162 = mul_164 = None
        slice_scatter_53 = torch.ops.aten.slice_scatter.default(slice_scatter_52, add_54, 1, 54, 9223372036854775807);  slice_scatter_52 = add_54 = None
        select_493 = torch.ops.aten.select.int(arg1_1, 0, 55)
        select_494 = torch.ops.aten.select.int(select_493, 0, 55);  select_493 = None
        select_495 = torch.ops.aten.select.int(slice_scatter_53, 1, 55)
        unsqueeze_110 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_55 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_496 = torch.ops.aten.select.int(slice_scatter_53, 1, 55)
        view_220 = torch.ops.aten.view.default(select_496, [8, 128, 2]);  select_496 = None
        baddbmm_55 = torch.ops.aten.baddbmm.default(unsqueeze_110, view_220, permute_55, alpha = -2.0);  unsqueeze_110 = view_220 = permute_55 = None
        argmin_55 = torch.ops.aten.argmin.default(baddbmm_55, -1);  baddbmm_55 = None
        unsqueeze_111 = torch.ops.aten.unsqueeze.default(argmin_55, -1)
        expand_110 = torch.ops.aten.expand.default(unsqueeze_111, [8, 128, 2]);  unsqueeze_111 = None
        gather_55 = torch.ops.aten.gather.default(arg3_1, 1, expand_110);  expand_110 = None
        view_221 = torch.ops.aten.view.default(gather_55, [-1]);  gather_55 = None
        select_498 = torch.ops.aten.select.int(select_scatter_54, 1, 55)
        copy_55 = torch.ops.aten.copy.default(select_498, argmin_55);  select_498 = argmin_55 = None
        select_scatter_55 = torch.ops.aten.select_scatter.default(select_scatter_54, copy_55, 1, 55);  select_scatter_54 = copy_55 = None
        sub_55 = torch.ops.aten.sub.Tensor(select_495, view_221);  select_495 = view_221 = None
        div_55 = torch.ops.aten.div.Tensor(sub_55, select_494);  sub_55 = select_494 = None
        select_500 = torch.ops.aten.select.int(arg1_1, 0, 55)
        slice_218 = torch.ops.aten.slice.Tensor(select_500, 0, 55, 9223372036854775807);  select_500 = None
        slice_219 = torch.ops.aten.slice.Tensor(slice_scatter_53, 1, 55, 9223372036854775807)
        expand_111 = torch.ops.aten.expand.default(slice_219, [2048, 73]);  slice_219 = None
        mul_165 = torch.ops.aten.mul.Tensor(expand_111, 1);  expand_111 = None
        view_222 = torch.ops.aten.view.default(div_55, [2048, 1]);  div_55 = None
        mul_166 = torch.ops.aten.mul.Tensor(view_222, slice_218);  view_222 = slice_218 = None
        mul_167 = torch.ops.aten.mul.Tensor(mul_166, -1.0);  mul_166 = None
        add_55 = torch.ops.aten.add.Tensor(mul_165, mul_167);  mul_165 = mul_167 = None
        slice_scatter_54 = torch.ops.aten.slice_scatter.default(slice_scatter_53, add_55, 1, 55, 9223372036854775807);  slice_scatter_53 = add_55 = None
        select_502 = torch.ops.aten.select.int(arg1_1, 0, 56)
        select_503 = torch.ops.aten.select.int(select_502, 0, 56);  select_502 = None
        select_504 = torch.ops.aten.select.int(slice_scatter_54, 1, 56)
        unsqueeze_112 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_56 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_505 = torch.ops.aten.select.int(slice_scatter_54, 1, 56)
        view_224 = torch.ops.aten.view.default(select_505, [8, 128, 2]);  select_505 = None
        baddbmm_56 = torch.ops.aten.baddbmm.default(unsqueeze_112, view_224, permute_56, alpha = -2.0);  unsqueeze_112 = view_224 = permute_56 = None
        argmin_56 = torch.ops.aten.argmin.default(baddbmm_56, -1);  baddbmm_56 = None
        unsqueeze_113 = torch.ops.aten.unsqueeze.default(argmin_56, -1)
        expand_112 = torch.ops.aten.expand.default(unsqueeze_113, [8, 128, 2]);  unsqueeze_113 = None
        gather_56 = torch.ops.aten.gather.default(arg3_1, 1, expand_112);  expand_112 = None
        view_225 = torch.ops.aten.view.default(gather_56, [-1]);  gather_56 = None
        select_507 = torch.ops.aten.select.int(select_scatter_55, 1, 56)
        copy_56 = torch.ops.aten.copy.default(select_507, argmin_56);  select_507 = argmin_56 = None
        select_scatter_56 = torch.ops.aten.select_scatter.default(select_scatter_55, copy_56, 1, 56);  select_scatter_55 = copy_56 = None
        sub_56 = torch.ops.aten.sub.Tensor(select_504, view_225);  select_504 = view_225 = None
        div_56 = torch.ops.aten.div.Tensor(sub_56, select_503);  sub_56 = select_503 = None
        select_509 = torch.ops.aten.select.int(arg1_1, 0, 56)
        slice_222 = torch.ops.aten.slice.Tensor(select_509, 0, 56, 9223372036854775807);  select_509 = None
        slice_223 = torch.ops.aten.slice.Tensor(slice_scatter_54, 1, 56, 9223372036854775807)
        expand_113 = torch.ops.aten.expand.default(slice_223, [2048, 72]);  slice_223 = None
        mul_168 = torch.ops.aten.mul.Tensor(expand_113, 1);  expand_113 = None
        view_226 = torch.ops.aten.view.default(div_56, [2048, 1]);  div_56 = None
        mul_169 = torch.ops.aten.mul.Tensor(view_226, slice_222);  view_226 = slice_222 = None
        mul_170 = torch.ops.aten.mul.Tensor(mul_169, -1.0);  mul_169 = None
        add_56 = torch.ops.aten.add.Tensor(mul_168, mul_170);  mul_168 = mul_170 = None
        slice_scatter_55 = torch.ops.aten.slice_scatter.default(slice_scatter_54, add_56, 1, 56, 9223372036854775807);  slice_scatter_54 = add_56 = None
        select_511 = torch.ops.aten.select.int(arg1_1, 0, 57)
        select_512 = torch.ops.aten.select.int(select_511, 0, 57);  select_511 = None
        select_513 = torch.ops.aten.select.int(slice_scatter_55, 1, 57)
        unsqueeze_114 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_57 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_514 = torch.ops.aten.select.int(slice_scatter_55, 1, 57)
        view_228 = torch.ops.aten.view.default(select_514, [8, 128, 2]);  select_514 = None
        baddbmm_57 = torch.ops.aten.baddbmm.default(unsqueeze_114, view_228, permute_57, alpha = -2.0);  unsqueeze_114 = view_228 = permute_57 = None
        argmin_57 = torch.ops.aten.argmin.default(baddbmm_57, -1);  baddbmm_57 = None
        unsqueeze_115 = torch.ops.aten.unsqueeze.default(argmin_57, -1)
        expand_114 = torch.ops.aten.expand.default(unsqueeze_115, [8, 128, 2]);  unsqueeze_115 = None
        gather_57 = torch.ops.aten.gather.default(arg3_1, 1, expand_114);  expand_114 = None
        view_229 = torch.ops.aten.view.default(gather_57, [-1]);  gather_57 = None
        select_516 = torch.ops.aten.select.int(select_scatter_56, 1, 57)
        copy_57 = torch.ops.aten.copy.default(select_516, argmin_57);  select_516 = argmin_57 = None
        select_scatter_57 = torch.ops.aten.select_scatter.default(select_scatter_56, copy_57, 1, 57);  select_scatter_56 = copy_57 = None
        sub_57 = torch.ops.aten.sub.Tensor(select_513, view_229);  select_513 = view_229 = None
        div_57 = torch.ops.aten.div.Tensor(sub_57, select_512);  sub_57 = select_512 = None
        select_518 = torch.ops.aten.select.int(arg1_1, 0, 57)
        slice_226 = torch.ops.aten.slice.Tensor(select_518, 0, 57, 9223372036854775807);  select_518 = None
        slice_227 = torch.ops.aten.slice.Tensor(slice_scatter_55, 1, 57, 9223372036854775807)
        expand_115 = torch.ops.aten.expand.default(slice_227, [2048, 71]);  slice_227 = None
        mul_171 = torch.ops.aten.mul.Tensor(expand_115, 1);  expand_115 = None
        view_230 = torch.ops.aten.view.default(div_57, [2048, 1]);  div_57 = None
        mul_172 = torch.ops.aten.mul.Tensor(view_230, slice_226);  view_230 = slice_226 = None
        mul_173 = torch.ops.aten.mul.Tensor(mul_172, -1.0);  mul_172 = None
        add_57 = torch.ops.aten.add.Tensor(mul_171, mul_173);  mul_171 = mul_173 = None
        slice_scatter_56 = torch.ops.aten.slice_scatter.default(slice_scatter_55, add_57, 1, 57, 9223372036854775807);  slice_scatter_55 = add_57 = None
        select_520 = torch.ops.aten.select.int(arg1_1, 0, 58)
        select_521 = torch.ops.aten.select.int(select_520, 0, 58);  select_520 = None
        select_522 = torch.ops.aten.select.int(slice_scatter_56, 1, 58)
        unsqueeze_116 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_58 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_523 = torch.ops.aten.select.int(slice_scatter_56, 1, 58)
        view_232 = torch.ops.aten.view.default(select_523, [8, 128, 2]);  select_523 = None
        baddbmm_58 = torch.ops.aten.baddbmm.default(unsqueeze_116, view_232, permute_58, alpha = -2.0);  unsqueeze_116 = view_232 = permute_58 = None
        argmin_58 = torch.ops.aten.argmin.default(baddbmm_58, -1);  baddbmm_58 = None
        unsqueeze_117 = torch.ops.aten.unsqueeze.default(argmin_58, -1)
        expand_116 = torch.ops.aten.expand.default(unsqueeze_117, [8, 128, 2]);  unsqueeze_117 = None
        gather_58 = torch.ops.aten.gather.default(arg3_1, 1, expand_116);  expand_116 = None
        view_233 = torch.ops.aten.view.default(gather_58, [-1]);  gather_58 = None
        select_525 = torch.ops.aten.select.int(select_scatter_57, 1, 58)
        copy_58 = torch.ops.aten.copy.default(select_525, argmin_58);  select_525 = argmin_58 = None
        select_scatter_58 = torch.ops.aten.select_scatter.default(select_scatter_57, copy_58, 1, 58);  select_scatter_57 = copy_58 = None
        sub_58 = torch.ops.aten.sub.Tensor(select_522, view_233);  select_522 = view_233 = None
        div_58 = torch.ops.aten.div.Tensor(sub_58, select_521);  sub_58 = select_521 = None
        select_527 = torch.ops.aten.select.int(arg1_1, 0, 58)
        slice_230 = torch.ops.aten.slice.Tensor(select_527, 0, 58, 9223372036854775807);  select_527 = None
        slice_231 = torch.ops.aten.slice.Tensor(slice_scatter_56, 1, 58, 9223372036854775807)
        expand_117 = torch.ops.aten.expand.default(slice_231, [2048, 70]);  slice_231 = None
        mul_174 = torch.ops.aten.mul.Tensor(expand_117, 1);  expand_117 = None
        view_234 = torch.ops.aten.view.default(div_58, [2048, 1]);  div_58 = None
        mul_175 = torch.ops.aten.mul.Tensor(view_234, slice_230);  view_234 = slice_230 = None
        mul_176 = torch.ops.aten.mul.Tensor(mul_175, -1.0);  mul_175 = None
        add_58 = torch.ops.aten.add.Tensor(mul_174, mul_176);  mul_174 = mul_176 = None
        slice_scatter_57 = torch.ops.aten.slice_scatter.default(slice_scatter_56, add_58, 1, 58, 9223372036854775807);  slice_scatter_56 = add_58 = None
        select_529 = torch.ops.aten.select.int(arg1_1, 0, 59)
        select_530 = torch.ops.aten.select.int(select_529, 0, 59);  select_529 = None
        select_531 = torch.ops.aten.select.int(slice_scatter_57, 1, 59)
        unsqueeze_118 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_59 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_532 = torch.ops.aten.select.int(slice_scatter_57, 1, 59)
        view_236 = torch.ops.aten.view.default(select_532, [8, 128, 2]);  select_532 = None
        baddbmm_59 = torch.ops.aten.baddbmm.default(unsqueeze_118, view_236, permute_59, alpha = -2.0);  unsqueeze_118 = view_236 = permute_59 = None
        argmin_59 = torch.ops.aten.argmin.default(baddbmm_59, -1);  baddbmm_59 = None
        unsqueeze_119 = torch.ops.aten.unsqueeze.default(argmin_59, -1)
        expand_118 = torch.ops.aten.expand.default(unsqueeze_119, [8, 128, 2]);  unsqueeze_119 = None
        gather_59 = torch.ops.aten.gather.default(arg3_1, 1, expand_118);  expand_118 = None
        view_237 = torch.ops.aten.view.default(gather_59, [-1]);  gather_59 = None
        select_534 = torch.ops.aten.select.int(select_scatter_58, 1, 59)
        copy_59 = torch.ops.aten.copy.default(select_534, argmin_59);  select_534 = argmin_59 = None
        select_scatter_59 = torch.ops.aten.select_scatter.default(select_scatter_58, copy_59, 1, 59);  select_scatter_58 = copy_59 = None
        sub_59 = torch.ops.aten.sub.Tensor(select_531, view_237);  select_531 = view_237 = None
        div_59 = torch.ops.aten.div.Tensor(sub_59, select_530);  sub_59 = select_530 = None
        select_536 = torch.ops.aten.select.int(arg1_1, 0, 59)
        slice_234 = torch.ops.aten.slice.Tensor(select_536, 0, 59, 9223372036854775807);  select_536 = None
        slice_235 = torch.ops.aten.slice.Tensor(slice_scatter_57, 1, 59, 9223372036854775807)
        expand_119 = torch.ops.aten.expand.default(slice_235, [2048, 69]);  slice_235 = None
        mul_177 = torch.ops.aten.mul.Tensor(expand_119, 1);  expand_119 = None
        view_238 = torch.ops.aten.view.default(div_59, [2048, 1]);  div_59 = None
        mul_178 = torch.ops.aten.mul.Tensor(view_238, slice_234);  view_238 = slice_234 = None
        mul_179 = torch.ops.aten.mul.Tensor(mul_178, -1.0);  mul_178 = None
        add_59 = torch.ops.aten.add.Tensor(mul_177, mul_179);  mul_177 = mul_179 = None
        slice_scatter_58 = torch.ops.aten.slice_scatter.default(slice_scatter_57, add_59, 1, 59, 9223372036854775807);  slice_scatter_57 = add_59 = None
        select_538 = torch.ops.aten.select.int(arg1_1, 0, 60)
        select_539 = torch.ops.aten.select.int(select_538, 0, 60);  select_538 = None
        select_540 = torch.ops.aten.select.int(slice_scatter_58, 1, 60)
        unsqueeze_120 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_60 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_541 = torch.ops.aten.select.int(slice_scatter_58, 1, 60)
        view_240 = torch.ops.aten.view.default(select_541, [8, 128, 2]);  select_541 = None
        baddbmm_60 = torch.ops.aten.baddbmm.default(unsqueeze_120, view_240, permute_60, alpha = -2.0);  unsqueeze_120 = view_240 = permute_60 = None
        argmin_60 = torch.ops.aten.argmin.default(baddbmm_60, -1);  baddbmm_60 = None
        unsqueeze_121 = torch.ops.aten.unsqueeze.default(argmin_60, -1)
        expand_120 = torch.ops.aten.expand.default(unsqueeze_121, [8, 128, 2]);  unsqueeze_121 = None
        gather_60 = torch.ops.aten.gather.default(arg3_1, 1, expand_120);  expand_120 = None
        view_241 = torch.ops.aten.view.default(gather_60, [-1]);  gather_60 = None
        select_543 = torch.ops.aten.select.int(select_scatter_59, 1, 60)
        copy_60 = torch.ops.aten.copy.default(select_543, argmin_60);  select_543 = argmin_60 = None
        select_scatter_60 = torch.ops.aten.select_scatter.default(select_scatter_59, copy_60, 1, 60);  select_scatter_59 = copy_60 = None
        sub_60 = torch.ops.aten.sub.Tensor(select_540, view_241);  select_540 = view_241 = None
        div_60 = torch.ops.aten.div.Tensor(sub_60, select_539);  sub_60 = select_539 = None
        select_545 = torch.ops.aten.select.int(arg1_1, 0, 60)
        slice_238 = torch.ops.aten.slice.Tensor(select_545, 0, 60, 9223372036854775807);  select_545 = None
        slice_239 = torch.ops.aten.slice.Tensor(slice_scatter_58, 1, 60, 9223372036854775807)
        expand_121 = torch.ops.aten.expand.default(slice_239, [2048, 68]);  slice_239 = None
        mul_180 = torch.ops.aten.mul.Tensor(expand_121, 1);  expand_121 = None
        view_242 = torch.ops.aten.view.default(div_60, [2048, 1]);  div_60 = None
        mul_181 = torch.ops.aten.mul.Tensor(view_242, slice_238);  view_242 = slice_238 = None
        mul_182 = torch.ops.aten.mul.Tensor(mul_181, -1.0);  mul_181 = None
        add_60 = torch.ops.aten.add.Tensor(mul_180, mul_182);  mul_180 = mul_182 = None
        slice_scatter_59 = torch.ops.aten.slice_scatter.default(slice_scatter_58, add_60, 1, 60, 9223372036854775807);  slice_scatter_58 = add_60 = None
        select_547 = torch.ops.aten.select.int(arg1_1, 0, 61)
        select_548 = torch.ops.aten.select.int(select_547, 0, 61);  select_547 = None
        select_549 = torch.ops.aten.select.int(slice_scatter_59, 1, 61)
        unsqueeze_122 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_61 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_550 = torch.ops.aten.select.int(slice_scatter_59, 1, 61)
        view_244 = torch.ops.aten.view.default(select_550, [8, 128, 2]);  select_550 = None
        baddbmm_61 = torch.ops.aten.baddbmm.default(unsqueeze_122, view_244, permute_61, alpha = -2.0);  unsqueeze_122 = view_244 = permute_61 = None
        argmin_61 = torch.ops.aten.argmin.default(baddbmm_61, -1);  baddbmm_61 = None
        unsqueeze_123 = torch.ops.aten.unsqueeze.default(argmin_61, -1)
        expand_122 = torch.ops.aten.expand.default(unsqueeze_123, [8, 128, 2]);  unsqueeze_123 = None
        gather_61 = torch.ops.aten.gather.default(arg3_1, 1, expand_122);  expand_122 = None
        view_245 = torch.ops.aten.view.default(gather_61, [-1]);  gather_61 = None
        select_552 = torch.ops.aten.select.int(select_scatter_60, 1, 61)
        copy_61 = torch.ops.aten.copy.default(select_552, argmin_61);  select_552 = argmin_61 = None
        select_scatter_61 = torch.ops.aten.select_scatter.default(select_scatter_60, copy_61, 1, 61);  select_scatter_60 = copy_61 = None
        sub_61 = torch.ops.aten.sub.Tensor(select_549, view_245);  select_549 = view_245 = None
        div_61 = torch.ops.aten.div.Tensor(sub_61, select_548);  sub_61 = select_548 = None
        select_554 = torch.ops.aten.select.int(arg1_1, 0, 61)
        slice_242 = torch.ops.aten.slice.Tensor(select_554, 0, 61, 9223372036854775807);  select_554 = None
        slice_243 = torch.ops.aten.slice.Tensor(slice_scatter_59, 1, 61, 9223372036854775807)
        expand_123 = torch.ops.aten.expand.default(slice_243, [2048, 67]);  slice_243 = None
        mul_183 = torch.ops.aten.mul.Tensor(expand_123, 1);  expand_123 = None
        view_246 = torch.ops.aten.view.default(div_61, [2048, 1]);  div_61 = None
        mul_184 = torch.ops.aten.mul.Tensor(view_246, slice_242);  view_246 = slice_242 = None
        mul_185 = torch.ops.aten.mul.Tensor(mul_184, -1.0);  mul_184 = None
        add_61 = torch.ops.aten.add.Tensor(mul_183, mul_185);  mul_183 = mul_185 = None
        slice_scatter_60 = torch.ops.aten.slice_scatter.default(slice_scatter_59, add_61, 1, 61, 9223372036854775807);  slice_scatter_59 = add_61 = None
        select_556 = torch.ops.aten.select.int(arg1_1, 0, 62)
        select_557 = torch.ops.aten.select.int(select_556, 0, 62);  select_556 = None
        select_558 = torch.ops.aten.select.int(slice_scatter_60, 1, 62)
        unsqueeze_124 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_62 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_559 = torch.ops.aten.select.int(slice_scatter_60, 1, 62)
        view_248 = torch.ops.aten.view.default(select_559, [8, 128, 2]);  select_559 = None
        baddbmm_62 = torch.ops.aten.baddbmm.default(unsqueeze_124, view_248, permute_62, alpha = -2.0);  unsqueeze_124 = view_248 = permute_62 = None
        argmin_62 = torch.ops.aten.argmin.default(baddbmm_62, -1);  baddbmm_62 = None
        unsqueeze_125 = torch.ops.aten.unsqueeze.default(argmin_62, -1)
        expand_124 = torch.ops.aten.expand.default(unsqueeze_125, [8, 128, 2]);  unsqueeze_125 = None
        gather_62 = torch.ops.aten.gather.default(arg3_1, 1, expand_124);  expand_124 = None
        view_249 = torch.ops.aten.view.default(gather_62, [-1]);  gather_62 = None
        select_561 = torch.ops.aten.select.int(select_scatter_61, 1, 62)
        copy_62 = torch.ops.aten.copy.default(select_561, argmin_62);  select_561 = argmin_62 = None
        select_scatter_62 = torch.ops.aten.select_scatter.default(select_scatter_61, copy_62, 1, 62);  select_scatter_61 = copy_62 = None
        sub_62 = torch.ops.aten.sub.Tensor(select_558, view_249);  select_558 = view_249 = None
        div_62 = torch.ops.aten.div.Tensor(sub_62, select_557);  sub_62 = select_557 = None
        select_563 = torch.ops.aten.select.int(arg1_1, 0, 62)
        slice_246 = torch.ops.aten.slice.Tensor(select_563, 0, 62, 9223372036854775807);  select_563 = None
        slice_247 = torch.ops.aten.slice.Tensor(slice_scatter_60, 1, 62, 9223372036854775807)
        expand_125 = torch.ops.aten.expand.default(slice_247, [2048, 66]);  slice_247 = None
        mul_186 = torch.ops.aten.mul.Tensor(expand_125, 1);  expand_125 = None
        view_250 = torch.ops.aten.view.default(div_62, [2048, 1]);  div_62 = None
        mul_187 = torch.ops.aten.mul.Tensor(view_250, slice_246);  view_250 = slice_246 = None
        mul_188 = torch.ops.aten.mul.Tensor(mul_187, -1.0);  mul_187 = None
        add_62 = torch.ops.aten.add.Tensor(mul_186, mul_188);  mul_186 = mul_188 = None
        slice_scatter_61 = torch.ops.aten.slice_scatter.default(slice_scatter_60, add_62, 1, 62, 9223372036854775807);  slice_scatter_60 = add_62 = None
        select_565 = torch.ops.aten.select.int(arg1_1, 0, 63)
        select_566 = torch.ops.aten.select.int(select_565, 0, 63);  select_565 = None
        select_567 = torch.ops.aten.select.int(slice_scatter_61, 1, 63)
        unsqueeze_126 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_63 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_568 = torch.ops.aten.select.int(slice_scatter_61, 1, 63)
        view_252 = torch.ops.aten.view.default(select_568, [8, 128, 2]);  select_568 = None
        baddbmm_63 = torch.ops.aten.baddbmm.default(unsqueeze_126, view_252, permute_63, alpha = -2.0);  unsqueeze_126 = view_252 = permute_63 = None
        argmin_63 = torch.ops.aten.argmin.default(baddbmm_63, -1);  baddbmm_63 = None
        unsqueeze_127 = torch.ops.aten.unsqueeze.default(argmin_63, -1)
        expand_126 = torch.ops.aten.expand.default(unsqueeze_127, [8, 128, 2]);  unsqueeze_127 = None
        gather_63 = torch.ops.aten.gather.default(arg3_1, 1, expand_126);  expand_126 = None
        view_253 = torch.ops.aten.view.default(gather_63, [-1]);  gather_63 = None
        select_570 = torch.ops.aten.select.int(select_scatter_62, 1, 63)
        copy_63 = torch.ops.aten.copy.default(select_570, argmin_63);  select_570 = argmin_63 = None
        select_scatter_63 = torch.ops.aten.select_scatter.default(select_scatter_62, copy_63, 1, 63);  select_scatter_62 = copy_63 = None
        sub_63 = torch.ops.aten.sub.Tensor(select_567, view_253);  select_567 = view_253 = None
        div_63 = torch.ops.aten.div.Tensor(sub_63, select_566);  sub_63 = select_566 = None
        select_572 = torch.ops.aten.select.int(arg1_1, 0, 63)
        slice_250 = torch.ops.aten.slice.Tensor(select_572, 0, 63, 9223372036854775807);  select_572 = None
        slice_251 = torch.ops.aten.slice.Tensor(slice_scatter_61, 1, 63, 9223372036854775807)
        expand_127 = torch.ops.aten.expand.default(slice_251, [2048, 65]);  slice_251 = None
        mul_189 = torch.ops.aten.mul.Tensor(expand_127, 1);  expand_127 = None
        view_254 = torch.ops.aten.view.default(div_63, [2048, 1]);  div_63 = None
        mul_190 = torch.ops.aten.mul.Tensor(view_254, slice_250);  view_254 = slice_250 = None
        mul_191 = torch.ops.aten.mul.Tensor(mul_190, -1.0);  mul_190 = None
        add_63 = torch.ops.aten.add.Tensor(mul_189, mul_191);  mul_189 = mul_191 = None
        slice_scatter_62 = torch.ops.aten.slice_scatter.default(slice_scatter_61, add_63, 1, 63, 9223372036854775807);  slice_scatter_61 = add_63 = None
        select_574 = torch.ops.aten.select.int(arg1_1, 0, 64)
        select_575 = torch.ops.aten.select.int(select_574, 0, 64);  select_574 = None
        select_576 = torch.ops.aten.select.int(slice_scatter_62, 1, 64)
        unsqueeze_128 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_64 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_577 = torch.ops.aten.select.int(slice_scatter_62, 1, 64)
        view_256 = torch.ops.aten.view.default(select_577, [8, 128, 2]);  select_577 = None
        baddbmm_64 = torch.ops.aten.baddbmm.default(unsqueeze_128, view_256, permute_64, alpha = -2.0);  unsqueeze_128 = view_256 = permute_64 = None
        argmin_64 = torch.ops.aten.argmin.default(baddbmm_64, -1);  baddbmm_64 = None
        unsqueeze_129 = torch.ops.aten.unsqueeze.default(argmin_64, -1)
        expand_128 = torch.ops.aten.expand.default(unsqueeze_129, [8, 128, 2]);  unsqueeze_129 = None
        gather_64 = torch.ops.aten.gather.default(arg3_1, 1, expand_128);  expand_128 = None
        view_257 = torch.ops.aten.view.default(gather_64, [-1]);  gather_64 = None
        select_579 = torch.ops.aten.select.int(select_scatter_63, 1, 64)
        copy_64 = torch.ops.aten.copy.default(select_579, argmin_64);  select_579 = argmin_64 = None
        select_scatter_64 = torch.ops.aten.select_scatter.default(select_scatter_63, copy_64, 1, 64);  select_scatter_63 = copy_64 = None
        sub_64 = torch.ops.aten.sub.Tensor(select_576, view_257);  select_576 = view_257 = None
        div_64 = torch.ops.aten.div.Tensor(sub_64, select_575);  sub_64 = select_575 = None
        select_581 = torch.ops.aten.select.int(arg1_1, 0, 64)
        slice_254 = torch.ops.aten.slice.Tensor(select_581, 0, 64, 9223372036854775807);  select_581 = None
        slice_255 = torch.ops.aten.slice.Tensor(slice_scatter_62, 1, 64, 9223372036854775807)
        expand_129 = torch.ops.aten.expand.default(slice_255, [2048, 64]);  slice_255 = None
        mul_192 = torch.ops.aten.mul.Tensor(expand_129, 1);  expand_129 = None
        view_258 = torch.ops.aten.view.default(div_64, [2048, 1]);  div_64 = None
        mul_193 = torch.ops.aten.mul.Tensor(view_258, slice_254);  view_258 = slice_254 = None
        mul_194 = torch.ops.aten.mul.Tensor(mul_193, -1.0);  mul_193 = None
        add_64 = torch.ops.aten.add.Tensor(mul_192, mul_194);  mul_192 = mul_194 = None
        slice_scatter_63 = torch.ops.aten.slice_scatter.default(slice_scatter_62, add_64, 1, 64, 9223372036854775807);  slice_scatter_62 = add_64 = None
        select_583 = torch.ops.aten.select.int(arg1_1, 0, 65)
        select_584 = torch.ops.aten.select.int(select_583, 0, 65);  select_583 = None
        select_585 = torch.ops.aten.select.int(slice_scatter_63, 1, 65)
        unsqueeze_130 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_65 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_586 = torch.ops.aten.select.int(slice_scatter_63, 1, 65)
        view_260 = torch.ops.aten.view.default(select_586, [8, 128, 2]);  select_586 = None
        baddbmm_65 = torch.ops.aten.baddbmm.default(unsqueeze_130, view_260, permute_65, alpha = -2.0);  unsqueeze_130 = view_260 = permute_65 = None
        argmin_65 = torch.ops.aten.argmin.default(baddbmm_65, -1);  baddbmm_65 = None
        unsqueeze_131 = torch.ops.aten.unsqueeze.default(argmin_65, -1)
        expand_130 = torch.ops.aten.expand.default(unsqueeze_131, [8, 128, 2]);  unsqueeze_131 = None
        gather_65 = torch.ops.aten.gather.default(arg3_1, 1, expand_130);  expand_130 = None
        view_261 = torch.ops.aten.view.default(gather_65, [-1]);  gather_65 = None
        select_588 = torch.ops.aten.select.int(select_scatter_64, 1, 65)
        copy_65 = torch.ops.aten.copy.default(select_588, argmin_65);  select_588 = argmin_65 = None
        select_scatter_65 = torch.ops.aten.select_scatter.default(select_scatter_64, copy_65, 1, 65);  select_scatter_64 = copy_65 = None
        sub_65 = torch.ops.aten.sub.Tensor(select_585, view_261);  select_585 = view_261 = None
        div_65 = torch.ops.aten.div.Tensor(sub_65, select_584);  sub_65 = select_584 = None
        select_590 = torch.ops.aten.select.int(arg1_1, 0, 65)
        slice_258 = torch.ops.aten.slice.Tensor(select_590, 0, 65, 9223372036854775807);  select_590 = None
        slice_259 = torch.ops.aten.slice.Tensor(slice_scatter_63, 1, 65, 9223372036854775807)
        expand_131 = torch.ops.aten.expand.default(slice_259, [2048, 63]);  slice_259 = None
        mul_195 = torch.ops.aten.mul.Tensor(expand_131, 1);  expand_131 = None
        view_262 = torch.ops.aten.view.default(div_65, [2048, 1]);  div_65 = None
        mul_196 = torch.ops.aten.mul.Tensor(view_262, slice_258);  view_262 = slice_258 = None
        mul_197 = torch.ops.aten.mul.Tensor(mul_196, -1.0);  mul_196 = None
        add_65 = torch.ops.aten.add.Tensor(mul_195, mul_197);  mul_195 = mul_197 = None
        slice_scatter_64 = torch.ops.aten.slice_scatter.default(slice_scatter_63, add_65, 1, 65, 9223372036854775807);  slice_scatter_63 = add_65 = None
        select_592 = torch.ops.aten.select.int(arg1_1, 0, 66)
        select_593 = torch.ops.aten.select.int(select_592, 0, 66);  select_592 = None
        select_594 = torch.ops.aten.select.int(slice_scatter_64, 1, 66)
        unsqueeze_132 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_66 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_595 = torch.ops.aten.select.int(slice_scatter_64, 1, 66)
        view_264 = torch.ops.aten.view.default(select_595, [8, 128, 2]);  select_595 = None
        baddbmm_66 = torch.ops.aten.baddbmm.default(unsqueeze_132, view_264, permute_66, alpha = -2.0);  unsqueeze_132 = view_264 = permute_66 = None
        argmin_66 = torch.ops.aten.argmin.default(baddbmm_66, -1);  baddbmm_66 = None
        unsqueeze_133 = torch.ops.aten.unsqueeze.default(argmin_66, -1)
        expand_132 = torch.ops.aten.expand.default(unsqueeze_133, [8, 128, 2]);  unsqueeze_133 = None
        gather_66 = torch.ops.aten.gather.default(arg3_1, 1, expand_132);  expand_132 = None
        view_265 = torch.ops.aten.view.default(gather_66, [-1]);  gather_66 = None
        select_597 = torch.ops.aten.select.int(select_scatter_65, 1, 66)
        copy_66 = torch.ops.aten.copy.default(select_597, argmin_66);  select_597 = argmin_66 = None
        select_scatter_66 = torch.ops.aten.select_scatter.default(select_scatter_65, copy_66, 1, 66);  select_scatter_65 = copy_66 = None
        sub_66 = torch.ops.aten.sub.Tensor(select_594, view_265);  select_594 = view_265 = None
        div_66 = torch.ops.aten.div.Tensor(sub_66, select_593);  sub_66 = select_593 = None
        select_599 = torch.ops.aten.select.int(arg1_1, 0, 66)
        slice_262 = torch.ops.aten.slice.Tensor(select_599, 0, 66, 9223372036854775807);  select_599 = None
        slice_263 = torch.ops.aten.slice.Tensor(slice_scatter_64, 1, 66, 9223372036854775807)
        expand_133 = torch.ops.aten.expand.default(slice_263, [2048, 62]);  slice_263 = None
        mul_198 = torch.ops.aten.mul.Tensor(expand_133, 1);  expand_133 = None
        view_266 = torch.ops.aten.view.default(div_66, [2048, 1]);  div_66 = None
        mul_199 = torch.ops.aten.mul.Tensor(view_266, slice_262);  view_266 = slice_262 = None
        mul_200 = torch.ops.aten.mul.Tensor(mul_199, -1.0);  mul_199 = None
        add_66 = torch.ops.aten.add.Tensor(mul_198, mul_200);  mul_198 = mul_200 = None
        slice_scatter_65 = torch.ops.aten.slice_scatter.default(slice_scatter_64, add_66, 1, 66, 9223372036854775807);  slice_scatter_64 = add_66 = None
        select_601 = torch.ops.aten.select.int(arg1_1, 0, 67)
        select_602 = torch.ops.aten.select.int(select_601, 0, 67);  select_601 = None
        select_603 = torch.ops.aten.select.int(slice_scatter_65, 1, 67)
        unsqueeze_134 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_67 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_604 = torch.ops.aten.select.int(slice_scatter_65, 1, 67)
        view_268 = torch.ops.aten.view.default(select_604, [8, 128, 2]);  select_604 = None
        baddbmm_67 = torch.ops.aten.baddbmm.default(unsqueeze_134, view_268, permute_67, alpha = -2.0);  unsqueeze_134 = view_268 = permute_67 = None
        argmin_67 = torch.ops.aten.argmin.default(baddbmm_67, -1);  baddbmm_67 = None
        unsqueeze_135 = torch.ops.aten.unsqueeze.default(argmin_67, -1)
        expand_134 = torch.ops.aten.expand.default(unsqueeze_135, [8, 128, 2]);  unsqueeze_135 = None
        gather_67 = torch.ops.aten.gather.default(arg3_1, 1, expand_134);  expand_134 = None
        view_269 = torch.ops.aten.view.default(gather_67, [-1]);  gather_67 = None
        select_606 = torch.ops.aten.select.int(select_scatter_66, 1, 67)
        copy_67 = torch.ops.aten.copy.default(select_606, argmin_67);  select_606 = argmin_67 = None
        select_scatter_67 = torch.ops.aten.select_scatter.default(select_scatter_66, copy_67, 1, 67);  select_scatter_66 = copy_67 = None
        sub_67 = torch.ops.aten.sub.Tensor(select_603, view_269);  select_603 = view_269 = None
        div_67 = torch.ops.aten.div.Tensor(sub_67, select_602);  sub_67 = select_602 = None
        select_608 = torch.ops.aten.select.int(arg1_1, 0, 67)
        slice_266 = torch.ops.aten.slice.Tensor(select_608, 0, 67, 9223372036854775807);  select_608 = None
        slice_267 = torch.ops.aten.slice.Tensor(slice_scatter_65, 1, 67, 9223372036854775807)
        expand_135 = torch.ops.aten.expand.default(slice_267, [2048, 61]);  slice_267 = None
        mul_201 = torch.ops.aten.mul.Tensor(expand_135, 1);  expand_135 = None
        view_270 = torch.ops.aten.view.default(div_67, [2048, 1]);  div_67 = None
        mul_202 = torch.ops.aten.mul.Tensor(view_270, slice_266);  view_270 = slice_266 = None
        mul_203 = torch.ops.aten.mul.Tensor(mul_202, -1.0);  mul_202 = None
        add_67 = torch.ops.aten.add.Tensor(mul_201, mul_203);  mul_201 = mul_203 = None
        slice_scatter_66 = torch.ops.aten.slice_scatter.default(slice_scatter_65, add_67, 1, 67, 9223372036854775807);  slice_scatter_65 = add_67 = None
        select_610 = torch.ops.aten.select.int(arg1_1, 0, 68)
        select_611 = torch.ops.aten.select.int(select_610, 0, 68);  select_610 = None
        select_612 = torch.ops.aten.select.int(slice_scatter_66, 1, 68)
        unsqueeze_136 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_68 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_613 = torch.ops.aten.select.int(slice_scatter_66, 1, 68)
        view_272 = torch.ops.aten.view.default(select_613, [8, 128, 2]);  select_613 = None
        baddbmm_68 = torch.ops.aten.baddbmm.default(unsqueeze_136, view_272, permute_68, alpha = -2.0);  unsqueeze_136 = view_272 = permute_68 = None
        argmin_68 = torch.ops.aten.argmin.default(baddbmm_68, -1);  baddbmm_68 = None
        unsqueeze_137 = torch.ops.aten.unsqueeze.default(argmin_68, -1)
        expand_136 = torch.ops.aten.expand.default(unsqueeze_137, [8, 128, 2]);  unsqueeze_137 = None
        gather_68 = torch.ops.aten.gather.default(arg3_1, 1, expand_136);  expand_136 = None
        view_273 = torch.ops.aten.view.default(gather_68, [-1]);  gather_68 = None
        select_615 = torch.ops.aten.select.int(select_scatter_67, 1, 68)
        copy_68 = torch.ops.aten.copy.default(select_615, argmin_68);  select_615 = argmin_68 = None
        select_scatter_68 = torch.ops.aten.select_scatter.default(select_scatter_67, copy_68, 1, 68);  select_scatter_67 = copy_68 = None
        sub_68 = torch.ops.aten.sub.Tensor(select_612, view_273);  select_612 = view_273 = None
        div_68 = torch.ops.aten.div.Tensor(sub_68, select_611);  sub_68 = select_611 = None
        select_617 = torch.ops.aten.select.int(arg1_1, 0, 68)
        slice_270 = torch.ops.aten.slice.Tensor(select_617, 0, 68, 9223372036854775807);  select_617 = None
        slice_271 = torch.ops.aten.slice.Tensor(slice_scatter_66, 1, 68, 9223372036854775807)
        expand_137 = torch.ops.aten.expand.default(slice_271, [2048, 60]);  slice_271 = None
        mul_204 = torch.ops.aten.mul.Tensor(expand_137, 1);  expand_137 = None
        view_274 = torch.ops.aten.view.default(div_68, [2048, 1]);  div_68 = None
        mul_205 = torch.ops.aten.mul.Tensor(view_274, slice_270);  view_274 = slice_270 = None
        mul_206 = torch.ops.aten.mul.Tensor(mul_205, -1.0);  mul_205 = None
        add_68 = torch.ops.aten.add.Tensor(mul_204, mul_206);  mul_204 = mul_206 = None
        slice_scatter_67 = torch.ops.aten.slice_scatter.default(slice_scatter_66, add_68, 1, 68, 9223372036854775807);  slice_scatter_66 = add_68 = None
        select_619 = torch.ops.aten.select.int(arg1_1, 0, 69)
        select_620 = torch.ops.aten.select.int(select_619, 0, 69);  select_619 = None
        select_621 = torch.ops.aten.select.int(slice_scatter_67, 1, 69)
        unsqueeze_138 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_69 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_622 = torch.ops.aten.select.int(slice_scatter_67, 1, 69)
        view_276 = torch.ops.aten.view.default(select_622, [8, 128, 2]);  select_622 = None
        baddbmm_69 = torch.ops.aten.baddbmm.default(unsqueeze_138, view_276, permute_69, alpha = -2.0);  unsqueeze_138 = view_276 = permute_69 = None
        argmin_69 = torch.ops.aten.argmin.default(baddbmm_69, -1);  baddbmm_69 = None
        unsqueeze_139 = torch.ops.aten.unsqueeze.default(argmin_69, -1)
        expand_138 = torch.ops.aten.expand.default(unsqueeze_139, [8, 128, 2]);  unsqueeze_139 = None
        gather_69 = torch.ops.aten.gather.default(arg3_1, 1, expand_138);  expand_138 = None
        view_277 = torch.ops.aten.view.default(gather_69, [-1]);  gather_69 = None
        select_624 = torch.ops.aten.select.int(select_scatter_68, 1, 69)
        copy_69 = torch.ops.aten.copy.default(select_624, argmin_69);  select_624 = argmin_69 = None
        select_scatter_69 = torch.ops.aten.select_scatter.default(select_scatter_68, copy_69, 1, 69);  select_scatter_68 = copy_69 = None
        sub_69 = torch.ops.aten.sub.Tensor(select_621, view_277);  select_621 = view_277 = None
        div_69 = torch.ops.aten.div.Tensor(sub_69, select_620);  sub_69 = select_620 = None
        select_626 = torch.ops.aten.select.int(arg1_1, 0, 69)
        slice_274 = torch.ops.aten.slice.Tensor(select_626, 0, 69, 9223372036854775807);  select_626 = None
        slice_275 = torch.ops.aten.slice.Tensor(slice_scatter_67, 1, 69, 9223372036854775807)
        expand_139 = torch.ops.aten.expand.default(slice_275, [2048, 59]);  slice_275 = None
        mul_207 = torch.ops.aten.mul.Tensor(expand_139, 1);  expand_139 = None
        view_278 = torch.ops.aten.view.default(div_69, [2048, 1]);  div_69 = None
        mul_208 = torch.ops.aten.mul.Tensor(view_278, slice_274);  view_278 = slice_274 = None
        mul_209 = torch.ops.aten.mul.Tensor(mul_208, -1.0);  mul_208 = None
        add_69 = torch.ops.aten.add.Tensor(mul_207, mul_209);  mul_207 = mul_209 = None
        slice_scatter_68 = torch.ops.aten.slice_scatter.default(slice_scatter_67, add_69, 1, 69, 9223372036854775807);  slice_scatter_67 = add_69 = None
        select_628 = torch.ops.aten.select.int(arg1_1, 0, 70)
        select_629 = torch.ops.aten.select.int(select_628, 0, 70);  select_628 = None
        select_630 = torch.ops.aten.select.int(slice_scatter_68, 1, 70)
        unsqueeze_140 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_70 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_631 = torch.ops.aten.select.int(slice_scatter_68, 1, 70)
        view_280 = torch.ops.aten.view.default(select_631, [8, 128, 2]);  select_631 = None
        baddbmm_70 = torch.ops.aten.baddbmm.default(unsqueeze_140, view_280, permute_70, alpha = -2.0);  unsqueeze_140 = view_280 = permute_70 = None
        argmin_70 = torch.ops.aten.argmin.default(baddbmm_70, -1);  baddbmm_70 = None
        unsqueeze_141 = torch.ops.aten.unsqueeze.default(argmin_70, -1)
        expand_140 = torch.ops.aten.expand.default(unsqueeze_141, [8, 128, 2]);  unsqueeze_141 = None
        gather_70 = torch.ops.aten.gather.default(arg3_1, 1, expand_140);  expand_140 = None
        view_281 = torch.ops.aten.view.default(gather_70, [-1]);  gather_70 = None
        select_633 = torch.ops.aten.select.int(select_scatter_69, 1, 70)
        copy_70 = torch.ops.aten.copy.default(select_633, argmin_70);  select_633 = argmin_70 = None
        select_scatter_70 = torch.ops.aten.select_scatter.default(select_scatter_69, copy_70, 1, 70);  select_scatter_69 = copy_70 = None
        sub_70 = torch.ops.aten.sub.Tensor(select_630, view_281);  select_630 = view_281 = None
        div_70 = torch.ops.aten.div.Tensor(sub_70, select_629);  sub_70 = select_629 = None
        select_635 = torch.ops.aten.select.int(arg1_1, 0, 70)
        slice_278 = torch.ops.aten.slice.Tensor(select_635, 0, 70, 9223372036854775807);  select_635 = None
        slice_279 = torch.ops.aten.slice.Tensor(slice_scatter_68, 1, 70, 9223372036854775807)
        expand_141 = torch.ops.aten.expand.default(slice_279, [2048, 58]);  slice_279 = None
        mul_210 = torch.ops.aten.mul.Tensor(expand_141, 1);  expand_141 = None
        view_282 = torch.ops.aten.view.default(div_70, [2048, 1]);  div_70 = None
        mul_211 = torch.ops.aten.mul.Tensor(view_282, slice_278);  view_282 = slice_278 = None
        mul_212 = torch.ops.aten.mul.Tensor(mul_211, -1.0);  mul_211 = None
        add_70 = torch.ops.aten.add.Tensor(mul_210, mul_212);  mul_210 = mul_212 = None
        slice_scatter_69 = torch.ops.aten.slice_scatter.default(slice_scatter_68, add_70, 1, 70, 9223372036854775807);  slice_scatter_68 = add_70 = None
        select_637 = torch.ops.aten.select.int(arg1_1, 0, 71)
        select_638 = torch.ops.aten.select.int(select_637, 0, 71);  select_637 = None
        select_639 = torch.ops.aten.select.int(slice_scatter_69, 1, 71)
        unsqueeze_142 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_71 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_640 = torch.ops.aten.select.int(slice_scatter_69, 1, 71)
        view_284 = torch.ops.aten.view.default(select_640, [8, 128, 2]);  select_640 = None
        baddbmm_71 = torch.ops.aten.baddbmm.default(unsqueeze_142, view_284, permute_71, alpha = -2.0);  unsqueeze_142 = view_284 = permute_71 = None
        argmin_71 = torch.ops.aten.argmin.default(baddbmm_71, -1);  baddbmm_71 = None
        unsqueeze_143 = torch.ops.aten.unsqueeze.default(argmin_71, -1)
        expand_142 = torch.ops.aten.expand.default(unsqueeze_143, [8, 128, 2]);  unsqueeze_143 = None
        gather_71 = torch.ops.aten.gather.default(arg3_1, 1, expand_142);  expand_142 = None
        view_285 = torch.ops.aten.view.default(gather_71, [-1]);  gather_71 = None
        select_642 = torch.ops.aten.select.int(select_scatter_70, 1, 71)
        copy_71 = torch.ops.aten.copy.default(select_642, argmin_71);  select_642 = argmin_71 = None
        select_scatter_71 = torch.ops.aten.select_scatter.default(select_scatter_70, copy_71, 1, 71);  select_scatter_70 = copy_71 = None
        sub_71 = torch.ops.aten.sub.Tensor(select_639, view_285);  select_639 = view_285 = None
        div_71 = torch.ops.aten.div.Tensor(sub_71, select_638);  sub_71 = select_638 = None
        select_644 = torch.ops.aten.select.int(arg1_1, 0, 71)
        slice_282 = torch.ops.aten.slice.Tensor(select_644, 0, 71, 9223372036854775807);  select_644 = None
        slice_283 = torch.ops.aten.slice.Tensor(slice_scatter_69, 1, 71, 9223372036854775807)
        expand_143 = torch.ops.aten.expand.default(slice_283, [2048, 57]);  slice_283 = None
        mul_213 = torch.ops.aten.mul.Tensor(expand_143, 1);  expand_143 = None
        view_286 = torch.ops.aten.view.default(div_71, [2048, 1]);  div_71 = None
        mul_214 = torch.ops.aten.mul.Tensor(view_286, slice_282);  view_286 = slice_282 = None
        mul_215 = torch.ops.aten.mul.Tensor(mul_214, -1.0);  mul_214 = None
        add_71 = torch.ops.aten.add.Tensor(mul_213, mul_215);  mul_213 = mul_215 = None
        slice_scatter_70 = torch.ops.aten.slice_scatter.default(slice_scatter_69, add_71, 1, 71, 9223372036854775807);  slice_scatter_69 = add_71 = None
        select_646 = torch.ops.aten.select.int(arg1_1, 0, 72)
        select_647 = torch.ops.aten.select.int(select_646, 0, 72);  select_646 = None
        select_648 = torch.ops.aten.select.int(slice_scatter_70, 1, 72)
        unsqueeze_144 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_72 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_649 = torch.ops.aten.select.int(slice_scatter_70, 1, 72)
        view_288 = torch.ops.aten.view.default(select_649, [8, 128, 2]);  select_649 = None
        baddbmm_72 = torch.ops.aten.baddbmm.default(unsqueeze_144, view_288, permute_72, alpha = -2.0);  unsqueeze_144 = view_288 = permute_72 = None
        argmin_72 = torch.ops.aten.argmin.default(baddbmm_72, -1);  baddbmm_72 = None
        unsqueeze_145 = torch.ops.aten.unsqueeze.default(argmin_72, -1)
        expand_144 = torch.ops.aten.expand.default(unsqueeze_145, [8, 128, 2]);  unsqueeze_145 = None
        gather_72 = torch.ops.aten.gather.default(arg3_1, 1, expand_144);  expand_144 = None
        view_289 = torch.ops.aten.view.default(gather_72, [-1]);  gather_72 = None
        select_651 = torch.ops.aten.select.int(select_scatter_71, 1, 72)
        copy_72 = torch.ops.aten.copy.default(select_651, argmin_72);  select_651 = argmin_72 = None
        select_scatter_72 = torch.ops.aten.select_scatter.default(select_scatter_71, copy_72, 1, 72);  select_scatter_71 = copy_72 = None
        sub_72 = torch.ops.aten.sub.Tensor(select_648, view_289);  select_648 = view_289 = None
        div_72 = torch.ops.aten.div.Tensor(sub_72, select_647);  sub_72 = select_647 = None
        select_653 = torch.ops.aten.select.int(arg1_1, 0, 72)
        slice_286 = torch.ops.aten.slice.Tensor(select_653, 0, 72, 9223372036854775807);  select_653 = None
        slice_287 = torch.ops.aten.slice.Tensor(slice_scatter_70, 1, 72, 9223372036854775807)
        expand_145 = torch.ops.aten.expand.default(slice_287, [2048, 56]);  slice_287 = None
        mul_216 = torch.ops.aten.mul.Tensor(expand_145, 1);  expand_145 = None
        view_290 = torch.ops.aten.view.default(div_72, [2048, 1]);  div_72 = None
        mul_217 = torch.ops.aten.mul.Tensor(view_290, slice_286);  view_290 = slice_286 = None
        mul_218 = torch.ops.aten.mul.Tensor(mul_217, -1.0);  mul_217 = None
        add_72 = torch.ops.aten.add.Tensor(mul_216, mul_218);  mul_216 = mul_218 = None
        slice_scatter_71 = torch.ops.aten.slice_scatter.default(slice_scatter_70, add_72, 1, 72, 9223372036854775807);  slice_scatter_70 = add_72 = None
        select_655 = torch.ops.aten.select.int(arg1_1, 0, 73)
        select_656 = torch.ops.aten.select.int(select_655, 0, 73);  select_655 = None
        select_657 = torch.ops.aten.select.int(slice_scatter_71, 1, 73)
        unsqueeze_146 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_73 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_658 = torch.ops.aten.select.int(slice_scatter_71, 1, 73)
        view_292 = torch.ops.aten.view.default(select_658, [8, 128, 2]);  select_658 = None
        baddbmm_73 = torch.ops.aten.baddbmm.default(unsqueeze_146, view_292, permute_73, alpha = -2.0);  unsqueeze_146 = view_292 = permute_73 = None
        argmin_73 = torch.ops.aten.argmin.default(baddbmm_73, -1);  baddbmm_73 = None
        unsqueeze_147 = torch.ops.aten.unsqueeze.default(argmin_73, -1)
        expand_146 = torch.ops.aten.expand.default(unsqueeze_147, [8, 128, 2]);  unsqueeze_147 = None
        gather_73 = torch.ops.aten.gather.default(arg3_1, 1, expand_146);  expand_146 = None
        view_293 = torch.ops.aten.view.default(gather_73, [-1]);  gather_73 = None
        select_660 = torch.ops.aten.select.int(select_scatter_72, 1, 73)
        copy_73 = torch.ops.aten.copy.default(select_660, argmin_73);  select_660 = argmin_73 = None
        select_scatter_73 = torch.ops.aten.select_scatter.default(select_scatter_72, copy_73, 1, 73);  select_scatter_72 = copy_73 = None
        sub_73 = torch.ops.aten.sub.Tensor(select_657, view_293);  select_657 = view_293 = None
        div_73 = torch.ops.aten.div.Tensor(sub_73, select_656);  sub_73 = select_656 = None
        select_662 = torch.ops.aten.select.int(arg1_1, 0, 73)
        slice_290 = torch.ops.aten.slice.Tensor(select_662, 0, 73, 9223372036854775807);  select_662 = None
        slice_291 = torch.ops.aten.slice.Tensor(slice_scatter_71, 1, 73, 9223372036854775807)
        expand_147 = torch.ops.aten.expand.default(slice_291, [2048, 55]);  slice_291 = None
        mul_219 = torch.ops.aten.mul.Tensor(expand_147, 1);  expand_147 = None
        view_294 = torch.ops.aten.view.default(div_73, [2048, 1]);  div_73 = None
        mul_220 = torch.ops.aten.mul.Tensor(view_294, slice_290);  view_294 = slice_290 = None
        mul_221 = torch.ops.aten.mul.Tensor(mul_220, -1.0);  mul_220 = None
        add_73 = torch.ops.aten.add.Tensor(mul_219, mul_221);  mul_219 = mul_221 = None
        slice_scatter_72 = torch.ops.aten.slice_scatter.default(slice_scatter_71, add_73, 1, 73, 9223372036854775807);  slice_scatter_71 = add_73 = None
        select_664 = torch.ops.aten.select.int(arg1_1, 0, 74)
        select_665 = torch.ops.aten.select.int(select_664, 0, 74);  select_664 = None
        select_666 = torch.ops.aten.select.int(slice_scatter_72, 1, 74)
        unsqueeze_148 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_74 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_667 = torch.ops.aten.select.int(slice_scatter_72, 1, 74)
        view_296 = torch.ops.aten.view.default(select_667, [8, 128, 2]);  select_667 = None
        baddbmm_74 = torch.ops.aten.baddbmm.default(unsqueeze_148, view_296, permute_74, alpha = -2.0);  unsqueeze_148 = view_296 = permute_74 = None
        argmin_74 = torch.ops.aten.argmin.default(baddbmm_74, -1);  baddbmm_74 = None
        unsqueeze_149 = torch.ops.aten.unsqueeze.default(argmin_74, -1)
        expand_148 = torch.ops.aten.expand.default(unsqueeze_149, [8, 128, 2]);  unsqueeze_149 = None
        gather_74 = torch.ops.aten.gather.default(arg3_1, 1, expand_148);  expand_148 = None
        view_297 = torch.ops.aten.view.default(gather_74, [-1]);  gather_74 = None
        select_669 = torch.ops.aten.select.int(select_scatter_73, 1, 74)
        copy_74 = torch.ops.aten.copy.default(select_669, argmin_74);  select_669 = argmin_74 = None
        select_scatter_74 = torch.ops.aten.select_scatter.default(select_scatter_73, copy_74, 1, 74);  select_scatter_73 = copy_74 = None
        sub_74 = torch.ops.aten.sub.Tensor(select_666, view_297);  select_666 = view_297 = None
        div_74 = torch.ops.aten.div.Tensor(sub_74, select_665);  sub_74 = select_665 = None
        select_671 = torch.ops.aten.select.int(arg1_1, 0, 74)
        slice_294 = torch.ops.aten.slice.Tensor(select_671, 0, 74, 9223372036854775807);  select_671 = None
        slice_295 = torch.ops.aten.slice.Tensor(slice_scatter_72, 1, 74, 9223372036854775807)
        expand_149 = torch.ops.aten.expand.default(slice_295, [2048, 54]);  slice_295 = None
        mul_222 = torch.ops.aten.mul.Tensor(expand_149, 1);  expand_149 = None
        view_298 = torch.ops.aten.view.default(div_74, [2048, 1]);  div_74 = None
        mul_223 = torch.ops.aten.mul.Tensor(view_298, slice_294);  view_298 = slice_294 = None
        mul_224 = torch.ops.aten.mul.Tensor(mul_223, -1.0);  mul_223 = None
        add_74 = torch.ops.aten.add.Tensor(mul_222, mul_224);  mul_222 = mul_224 = None
        slice_scatter_73 = torch.ops.aten.slice_scatter.default(slice_scatter_72, add_74, 1, 74, 9223372036854775807);  slice_scatter_72 = add_74 = None
        select_673 = torch.ops.aten.select.int(arg1_1, 0, 75)
        select_674 = torch.ops.aten.select.int(select_673, 0, 75);  select_673 = None
        select_675 = torch.ops.aten.select.int(slice_scatter_73, 1, 75)
        unsqueeze_150 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_75 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_676 = torch.ops.aten.select.int(slice_scatter_73, 1, 75)
        view_300 = torch.ops.aten.view.default(select_676, [8, 128, 2]);  select_676 = None
        baddbmm_75 = torch.ops.aten.baddbmm.default(unsqueeze_150, view_300, permute_75, alpha = -2.0);  unsqueeze_150 = view_300 = permute_75 = None
        argmin_75 = torch.ops.aten.argmin.default(baddbmm_75, -1);  baddbmm_75 = None
        unsqueeze_151 = torch.ops.aten.unsqueeze.default(argmin_75, -1)
        expand_150 = torch.ops.aten.expand.default(unsqueeze_151, [8, 128, 2]);  unsqueeze_151 = None
        gather_75 = torch.ops.aten.gather.default(arg3_1, 1, expand_150);  expand_150 = None
        view_301 = torch.ops.aten.view.default(gather_75, [-1]);  gather_75 = None
        select_678 = torch.ops.aten.select.int(select_scatter_74, 1, 75)
        copy_75 = torch.ops.aten.copy.default(select_678, argmin_75);  select_678 = argmin_75 = None
        select_scatter_75 = torch.ops.aten.select_scatter.default(select_scatter_74, copy_75, 1, 75);  select_scatter_74 = copy_75 = None
        sub_75 = torch.ops.aten.sub.Tensor(select_675, view_301);  select_675 = view_301 = None
        div_75 = torch.ops.aten.div.Tensor(sub_75, select_674);  sub_75 = select_674 = None
        select_680 = torch.ops.aten.select.int(arg1_1, 0, 75)
        slice_298 = torch.ops.aten.slice.Tensor(select_680, 0, 75, 9223372036854775807);  select_680 = None
        slice_299 = torch.ops.aten.slice.Tensor(slice_scatter_73, 1, 75, 9223372036854775807)
        expand_151 = torch.ops.aten.expand.default(slice_299, [2048, 53]);  slice_299 = None
        mul_225 = torch.ops.aten.mul.Tensor(expand_151, 1);  expand_151 = None
        view_302 = torch.ops.aten.view.default(div_75, [2048, 1]);  div_75 = None
        mul_226 = torch.ops.aten.mul.Tensor(view_302, slice_298);  view_302 = slice_298 = None
        mul_227 = torch.ops.aten.mul.Tensor(mul_226, -1.0);  mul_226 = None
        add_75 = torch.ops.aten.add.Tensor(mul_225, mul_227);  mul_225 = mul_227 = None
        slice_scatter_74 = torch.ops.aten.slice_scatter.default(slice_scatter_73, add_75, 1, 75, 9223372036854775807);  slice_scatter_73 = add_75 = None
        select_682 = torch.ops.aten.select.int(arg1_1, 0, 76)
        select_683 = torch.ops.aten.select.int(select_682, 0, 76);  select_682 = None
        select_684 = torch.ops.aten.select.int(slice_scatter_74, 1, 76)
        unsqueeze_152 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_76 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_685 = torch.ops.aten.select.int(slice_scatter_74, 1, 76)
        view_304 = torch.ops.aten.view.default(select_685, [8, 128, 2]);  select_685 = None
        baddbmm_76 = torch.ops.aten.baddbmm.default(unsqueeze_152, view_304, permute_76, alpha = -2.0);  unsqueeze_152 = view_304 = permute_76 = None
        argmin_76 = torch.ops.aten.argmin.default(baddbmm_76, -1);  baddbmm_76 = None
        unsqueeze_153 = torch.ops.aten.unsqueeze.default(argmin_76, -1)
        expand_152 = torch.ops.aten.expand.default(unsqueeze_153, [8, 128, 2]);  unsqueeze_153 = None
        gather_76 = torch.ops.aten.gather.default(arg3_1, 1, expand_152);  expand_152 = None
        view_305 = torch.ops.aten.view.default(gather_76, [-1]);  gather_76 = None
        select_687 = torch.ops.aten.select.int(select_scatter_75, 1, 76)
        copy_76 = torch.ops.aten.copy.default(select_687, argmin_76);  select_687 = argmin_76 = None
        select_scatter_76 = torch.ops.aten.select_scatter.default(select_scatter_75, copy_76, 1, 76);  select_scatter_75 = copy_76 = None
        sub_76 = torch.ops.aten.sub.Tensor(select_684, view_305);  select_684 = view_305 = None
        div_76 = torch.ops.aten.div.Tensor(sub_76, select_683);  sub_76 = select_683 = None
        select_689 = torch.ops.aten.select.int(arg1_1, 0, 76)
        slice_302 = torch.ops.aten.slice.Tensor(select_689, 0, 76, 9223372036854775807);  select_689 = None
        slice_303 = torch.ops.aten.slice.Tensor(slice_scatter_74, 1, 76, 9223372036854775807)
        expand_153 = torch.ops.aten.expand.default(slice_303, [2048, 52]);  slice_303 = None
        mul_228 = torch.ops.aten.mul.Tensor(expand_153, 1);  expand_153 = None
        view_306 = torch.ops.aten.view.default(div_76, [2048, 1]);  div_76 = None
        mul_229 = torch.ops.aten.mul.Tensor(view_306, slice_302);  view_306 = slice_302 = None
        mul_230 = torch.ops.aten.mul.Tensor(mul_229, -1.0);  mul_229 = None
        add_76 = torch.ops.aten.add.Tensor(mul_228, mul_230);  mul_228 = mul_230 = None
        slice_scatter_75 = torch.ops.aten.slice_scatter.default(slice_scatter_74, add_76, 1, 76, 9223372036854775807);  slice_scatter_74 = add_76 = None
        select_691 = torch.ops.aten.select.int(arg1_1, 0, 77)
        select_692 = torch.ops.aten.select.int(select_691, 0, 77);  select_691 = None
        select_693 = torch.ops.aten.select.int(slice_scatter_75, 1, 77)
        unsqueeze_154 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_77 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_694 = torch.ops.aten.select.int(slice_scatter_75, 1, 77)
        view_308 = torch.ops.aten.view.default(select_694, [8, 128, 2]);  select_694 = None
        baddbmm_77 = torch.ops.aten.baddbmm.default(unsqueeze_154, view_308, permute_77, alpha = -2.0);  unsqueeze_154 = view_308 = permute_77 = None
        argmin_77 = torch.ops.aten.argmin.default(baddbmm_77, -1);  baddbmm_77 = None
        unsqueeze_155 = torch.ops.aten.unsqueeze.default(argmin_77, -1)
        expand_154 = torch.ops.aten.expand.default(unsqueeze_155, [8, 128, 2]);  unsqueeze_155 = None
        gather_77 = torch.ops.aten.gather.default(arg3_1, 1, expand_154);  expand_154 = None
        view_309 = torch.ops.aten.view.default(gather_77, [-1]);  gather_77 = None
        select_696 = torch.ops.aten.select.int(select_scatter_76, 1, 77)
        copy_77 = torch.ops.aten.copy.default(select_696, argmin_77);  select_696 = argmin_77 = None
        select_scatter_77 = torch.ops.aten.select_scatter.default(select_scatter_76, copy_77, 1, 77);  select_scatter_76 = copy_77 = None
        sub_77 = torch.ops.aten.sub.Tensor(select_693, view_309);  select_693 = view_309 = None
        div_77 = torch.ops.aten.div.Tensor(sub_77, select_692);  sub_77 = select_692 = None
        select_698 = torch.ops.aten.select.int(arg1_1, 0, 77)
        slice_306 = torch.ops.aten.slice.Tensor(select_698, 0, 77, 9223372036854775807);  select_698 = None
        slice_307 = torch.ops.aten.slice.Tensor(slice_scatter_75, 1, 77, 9223372036854775807)
        expand_155 = torch.ops.aten.expand.default(slice_307, [2048, 51]);  slice_307 = None
        mul_231 = torch.ops.aten.mul.Tensor(expand_155, 1);  expand_155 = None
        view_310 = torch.ops.aten.view.default(div_77, [2048, 1]);  div_77 = None
        mul_232 = torch.ops.aten.mul.Tensor(view_310, slice_306);  view_310 = slice_306 = None
        mul_233 = torch.ops.aten.mul.Tensor(mul_232, -1.0);  mul_232 = None
        add_77 = torch.ops.aten.add.Tensor(mul_231, mul_233);  mul_231 = mul_233 = None
        slice_scatter_76 = torch.ops.aten.slice_scatter.default(slice_scatter_75, add_77, 1, 77, 9223372036854775807);  slice_scatter_75 = add_77 = None
        select_700 = torch.ops.aten.select.int(arg1_1, 0, 78)
        select_701 = torch.ops.aten.select.int(select_700, 0, 78);  select_700 = None
        select_702 = torch.ops.aten.select.int(slice_scatter_76, 1, 78)
        unsqueeze_156 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_78 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_703 = torch.ops.aten.select.int(slice_scatter_76, 1, 78)
        view_312 = torch.ops.aten.view.default(select_703, [8, 128, 2]);  select_703 = None
        baddbmm_78 = torch.ops.aten.baddbmm.default(unsqueeze_156, view_312, permute_78, alpha = -2.0);  unsqueeze_156 = view_312 = permute_78 = None
        argmin_78 = torch.ops.aten.argmin.default(baddbmm_78, -1);  baddbmm_78 = None
        unsqueeze_157 = torch.ops.aten.unsqueeze.default(argmin_78, -1)
        expand_156 = torch.ops.aten.expand.default(unsqueeze_157, [8, 128, 2]);  unsqueeze_157 = None
        gather_78 = torch.ops.aten.gather.default(arg3_1, 1, expand_156);  expand_156 = None
        view_313 = torch.ops.aten.view.default(gather_78, [-1]);  gather_78 = None
        select_705 = torch.ops.aten.select.int(select_scatter_77, 1, 78)
        copy_78 = torch.ops.aten.copy.default(select_705, argmin_78);  select_705 = argmin_78 = None
        select_scatter_78 = torch.ops.aten.select_scatter.default(select_scatter_77, copy_78, 1, 78);  select_scatter_77 = copy_78 = None
        sub_78 = torch.ops.aten.sub.Tensor(select_702, view_313);  select_702 = view_313 = None
        div_78 = torch.ops.aten.div.Tensor(sub_78, select_701);  sub_78 = select_701 = None
        select_707 = torch.ops.aten.select.int(arg1_1, 0, 78)
        slice_310 = torch.ops.aten.slice.Tensor(select_707, 0, 78, 9223372036854775807);  select_707 = None
        slice_311 = torch.ops.aten.slice.Tensor(slice_scatter_76, 1, 78, 9223372036854775807)
        expand_157 = torch.ops.aten.expand.default(slice_311, [2048, 50]);  slice_311 = None
        mul_234 = torch.ops.aten.mul.Tensor(expand_157, 1);  expand_157 = None
        view_314 = torch.ops.aten.view.default(div_78, [2048, 1]);  div_78 = None
        mul_235 = torch.ops.aten.mul.Tensor(view_314, slice_310);  view_314 = slice_310 = None
        mul_236 = torch.ops.aten.mul.Tensor(mul_235, -1.0);  mul_235 = None
        add_78 = torch.ops.aten.add.Tensor(mul_234, mul_236);  mul_234 = mul_236 = None
        slice_scatter_77 = torch.ops.aten.slice_scatter.default(slice_scatter_76, add_78, 1, 78, 9223372036854775807);  slice_scatter_76 = add_78 = None
        select_709 = torch.ops.aten.select.int(arg1_1, 0, 79)
        select_710 = torch.ops.aten.select.int(select_709, 0, 79);  select_709 = None
        select_711 = torch.ops.aten.select.int(slice_scatter_77, 1, 79)
        unsqueeze_158 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_79 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_712 = torch.ops.aten.select.int(slice_scatter_77, 1, 79)
        view_316 = torch.ops.aten.view.default(select_712, [8, 128, 2]);  select_712 = None
        baddbmm_79 = torch.ops.aten.baddbmm.default(unsqueeze_158, view_316, permute_79, alpha = -2.0);  unsqueeze_158 = view_316 = permute_79 = None
        argmin_79 = torch.ops.aten.argmin.default(baddbmm_79, -1);  baddbmm_79 = None
        unsqueeze_159 = torch.ops.aten.unsqueeze.default(argmin_79, -1)
        expand_158 = torch.ops.aten.expand.default(unsqueeze_159, [8, 128, 2]);  unsqueeze_159 = None
        gather_79 = torch.ops.aten.gather.default(arg3_1, 1, expand_158);  expand_158 = None
        view_317 = torch.ops.aten.view.default(gather_79, [-1]);  gather_79 = None
        select_714 = torch.ops.aten.select.int(select_scatter_78, 1, 79)
        copy_79 = torch.ops.aten.copy.default(select_714, argmin_79);  select_714 = argmin_79 = None
        select_scatter_79 = torch.ops.aten.select_scatter.default(select_scatter_78, copy_79, 1, 79);  select_scatter_78 = copy_79 = None
        sub_79 = torch.ops.aten.sub.Tensor(select_711, view_317);  select_711 = view_317 = None
        div_79 = torch.ops.aten.div.Tensor(sub_79, select_710);  sub_79 = select_710 = None
        select_716 = torch.ops.aten.select.int(arg1_1, 0, 79)
        slice_314 = torch.ops.aten.slice.Tensor(select_716, 0, 79, 9223372036854775807);  select_716 = None
        slice_315 = torch.ops.aten.slice.Tensor(slice_scatter_77, 1, 79, 9223372036854775807)
        expand_159 = torch.ops.aten.expand.default(slice_315, [2048, 49]);  slice_315 = None
        mul_237 = torch.ops.aten.mul.Tensor(expand_159, 1);  expand_159 = None
        view_318 = torch.ops.aten.view.default(div_79, [2048, 1]);  div_79 = None
        mul_238 = torch.ops.aten.mul.Tensor(view_318, slice_314);  view_318 = slice_314 = None
        mul_239 = torch.ops.aten.mul.Tensor(mul_238, -1.0);  mul_238 = None
        add_79 = torch.ops.aten.add.Tensor(mul_237, mul_239);  mul_237 = mul_239 = None
        slice_scatter_78 = torch.ops.aten.slice_scatter.default(slice_scatter_77, add_79, 1, 79, 9223372036854775807);  slice_scatter_77 = add_79 = None
        select_718 = torch.ops.aten.select.int(arg1_1, 0, 80)
        select_719 = torch.ops.aten.select.int(select_718, 0, 80);  select_718 = None
        select_720 = torch.ops.aten.select.int(slice_scatter_78, 1, 80)
        unsqueeze_160 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_80 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_721 = torch.ops.aten.select.int(slice_scatter_78, 1, 80)
        view_320 = torch.ops.aten.view.default(select_721, [8, 128, 2]);  select_721 = None
        baddbmm_80 = torch.ops.aten.baddbmm.default(unsqueeze_160, view_320, permute_80, alpha = -2.0);  unsqueeze_160 = view_320 = permute_80 = None
        argmin_80 = torch.ops.aten.argmin.default(baddbmm_80, -1);  baddbmm_80 = None
        unsqueeze_161 = torch.ops.aten.unsqueeze.default(argmin_80, -1)
        expand_160 = torch.ops.aten.expand.default(unsqueeze_161, [8, 128, 2]);  unsqueeze_161 = None
        gather_80 = torch.ops.aten.gather.default(arg3_1, 1, expand_160);  expand_160 = None
        view_321 = torch.ops.aten.view.default(gather_80, [-1]);  gather_80 = None
        select_723 = torch.ops.aten.select.int(select_scatter_79, 1, 80)
        copy_80 = torch.ops.aten.copy.default(select_723, argmin_80);  select_723 = argmin_80 = None
        select_scatter_80 = torch.ops.aten.select_scatter.default(select_scatter_79, copy_80, 1, 80);  select_scatter_79 = copy_80 = None
        sub_80 = torch.ops.aten.sub.Tensor(select_720, view_321);  select_720 = view_321 = None
        div_80 = torch.ops.aten.div.Tensor(sub_80, select_719);  sub_80 = select_719 = None
        select_725 = torch.ops.aten.select.int(arg1_1, 0, 80)
        slice_318 = torch.ops.aten.slice.Tensor(select_725, 0, 80, 9223372036854775807);  select_725 = None
        slice_319 = torch.ops.aten.slice.Tensor(slice_scatter_78, 1, 80, 9223372036854775807)
        expand_161 = torch.ops.aten.expand.default(slice_319, [2048, 48]);  slice_319 = None
        mul_240 = torch.ops.aten.mul.Tensor(expand_161, 1);  expand_161 = None
        view_322 = torch.ops.aten.view.default(div_80, [2048, 1]);  div_80 = None
        mul_241 = torch.ops.aten.mul.Tensor(view_322, slice_318);  view_322 = slice_318 = None
        mul_242 = torch.ops.aten.mul.Tensor(mul_241, -1.0);  mul_241 = None
        add_80 = torch.ops.aten.add.Tensor(mul_240, mul_242);  mul_240 = mul_242 = None
        slice_scatter_79 = torch.ops.aten.slice_scatter.default(slice_scatter_78, add_80, 1, 80, 9223372036854775807);  slice_scatter_78 = add_80 = None
        select_727 = torch.ops.aten.select.int(arg1_1, 0, 81)
        select_728 = torch.ops.aten.select.int(select_727, 0, 81);  select_727 = None
        select_729 = torch.ops.aten.select.int(slice_scatter_79, 1, 81)
        unsqueeze_162 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_81 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_730 = torch.ops.aten.select.int(slice_scatter_79, 1, 81)
        view_324 = torch.ops.aten.view.default(select_730, [8, 128, 2]);  select_730 = None
        baddbmm_81 = torch.ops.aten.baddbmm.default(unsqueeze_162, view_324, permute_81, alpha = -2.0);  unsqueeze_162 = view_324 = permute_81 = None
        argmin_81 = torch.ops.aten.argmin.default(baddbmm_81, -1);  baddbmm_81 = None
        unsqueeze_163 = torch.ops.aten.unsqueeze.default(argmin_81, -1)
        expand_162 = torch.ops.aten.expand.default(unsqueeze_163, [8, 128, 2]);  unsqueeze_163 = None
        gather_81 = torch.ops.aten.gather.default(arg3_1, 1, expand_162);  expand_162 = None
        view_325 = torch.ops.aten.view.default(gather_81, [-1]);  gather_81 = None
        select_732 = torch.ops.aten.select.int(select_scatter_80, 1, 81)
        copy_81 = torch.ops.aten.copy.default(select_732, argmin_81);  select_732 = argmin_81 = None
        select_scatter_81 = torch.ops.aten.select_scatter.default(select_scatter_80, copy_81, 1, 81);  select_scatter_80 = copy_81 = None
        sub_81 = torch.ops.aten.sub.Tensor(select_729, view_325);  select_729 = view_325 = None
        div_81 = torch.ops.aten.div.Tensor(sub_81, select_728);  sub_81 = select_728 = None
        select_734 = torch.ops.aten.select.int(arg1_1, 0, 81)
        slice_322 = torch.ops.aten.slice.Tensor(select_734, 0, 81, 9223372036854775807);  select_734 = None
        slice_323 = torch.ops.aten.slice.Tensor(slice_scatter_79, 1, 81, 9223372036854775807)
        expand_163 = torch.ops.aten.expand.default(slice_323, [2048, 47]);  slice_323 = None
        mul_243 = torch.ops.aten.mul.Tensor(expand_163, 1);  expand_163 = None
        view_326 = torch.ops.aten.view.default(div_81, [2048, 1]);  div_81 = None
        mul_244 = torch.ops.aten.mul.Tensor(view_326, slice_322);  view_326 = slice_322 = None
        mul_245 = torch.ops.aten.mul.Tensor(mul_244, -1.0);  mul_244 = None
        add_81 = torch.ops.aten.add.Tensor(mul_243, mul_245);  mul_243 = mul_245 = None
        slice_scatter_80 = torch.ops.aten.slice_scatter.default(slice_scatter_79, add_81, 1, 81, 9223372036854775807);  slice_scatter_79 = add_81 = None
        select_736 = torch.ops.aten.select.int(arg1_1, 0, 82)
        select_737 = torch.ops.aten.select.int(select_736, 0, 82);  select_736 = None
        select_738 = torch.ops.aten.select.int(slice_scatter_80, 1, 82)
        unsqueeze_164 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_82 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_739 = torch.ops.aten.select.int(slice_scatter_80, 1, 82)
        view_328 = torch.ops.aten.view.default(select_739, [8, 128, 2]);  select_739 = None
        baddbmm_82 = torch.ops.aten.baddbmm.default(unsqueeze_164, view_328, permute_82, alpha = -2.0);  unsqueeze_164 = view_328 = permute_82 = None
        argmin_82 = torch.ops.aten.argmin.default(baddbmm_82, -1);  baddbmm_82 = None
        unsqueeze_165 = torch.ops.aten.unsqueeze.default(argmin_82, -1)
        expand_164 = torch.ops.aten.expand.default(unsqueeze_165, [8, 128, 2]);  unsqueeze_165 = None
        gather_82 = torch.ops.aten.gather.default(arg3_1, 1, expand_164);  expand_164 = None
        view_329 = torch.ops.aten.view.default(gather_82, [-1]);  gather_82 = None
        select_741 = torch.ops.aten.select.int(select_scatter_81, 1, 82)
        copy_82 = torch.ops.aten.copy.default(select_741, argmin_82);  select_741 = argmin_82 = None
        select_scatter_82 = torch.ops.aten.select_scatter.default(select_scatter_81, copy_82, 1, 82);  select_scatter_81 = copy_82 = None
        sub_82 = torch.ops.aten.sub.Tensor(select_738, view_329);  select_738 = view_329 = None
        div_82 = torch.ops.aten.div.Tensor(sub_82, select_737);  sub_82 = select_737 = None
        select_743 = torch.ops.aten.select.int(arg1_1, 0, 82)
        slice_326 = torch.ops.aten.slice.Tensor(select_743, 0, 82, 9223372036854775807);  select_743 = None
        slice_327 = torch.ops.aten.slice.Tensor(slice_scatter_80, 1, 82, 9223372036854775807)
        expand_165 = torch.ops.aten.expand.default(slice_327, [2048, 46]);  slice_327 = None
        mul_246 = torch.ops.aten.mul.Tensor(expand_165, 1);  expand_165 = None
        view_330 = torch.ops.aten.view.default(div_82, [2048, 1]);  div_82 = None
        mul_247 = torch.ops.aten.mul.Tensor(view_330, slice_326);  view_330 = slice_326 = None
        mul_248 = torch.ops.aten.mul.Tensor(mul_247, -1.0);  mul_247 = None
        add_82 = torch.ops.aten.add.Tensor(mul_246, mul_248);  mul_246 = mul_248 = None
        slice_scatter_81 = torch.ops.aten.slice_scatter.default(slice_scatter_80, add_82, 1, 82, 9223372036854775807);  slice_scatter_80 = add_82 = None
        select_745 = torch.ops.aten.select.int(arg1_1, 0, 83)
        select_746 = torch.ops.aten.select.int(select_745, 0, 83);  select_745 = None
        select_747 = torch.ops.aten.select.int(slice_scatter_81, 1, 83)
        unsqueeze_166 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_83 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_748 = torch.ops.aten.select.int(slice_scatter_81, 1, 83)
        view_332 = torch.ops.aten.view.default(select_748, [8, 128, 2]);  select_748 = None
        baddbmm_83 = torch.ops.aten.baddbmm.default(unsqueeze_166, view_332, permute_83, alpha = -2.0);  unsqueeze_166 = view_332 = permute_83 = None
        argmin_83 = torch.ops.aten.argmin.default(baddbmm_83, -1);  baddbmm_83 = None
        unsqueeze_167 = torch.ops.aten.unsqueeze.default(argmin_83, -1)
        expand_166 = torch.ops.aten.expand.default(unsqueeze_167, [8, 128, 2]);  unsqueeze_167 = None
        gather_83 = torch.ops.aten.gather.default(arg3_1, 1, expand_166);  expand_166 = None
        view_333 = torch.ops.aten.view.default(gather_83, [-1]);  gather_83 = None
        select_750 = torch.ops.aten.select.int(select_scatter_82, 1, 83)
        copy_83 = torch.ops.aten.copy.default(select_750, argmin_83);  select_750 = argmin_83 = None
        select_scatter_83 = torch.ops.aten.select_scatter.default(select_scatter_82, copy_83, 1, 83);  select_scatter_82 = copy_83 = None
        sub_83 = torch.ops.aten.sub.Tensor(select_747, view_333);  select_747 = view_333 = None
        div_83 = torch.ops.aten.div.Tensor(sub_83, select_746);  sub_83 = select_746 = None
        select_752 = torch.ops.aten.select.int(arg1_1, 0, 83)
        slice_330 = torch.ops.aten.slice.Tensor(select_752, 0, 83, 9223372036854775807);  select_752 = None
        slice_331 = torch.ops.aten.slice.Tensor(slice_scatter_81, 1, 83, 9223372036854775807)
        expand_167 = torch.ops.aten.expand.default(slice_331, [2048, 45]);  slice_331 = None
        mul_249 = torch.ops.aten.mul.Tensor(expand_167, 1);  expand_167 = None
        view_334 = torch.ops.aten.view.default(div_83, [2048, 1]);  div_83 = None
        mul_250 = torch.ops.aten.mul.Tensor(view_334, slice_330);  view_334 = slice_330 = None
        mul_251 = torch.ops.aten.mul.Tensor(mul_250, -1.0);  mul_250 = None
        add_83 = torch.ops.aten.add.Tensor(mul_249, mul_251);  mul_249 = mul_251 = None
        slice_scatter_82 = torch.ops.aten.slice_scatter.default(slice_scatter_81, add_83, 1, 83, 9223372036854775807);  slice_scatter_81 = add_83 = None
        select_754 = torch.ops.aten.select.int(arg1_1, 0, 84)
        select_755 = torch.ops.aten.select.int(select_754, 0, 84);  select_754 = None
        select_756 = torch.ops.aten.select.int(slice_scatter_82, 1, 84)
        unsqueeze_168 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_84 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_757 = torch.ops.aten.select.int(slice_scatter_82, 1, 84)
        view_336 = torch.ops.aten.view.default(select_757, [8, 128, 2]);  select_757 = None
        baddbmm_84 = torch.ops.aten.baddbmm.default(unsqueeze_168, view_336, permute_84, alpha = -2.0);  unsqueeze_168 = view_336 = permute_84 = None
        argmin_84 = torch.ops.aten.argmin.default(baddbmm_84, -1);  baddbmm_84 = None
        unsqueeze_169 = torch.ops.aten.unsqueeze.default(argmin_84, -1)
        expand_168 = torch.ops.aten.expand.default(unsqueeze_169, [8, 128, 2]);  unsqueeze_169 = None
        gather_84 = torch.ops.aten.gather.default(arg3_1, 1, expand_168);  expand_168 = None
        view_337 = torch.ops.aten.view.default(gather_84, [-1]);  gather_84 = None
        select_759 = torch.ops.aten.select.int(select_scatter_83, 1, 84)
        copy_84 = torch.ops.aten.copy.default(select_759, argmin_84);  select_759 = argmin_84 = None
        select_scatter_84 = torch.ops.aten.select_scatter.default(select_scatter_83, copy_84, 1, 84);  select_scatter_83 = copy_84 = None
        sub_84 = torch.ops.aten.sub.Tensor(select_756, view_337);  select_756 = view_337 = None
        div_84 = torch.ops.aten.div.Tensor(sub_84, select_755);  sub_84 = select_755 = None
        select_761 = torch.ops.aten.select.int(arg1_1, 0, 84)
        slice_334 = torch.ops.aten.slice.Tensor(select_761, 0, 84, 9223372036854775807);  select_761 = None
        slice_335 = torch.ops.aten.slice.Tensor(slice_scatter_82, 1, 84, 9223372036854775807)
        expand_169 = torch.ops.aten.expand.default(slice_335, [2048, 44]);  slice_335 = None
        mul_252 = torch.ops.aten.mul.Tensor(expand_169, 1);  expand_169 = None
        view_338 = torch.ops.aten.view.default(div_84, [2048, 1]);  div_84 = None
        mul_253 = torch.ops.aten.mul.Tensor(view_338, slice_334);  view_338 = slice_334 = None
        mul_254 = torch.ops.aten.mul.Tensor(mul_253, -1.0);  mul_253 = None
        add_84 = torch.ops.aten.add.Tensor(mul_252, mul_254);  mul_252 = mul_254 = None
        slice_scatter_83 = torch.ops.aten.slice_scatter.default(slice_scatter_82, add_84, 1, 84, 9223372036854775807);  slice_scatter_82 = add_84 = None
        select_763 = torch.ops.aten.select.int(arg1_1, 0, 85)
        select_764 = torch.ops.aten.select.int(select_763, 0, 85);  select_763 = None
        select_765 = torch.ops.aten.select.int(slice_scatter_83, 1, 85)
        unsqueeze_170 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_85 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_766 = torch.ops.aten.select.int(slice_scatter_83, 1, 85)
        view_340 = torch.ops.aten.view.default(select_766, [8, 128, 2]);  select_766 = None
        baddbmm_85 = torch.ops.aten.baddbmm.default(unsqueeze_170, view_340, permute_85, alpha = -2.0);  unsqueeze_170 = view_340 = permute_85 = None
        argmin_85 = torch.ops.aten.argmin.default(baddbmm_85, -1);  baddbmm_85 = None
        unsqueeze_171 = torch.ops.aten.unsqueeze.default(argmin_85, -1)
        expand_170 = torch.ops.aten.expand.default(unsqueeze_171, [8, 128, 2]);  unsqueeze_171 = None
        gather_85 = torch.ops.aten.gather.default(arg3_1, 1, expand_170);  expand_170 = None
        view_341 = torch.ops.aten.view.default(gather_85, [-1]);  gather_85 = None
        select_768 = torch.ops.aten.select.int(select_scatter_84, 1, 85)
        copy_85 = torch.ops.aten.copy.default(select_768, argmin_85);  select_768 = argmin_85 = None
        select_scatter_85 = torch.ops.aten.select_scatter.default(select_scatter_84, copy_85, 1, 85);  select_scatter_84 = copy_85 = None
        sub_85 = torch.ops.aten.sub.Tensor(select_765, view_341);  select_765 = view_341 = None
        div_85 = torch.ops.aten.div.Tensor(sub_85, select_764);  sub_85 = select_764 = None
        select_770 = torch.ops.aten.select.int(arg1_1, 0, 85)
        slice_338 = torch.ops.aten.slice.Tensor(select_770, 0, 85, 9223372036854775807);  select_770 = None
        slice_339 = torch.ops.aten.slice.Tensor(slice_scatter_83, 1, 85, 9223372036854775807)
        expand_171 = torch.ops.aten.expand.default(slice_339, [2048, 43]);  slice_339 = None
        mul_255 = torch.ops.aten.mul.Tensor(expand_171, 1);  expand_171 = None
        view_342 = torch.ops.aten.view.default(div_85, [2048, 1]);  div_85 = None
        mul_256 = torch.ops.aten.mul.Tensor(view_342, slice_338);  view_342 = slice_338 = None
        mul_257 = torch.ops.aten.mul.Tensor(mul_256, -1.0);  mul_256 = None
        add_85 = torch.ops.aten.add.Tensor(mul_255, mul_257);  mul_255 = mul_257 = None
        slice_scatter_84 = torch.ops.aten.slice_scatter.default(slice_scatter_83, add_85, 1, 85, 9223372036854775807);  slice_scatter_83 = add_85 = None
        select_772 = torch.ops.aten.select.int(arg1_1, 0, 86)
        select_773 = torch.ops.aten.select.int(select_772, 0, 86);  select_772 = None
        select_774 = torch.ops.aten.select.int(slice_scatter_84, 1, 86)
        unsqueeze_172 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_86 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_775 = torch.ops.aten.select.int(slice_scatter_84, 1, 86)
        view_344 = torch.ops.aten.view.default(select_775, [8, 128, 2]);  select_775 = None
        baddbmm_86 = torch.ops.aten.baddbmm.default(unsqueeze_172, view_344, permute_86, alpha = -2.0);  unsqueeze_172 = view_344 = permute_86 = None
        argmin_86 = torch.ops.aten.argmin.default(baddbmm_86, -1);  baddbmm_86 = None
        unsqueeze_173 = torch.ops.aten.unsqueeze.default(argmin_86, -1)
        expand_172 = torch.ops.aten.expand.default(unsqueeze_173, [8, 128, 2]);  unsqueeze_173 = None
        gather_86 = torch.ops.aten.gather.default(arg3_1, 1, expand_172);  expand_172 = None
        view_345 = torch.ops.aten.view.default(gather_86, [-1]);  gather_86 = None
        select_777 = torch.ops.aten.select.int(select_scatter_85, 1, 86)
        copy_86 = torch.ops.aten.copy.default(select_777, argmin_86);  select_777 = argmin_86 = None
        select_scatter_86 = torch.ops.aten.select_scatter.default(select_scatter_85, copy_86, 1, 86);  select_scatter_85 = copy_86 = None
        sub_86 = torch.ops.aten.sub.Tensor(select_774, view_345);  select_774 = view_345 = None
        div_86 = torch.ops.aten.div.Tensor(sub_86, select_773);  sub_86 = select_773 = None
        select_779 = torch.ops.aten.select.int(arg1_1, 0, 86)
        slice_342 = torch.ops.aten.slice.Tensor(select_779, 0, 86, 9223372036854775807);  select_779 = None
        slice_343 = torch.ops.aten.slice.Tensor(slice_scatter_84, 1, 86, 9223372036854775807)
        expand_173 = torch.ops.aten.expand.default(slice_343, [2048, 42]);  slice_343 = None
        mul_258 = torch.ops.aten.mul.Tensor(expand_173, 1);  expand_173 = None
        view_346 = torch.ops.aten.view.default(div_86, [2048, 1]);  div_86 = None
        mul_259 = torch.ops.aten.mul.Tensor(view_346, slice_342);  view_346 = slice_342 = None
        mul_260 = torch.ops.aten.mul.Tensor(mul_259, -1.0);  mul_259 = None
        add_86 = torch.ops.aten.add.Tensor(mul_258, mul_260);  mul_258 = mul_260 = None
        slice_scatter_85 = torch.ops.aten.slice_scatter.default(slice_scatter_84, add_86, 1, 86, 9223372036854775807);  slice_scatter_84 = add_86 = None
        select_781 = torch.ops.aten.select.int(arg1_1, 0, 87)
        select_782 = torch.ops.aten.select.int(select_781, 0, 87);  select_781 = None
        select_783 = torch.ops.aten.select.int(slice_scatter_85, 1, 87)
        unsqueeze_174 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_87 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_784 = torch.ops.aten.select.int(slice_scatter_85, 1, 87)
        view_348 = torch.ops.aten.view.default(select_784, [8, 128, 2]);  select_784 = None
        baddbmm_87 = torch.ops.aten.baddbmm.default(unsqueeze_174, view_348, permute_87, alpha = -2.0);  unsqueeze_174 = view_348 = permute_87 = None
        argmin_87 = torch.ops.aten.argmin.default(baddbmm_87, -1);  baddbmm_87 = None
        unsqueeze_175 = torch.ops.aten.unsqueeze.default(argmin_87, -1)
        expand_174 = torch.ops.aten.expand.default(unsqueeze_175, [8, 128, 2]);  unsqueeze_175 = None
        gather_87 = torch.ops.aten.gather.default(arg3_1, 1, expand_174);  expand_174 = None
        view_349 = torch.ops.aten.view.default(gather_87, [-1]);  gather_87 = None
        select_786 = torch.ops.aten.select.int(select_scatter_86, 1, 87)
        copy_87 = torch.ops.aten.copy.default(select_786, argmin_87);  select_786 = argmin_87 = None
        select_scatter_87 = torch.ops.aten.select_scatter.default(select_scatter_86, copy_87, 1, 87);  select_scatter_86 = copy_87 = None
        sub_87 = torch.ops.aten.sub.Tensor(select_783, view_349);  select_783 = view_349 = None
        div_87 = torch.ops.aten.div.Tensor(sub_87, select_782);  sub_87 = select_782 = None
        select_788 = torch.ops.aten.select.int(arg1_1, 0, 87)
        slice_346 = torch.ops.aten.slice.Tensor(select_788, 0, 87, 9223372036854775807);  select_788 = None
        slice_347 = torch.ops.aten.slice.Tensor(slice_scatter_85, 1, 87, 9223372036854775807)
        expand_175 = torch.ops.aten.expand.default(slice_347, [2048, 41]);  slice_347 = None
        mul_261 = torch.ops.aten.mul.Tensor(expand_175, 1);  expand_175 = None
        view_350 = torch.ops.aten.view.default(div_87, [2048, 1]);  div_87 = None
        mul_262 = torch.ops.aten.mul.Tensor(view_350, slice_346);  view_350 = slice_346 = None
        mul_263 = torch.ops.aten.mul.Tensor(mul_262, -1.0);  mul_262 = None
        add_87 = torch.ops.aten.add.Tensor(mul_261, mul_263);  mul_261 = mul_263 = None
        slice_scatter_86 = torch.ops.aten.slice_scatter.default(slice_scatter_85, add_87, 1, 87, 9223372036854775807);  slice_scatter_85 = add_87 = None
        select_790 = torch.ops.aten.select.int(arg1_1, 0, 88)
        select_791 = torch.ops.aten.select.int(select_790, 0, 88);  select_790 = None
        select_792 = torch.ops.aten.select.int(slice_scatter_86, 1, 88)
        unsqueeze_176 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_88 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_793 = torch.ops.aten.select.int(slice_scatter_86, 1, 88)
        view_352 = torch.ops.aten.view.default(select_793, [8, 128, 2]);  select_793 = None
        baddbmm_88 = torch.ops.aten.baddbmm.default(unsqueeze_176, view_352, permute_88, alpha = -2.0);  unsqueeze_176 = view_352 = permute_88 = None
        argmin_88 = torch.ops.aten.argmin.default(baddbmm_88, -1);  baddbmm_88 = None
        unsqueeze_177 = torch.ops.aten.unsqueeze.default(argmin_88, -1)
        expand_176 = torch.ops.aten.expand.default(unsqueeze_177, [8, 128, 2]);  unsqueeze_177 = None
        gather_88 = torch.ops.aten.gather.default(arg3_1, 1, expand_176);  expand_176 = None
        view_353 = torch.ops.aten.view.default(gather_88, [-1]);  gather_88 = None
        select_795 = torch.ops.aten.select.int(select_scatter_87, 1, 88)
        copy_88 = torch.ops.aten.copy.default(select_795, argmin_88);  select_795 = argmin_88 = None
        select_scatter_88 = torch.ops.aten.select_scatter.default(select_scatter_87, copy_88, 1, 88);  select_scatter_87 = copy_88 = None
        sub_88 = torch.ops.aten.sub.Tensor(select_792, view_353);  select_792 = view_353 = None
        div_88 = torch.ops.aten.div.Tensor(sub_88, select_791);  sub_88 = select_791 = None
        select_797 = torch.ops.aten.select.int(arg1_1, 0, 88)
        slice_350 = torch.ops.aten.slice.Tensor(select_797, 0, 88, 9223372036854775807);  select_797 = None
        slice_351 = torch.ops.aten.slice.Tensor(slice_scatter_86, 1, 88, 9223372036854775807)
        expand_177 = torch.ops.aten.expand.default(slice_351, [2048, 40]);  slice_351 = None
        mul_264 = torch.ops.aten.mul.Tensor(expand_177, 1);  expand_177 = None
        view_354 = torch.ops.aten.view.default(div_88, [2048, 1]);  div_88 = None
        mul_265 = torch.ops.aten.mul.Tensor(view_354, slice_350);  view_354 = slice_350 = None
        mul_266 = torch.ops.aten.mul.Tensor(mul_265, -1.0);  mul_265 = None
        add_88 = torch.ops.aten.add.Tensor(mul_264, mul_266);  mul_264 = mul_266 = None
        slice_scatter_87 = torch.ops.aten.slice_scatter.default(slice_scatter_86, add_88, 1, 88, 9223372036854775807);  slice_scatter_86 = add_88 = None
        select_799 = torch.ops.aten.select.int(arg1_1, 0, 89)
        select_800 = torch.ops.aten.select.int(select_799, 0, 89);  select_799 = None
        select_801 = torch.ops.aten.select.int(slice_scatter_87, 1, 89)
        unsqueeze_178 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_89 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_802 = torch.ops.aten.select.int(slice_scatter_87, 1, 89)
        view_356 = torch.ops.aten.view.default(select_802, [8, 128, 2]);  select_802 = None
        baddbmm_89 = torch.ops.aten.baddbmm.default(unsqueeze_178, view_356, permute_89, alpha = -2.0);  unsqueeze_178 = view_356 = permute_89 = None
        argmin_89 = torch.ops.aten.argmin.default(baddbmm_89, -1);  baddbmm_89 = None
        unsqueeze_179 = torch.ops.aten.unsqueeze.default(argmin_89, -1)
        expand_178 = torch.ops.aten.expand.default(unsqueeze_179, [8, 128, 2]);  unsqueeze_179 = None
        gather_89 = torch.ops.aten.gather.default(arg3_1, 1, expand_178);  expand_178 = None
        view_357 = torch.ops.aten.view.default(gather_89, [-1]);  gather_89 = None
        select_804 = torch.ops.aten.select.int(select_scatter_88, 1, 89)
        copy_89 = torch.ops.aten.copy.default(select_804, argmin_89);  select_804 = argmin_89 = None
        select_scatter_89 = torch.ops.aten.select_scatter.default(select_scatter_88, copy_89, 1, 89);  select_scatter_88 = copy_89 = None
        sub_89 = torch.ops.aten.sub.Tensor(select_801, view_357);  select_801 = view_357 = None
        div_89 = torch.ops.aten.div.Tensor(sub_89, select_800);  sub_89 = select_800 = None
        select_806 = torch.ops.aten.select.int(arg1_1, 0, 89)
        slice_354 = torch.ops.aten.slice.Tensor(select_806, 0, 89, 9223372036854775807);  select_806 = None
        slice_355 = torch.ops.aten.slice.Tensor(slice_scatter_87, 1, 89, 9223372036854775807)
        expand_179 = torch.ops.aten.expand.default(slice_355, [2048, 39]);  slice_355 = None
        mul_267 = torch.ops.aten.mul.Tensor(expand_179, 1);  expand_179 = None
        view_358 = torch.ops.aten.view.default(div_89, [2048, 1]);  div_89 = None
        mul_268 = torch.ops.aten.mul.Tensor(view_358, slice_354);  view_358 = slice_354 = None
        mul_269 = torch.ops.aten.mul.Tensor(mul_268, -1.0);  mul_268 = None
        add_89 = torch.ops.aten.add.Tensor(mul_267, mul_269);  mul_267 = mul_269 = None
        slice_scatter_88 = torch.ops.aten.slice_scatter.default(slice_scatter_87, add_89, 1, 89, 9223372036854775807);  slice_scatter_87 = add_89 = None
        select_808 = torch.ops.aten.select.int(arg1_1, 0, 90)
        select_809 = torch.ops.aten.select.int(select_808, 0, 90);  select_808 = None
        select_810 = torch.ops.aten.select.int(slice_scatter_88, 1, 90)
        unsqueeze_180 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_90 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_811 = torch.ops.aten.select.int(slice_scatter_88, 1, 90)
        view_360 = torch.ops.aten.view.default(select_811, [8, 128, 2]);  select_811 = None
        baddbmm_90 = torch.ops.aten.baddbmm.default(unsqueeze_180, view_360, permute_90, alpha = -2.0);  unsqueeze_180 = view_360 = permute_90 = None
        argmin_90 = torch.ops.aten.argmin.default(baddbmm_90, -1);  baddbmm_90 = None
        unsqueeze_181 = torch.ops.aten.unsqueeze.default(argmin_90, -1)
        expand_180 = torch.ops.aten.expand.default(unsqueeze_181, [8, 128, 2]);  unsqueeze_181 = None
        gather_90 = torch.ops.aten.gather.default(arg3_1, 1, expand_180);  expand_180 = None
        view_361 = torch.ops.aten.view.default(gather_90, [-1]);  gather_90 = None
        select_813 = torch.ops.aten.select.int(select_scatter_89, 1, 90)
        copy_90 = torch.ops.aten.copy.default(select_813, argmin_90);  select_813 = argmin_90 = None
        select_scatter_90 = torch.ops.aten.select_scatter.default(select_scatter_89, copy_90, 1, 90);  select_scatter_89 = copy_90 = None
        sub_90 = torch.ops.aten.sub.Tensor(select_810, view_361);  select_810 = view_361 = None
        div_90 = torch.ops.aten.div.Tensor(sub_90, select_809);  sub_90 = select_809 = None
        select_815 = torch.ops.aten.select.int(arg1_1, 0, 90)
        slice_358 = torch.ops.aten.slice.Tensor(select_815, 0, 90, 9223372036854775807);  select_815 = None
        slice_359 = torch.ops.aten.slice.Tensor(slice_scatter_88, 1, 90, 9223372036854775807)
        expand_181 = torch.ops.aten.expand.default(slice_359, [2048, 38]);  slice_359 = None
        mul_270 = torch.ops.aten.mul.Tensor(expand_181, 1);  expand_181 = None
        view_362 = torch.ops.aten.view.default(div_90, [2048, 1]);  div_90 = None
        mul_271 = torch.ops.aten.mul.Tensor(view_362, slice_358);  view_362 = slice_358 = None
        mul_272 = torch.ops.aten.mul.Tensor(mul_271, -1.0);  mul_271 = None
        add_90 = torch.ops.aten.add.Tensor(mul_270, mul_272);  mul_270 = mul_272 = None
        slice_scatter_89 = torch.ops.aten.slice_scatter.default(slice_scatter_88, add_90, 1, 90, 9223372036854775807);  slice_scatter_88 = add_90 = None
        select_817 = torch.ops.aten.select.int(arg1_1, 0, 91)
        select_818 = torch.ops.aten.select.int(select_817, 0, 91);  select_817 = None
        select_819 = torch.ops.aten.select.int(slice_scatter_89, 1, 91)
        unsqueeze_182 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_91 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_820 = torch.ops.aten.select.int(slice_scatter_89, 1, 91)
        view_364 = torch.ops.aten.view.default(select_820, [8, 128, 2]);  select_820 = None
        baddbmm_91 = torch.ops.aten.baddbmm.default(unsqueeze_182, view_364, permute_91, alpha = -2.0);  unsqueeze_182 = view_364 = permute_91 = None
        argmin_91 = torch.ops.aten.argmin.default(baddbmm_91, -1);  baddbmm_91 = None
        unsqueeze_183 = torch.ops.aten.unsqueeze.default(argmin_91, -1)
        expand_182 = torch.ops.aten.expand.default(unsqueeze_183, [8, 128, 2]);  unsqueeze_183 = None
        gather_91 = torch.ops.aten.gather.default(arg3_1, 1, expand_182);  expand_182 = None
        view_365 = torch.ops.aten.view.default(gather_91, [-1]);  gather_91 = None
        select_822 = torch.ops.aten.select.int(select_scatter_90, 1, 91)
        copy_91 = torch.ops.aten.copy.default(select_822, argmin_91);  select_822 = argmin_91 = None
        select_scatter_91 = torch.ops.aten.select_scatter.default(select_scatter_90, copy_91, 1, 91);  select_scatter_90 = copy_91 = None
        sub_91 = torch.ops.aten.sub.Tensor(select_819, view_365);  select_819 = view_365 = None
        div_91 = torch.ops.aten.div.Tensor(sub_91, select_818);  sub_91 = select_818 = None
        select_824 = torch.ops.aten.select.int(arg1_1, 0, 91)
        slice_362 = torch.ops.aten.slice.Tensor(select_824, 0, 91, 9223372036854775807);  select_824 = None
        slice_363 = torch.ops.aten.slice.Tensor(slice_scatter_89, 1, 91, 9223372036854775807)
        expand_183 = torch.ops.aten.expand.default(slice_363, [2048, 37]);  slice_363 = None
        mul_273 = torch.ops.aten.mul.Tensor(expand_183, 1);  expand_183 = None
        view_366 = torch.ops.aten.view.default(div_91, [2048, 1]);  div_91 = None
        mul_274 = torch.ops.aten.mul.Tensor(view_366, slice_362);  view_366 = slice_362 = None
        mul_275 = torch.ops.aten.mul.Tensor(mul_274, -1.0);  mul_274 = None
        add_91 = torch.ops.aten.add.Tensor(mul_273, mul_275);  mul_273 = mul_275 = None
        slice_scatter_90 = torch.ops.aten.slice_scatter.default(slice_scatter_89, add_91, 1, 91, 9223372036854775807);  slice_scatter_89 = add_91 = None
        select_826 = torch.ops.aten.select.int(arg1_1, 0, 92)
        select_827 = torch.ops.aten.select.int(select_826, 0, 92);  select_826 = None
        select_828 = torch.ops.aten.select.int(slice_scatter_90, 1, 92)
        unsqueeze_184 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_92 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_829 = torch.ops.aten.select.int(slice_scatter_90, 1, 92)
        view_368 = torch.ops.aten.view.default(select_829, [8, 128, 2]);  select_829 = None
        baddbmm_92 = torch.ops.aten.baddbmm.default(unsqueeze_184, view_368, permute_92, alpha = -2.0);  unsqueeze_184 = view_368 = permute_92 = None
        argmin_92 = torch.ops.aten.argmin.default(baddbmm_92, -1);  baddbmm_92 = None
        unsqueeze_185 = torch.ops.aten.unsqueeze.default(argmin_92, -1)
        expand_184 = torch.ops.aten.expand.default(unsqueeze_185, [8, 128, 2]);  unsqueeze_185 = None
        gather_92 = torch.ops.aten.gather.default(arg3_1, 1, expand_184);  expand_184 = None
        view_369 = torch.ops.aten.view.default(gather_92, [-1]);  gather_92 = None
        select_831 = torch.ops.aten.select.int(select_scatter_91, 1, 92)
        copy_92 = torch.ops.aten.copy.default(select_831, argmin_92);  select_831 = argmin_92 = None
        select_scatter_92 = torch.ops.aten.select_scatter.default(select_scatter_91, copy_92, 1, 92);  select_scatter_91 = copy_92 = None
        sub_92 = torch.ops.aten.sub.Tensor(select_828, view_369);  select_828 = view_369 = None
        div_92 = torch.ops.aten.div.Tensor(sub_92, select_827);  sub_92 = select_827 = None
        select_833 = torch.ops.aten.select.int(arg1_1, 0, 92)
        slice_366 = torch.ops.aten.slice.Tensor(select_833, 0, 92, 9223372036854775807);  select_833 = None
        slice_367 = torch.ops.aten.slice.Tensor(slice_scatter_90, 1, 92, 9223372036854775807)
        expand_185 = torch.ops.aten.expand.default(slice_367, [2048, 36]);  slice_367 = None
        mul_276 = torch.ops.aten.mul.Tensor(expand_185, 1);  expand_185 = None
        view_370 = torch.ops.aten.view.default(div_92, [2048, 1]);  div_92 = None
        mul_277 = torch.ops.aten.mul.Tensor(view_370, slice_366);  view_370 = slice_366 = None
        mul_278 = torch.ops.aten.mul.Tensor(mul_277, -1.0);  mul_277 = None
        add_92 = torch.ops.aten.add.Tensor(mul_276, mul_278);  mul_276 = mul_278 = None
        slice_scatter_91 = torch.ops.aten.slice_scatter.default(slice_scatter_90, add_92, 1, 92, 9223372036854775807);  slice_scatter_90 = add_92 = None
        select_835 = torch.ops.aten.select.int(arg1_1, 0, 93)
        select_836 = torch.ops.aten.select.int(select_835, 0, 93);  select_835 = None
        select_837 = torch.ops.aten.select.int(slice_scatter_91, 1, 93)
        unsqueeze_186 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_93 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_838 = torch.ops.aten.select.int(slice_scatter_91, 1, 93)
        view_372 = torch.ops.aten.view.default(select_838, [8, 128, 2]);  select_838 = None
        baddbmm_93 = torch.ops.aten.baddbmm.default(unsqueeze_186, view_372, permute_93, alpha = -2.0);  unsqueeze_186 = view_372 = permute_93 = None
        argmin_93 = torch.ops.aten.argmin.default(baddbmm_93, -1);  baddbmm_93 = None
        unsqueeze_187 = torch.ops.aten.unsqueeze.default(argmin_93, -1)
        expand_186 = torch.ops.aten.expand.default(unsqueeze_187, [8, 128, 2]);  unsqueeze_187 = None
        gather_93 = torch.ops.aten.gather.default(arg3_1, 1, expand_186);  expand_186 = None
        view_373 = torch.ops.aten.view.default(gather_93, [-1]);  gather_93 = None
        select_840 = torch.ops.aten.select.int(select_scatter_92, 1, 93)
        copy_93 = torch.ops.aten.copy.default(select_840, argmin_93);  select_840 = argmin_93 = None
        select_scatter_93 = torch.ops.aten.select_scatter.default(select_scatter_92, copy_93, 1, 93);  select_scatter_92 = copy_93 = None
        sub_93 = torch.ops.aten.sub.Tensor(select_837, view_373);  select_837 = view_373 = None
        div_93 = torch.ops.aten.div.Tensor(sub_93, select_836);  sub_93 = select_836 = None
        select_842 = torch.ops.aten.select.int(arg1_1, 0, 93)
        slice_370 = torch.ops.aten.slice.Tensor(select_842, 0, 93, 9223372036854775807);  select_842 = None
        slice_371 = torch.ops.aten.slice.Tensor(slice_scatter_91, 1, 93, 9223372036854775807)
        expand_187 = torch.ops.aten.expand.default(slice_371, [2048, 35]);  slice_371 = None
        mul_279 = torch.ops.aten.mul.Tensor(expand_187, 1);  expand_187 = None
        view_374 = torch.ops.aten.view.default(div_93, [2048, 1]);  div_93 = None
        mul_280 = torch.ops.aten.mul.Tensor(view_374, slice_370);  view_374 = slice_370 = None
        mul_281 = torch.ops.aten.mul.Tensor(mul_280, -1.0);  mul_280 = None
        add_93 = torch.ops.aten.add.Tensor(mul_279, mul_281);  mul_279 = mul_281 = None
        slice_scatter_92 = torch.ops.aten.slice_scatter.default(slice_scatter_91, add_93, 1, 93, 9223372036854775807);  slice_scatter_91 = add_93 = None
        select_844 = torch.ops.aten.select.int(arg1_1, 0, 94)
        select_845 = torch.ops.aten.select.int(select_844, 0, 94);  select_844 = None
        select_846 = torch.ops.aten.select.int(slice_scatter_92, 1, 94)
        unsqueeze_188 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_94 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_847 = torch.ops.aten.select.int(slice_scatter_92, 1, 94)
        view_376 = torch.ops.aten.view.default(select_847, [8, 128, 2]);  select_847 = None
        baddbmm_94 = torch.ops.aten.baddbmm.default(unsqueeze_188, view_376, permute_94, alpha = -2.0);  unsqueeze_188 = view_376 = permute_94 = None
        argmin_94 = torch.ops.aten.argmin.default(baddbmm_94, -1);  baddbmm_94 = None
        unsqueeze_189 = torch.ops.aten.unsqueeze.default(argmin_94, -1)
        expand_188 = torch.ops.aten.expand.default(unsqueeze_189, [8, 128, 2]);  unsqueeze_189 = None
        gather_94 = torch.ops.aten.gather.default(arg3_1, 1, expand_188);  expand_188 = None
        view_377 = torch.ops.aten.view.default(gather_94, [-1]);  gather_94 = None
        select_849 = torch.ops.aten.select.int(select_scatter_93, 1, 94)
        copy_94 = torch.ops.aten.copy.default(select_849, argmin_94);  select_849 = argmin_94 = None
        select_scatter_94 = torch.ops.aten.select_scatter.default(select_scatter_93, copy_94, 1, 94);  select_scatter_93 = copy_94 = None
        sub_94 = torch.ops.aten.sub.Tensor(select_846, view_377);  select_846 = view_377 = None
        div_94 = torch.ops.aten.div.Tensor(sub_94, select_845);  sub_94 = select_845 = None
        select_851 = torch.ops.aten.select.int(arg1_1, 0, 94)
        slice_374 = torch.ops.aten.slice.Tensor(select_851, 0, 94, 9223372036854775807);  select_851 = None
        slice_375 = torch.ops.aten.slice.Tensor(slice_scatter_92, 1, 94, 9223372036854775807)
        expand_189 = torch.ops.aten.expand.default(slice_375, [2048, 34]);  slice_375 = None
        mul_282 = torch.ops.aten.mul.Tensor(expand_189, 1);  expand_189 = None
        view_378 = torch.ops.aten.view.default(div_94, [2048, 1]);  div_94 = None
        mul_283 = torch.ops.aten.mul.Tensor(view_378, slice_374);  view_378 = slice_374 = None
        mul_284 = torch.ops.aten.mul.Tensor(mul_283, -1.0);  mul_283 = None
        add_94 = torch.ops.aten.add.Tensor(mul_282, mul_284);  mul_282 = mul_284 = None
        slice_scatter_93 = torch.ops.aten.slice_scatter.default(slice_scatter_92, add_94, 1, 94, 9223372036854775807);  slice_scatter_92 = add_94 = None
        select_853 = torch.ops.aten.select.int(arg1_1, 0, 95)
        select_854 = torch.ops.aten.select.int(select_853, 0, 95);  select_853 = None
        select_855 = torch.ops.aten.select.int(slice_scatter_93, 1, 95)
        unsqueeze_190 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_95 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_856 = torch.ops.aten.select.int(slice_scatter_93, 1, 95)
        view_380 = torch.ops.aten.view.default(select_856, [8, 128, 2]);  select_856 = None
        baddbmm_95 = torch.ops.aten.baddbmm.default(unsqueeze_190, view_380, permute_95, alpha = -2.0);  unsqueeze_190 = view_380 = permute_95 = None
        argmin_95 = torch.ops.aten.argmin.default(baddbmm_95, -1);  baddbmm_95 = None
        unsqueeze_191 = torch.ops.aten.unsqueeze.default(argmin_95, -1)
        expand_190 = torch.ops.aten.expand.default(unsqueeze_191, [8, 128, 2]);  unsqueeze_191 = None
        gather_95 = torch.ops.aten.gather.default(arg3_1, 1, expand_190);  expand_190 = None
        view_381 = torch.ops.aten.view.default(gather_95, [-1]);  gather_95 = None
        select_858 = torch.ops.aten.select.int(select_scatter_94, 1, 95)
        copy_95 = torch.ops.aten.copy.default(select_858, argmin_95);  select_858 = argmin_95 = None
        select_scatter_95 = torch.ops.aten.select_scatter.default(select_scatter_94, copy_95, 1, 95);  select_scatter_94 = copy_95 = None
        sub_95 = torch.ops.aten.sub.Tensor(select_855, view_381);  select_855 = view_381 = None
        div_95 = torch.ops.aten.div.Tensor(sub_95, select_854);  sub_95 = select_854 = None
        select_860 = torch.ops.aten.select.int(arg1_1, 0, 95)
        slice_378 = torch.ops.aten.slice.Tensor(select_860, 0, 95, 9223372036854775807);  select_860 = None
        slice_379 = torch.ops.aten.slice.Tensor(slice_scatter_93, 1, 95, 9223372036854775807)
        expand_191 = torch.ops.aten.expand.default(slice_379, [2048, 33]);  slice_379 = None
        mul_285 = torch.ops.aten.mul.Tensor(expand_191, 1);  expand_191 = None
        view_382 = torch.ops.aten.view.default(div_95, [2048, 1]);  div_95 = None
        mul_286 = torch.ops.aten.mul.Tensor(view_382, slice_378);  view_382 = slice_378 = None
        mul_287 = torch.ops.aten.mul.Tensor(mul_286, -1.0);  mul_286 = None
        add_95 = torch.ops.aten.add.Tensor(mul_285, mul_287);  mul_285 = mul_287 = None
        slice_scatter_94 = torch.ops.aten.slice_scatter.default(slice_scatter_93, add_95, 1, 95, 9223372036854775807);  slice_scatter_93 = add_95 = None
        select_862 = torch.ops.aten.select.int(arg1_1, 0, 96)
        select_863 = torch.ops.aten.select.int(select_862, 0, 96);  select_862 = None
        select_864 = torch.ops.aten.select.int(slice_scatter_94, 1, 96)
        unsqueeze_192 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_96 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_865 = torch.ops.aten.select.int(slice_scatter_94, 1, 96)
        view_384 = torch.ops.aten.view.default(select_865, [8, 128, 2]);  select_865 = None
        baddbmm_96 = torch.ops.aten.baddbmm.default(unsqueeze_192, view_384, permute_96, alpha = -2.0);  unsqueeze_192 = view_384 = permute_96 = None
        argmin_96 = torch.ops.aten.argmin.default(baddbmm_96, -1);  baddbmm_96 = None
        unsqueeze_193 = torch.ops.aten.unsqueeze.default(argmin_96, -1)
        expand_192 = torch.ops.aten.expand.default(unsqueeze_193, [8, 128, 2]);  unsqueeze_193 = None
        gather_96 = torch.ops.aten.gather.default(arg3_1, 1, expand_192);  expand_192 = None
        view_385 = torch.ops.aten.view.default(gather_96, [-1]);  gather_96 = None
        select_867 = torch.ops.aten.select.int(select_scatter_95, 1, 96)
        copy_96 = torch.ops.aten.copy.default(select_867, argmin_96);  select_867 = argmin_96 = None
        select_scatter_96 = torch.ops.aten.select_scatter.default(select_scatter_95, copy_96, 1, 96);  select_scatter_95 = copy_96 = None
        sub_96 = torch.ops.aten.sub.Tensor(select_864, view_385);  select_864 = view_385 = None
        div_96 = torch.ops.aten.div.Tensor(sub_96, select_863);  sub_96 = select_863 = None
        select_869 = torch.ops.aten.select.int(arg1_1, 0, 96)
        slice_382 = torch.ops.aten.slice.Tensor(select_869, 0, 96, 9223372036854775807);  select_869 = None
        slice_383 = torch.ops.aten.slice.Tensor(slice_scatter_94, 1, 96, 9223372036854775807)
        expand_193 = torch.ops.aten.expand.default(slice_383, [2048, 32]);  slice_383 = None
        mul_288 = torch.ops.aten.mul.Tensor(expand_193, 1);  expand_193 = None
        view_386 = torch.ops.aten.view.default(div_96, [2048, 1]);  div_96 = None
        mul_289 = torch.ops.aten.mul.Tensor(view_386, slice_382);  view_386 = slice_382 = None
        mul_290 = torch.ops.aten.mul.Tensor(mul_289, -1.0);  mul_289 = None
        add_96 = torch.ops.aten.add.Tensor(mul_288, mul_290);  mul_288 = mul_290 = None
        slice_scatter_95 = torch.ops.aten.slice_scatter.default(slice_scatter_94, add_96, 1, 96, 9223372036854775807);  slice_scatter_94 = add_96 = None
        select_871 = torch.ops.aten.select.int(arg1_1, 0, 97)
        select_872 = torch.ops.aten.select.int(select_871, 0, 97);  select_871 = None
        select_873 = torch.ops.aten.select.int(slice_scatter_95, 1, 97)
        unsqueeze_194 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_97 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_874 = torch.ops.aten.select.int(slice_scatter_95, 1, 97)
        view_388 = torch.ops.aten.view.default(select_874, [8, 128, 2]);  select_874 = None
        baddbmm_97 = torch.ops.aten.baddbmm.default(unsqueeze_194, view_388, permute_97, alpha = -2.0);  unsqueeze_194 = view_388 = permute_97 = None
        argmin_97 = torch.ops.aten.argmin.default(baddbmm_97, -1);  baddbmm_97 = None
        unsqueeze_195 = torch.ops.aten.unsqueeze.default(argmin_97, -1)
        expand_194 = torch.ops.aten.expand.default(unsqueeze_195, [8, 128, 2]);  unsqueeze_195 = None
        gather_97 = torch.ops.aten.gather.default(arg3_1, 1, expand_194);  expand_194 = None
        view_389 = torch.ops.aten.view.default(gather_97, [-1]);  gather_97 = None
        select_876 = torch.ops.aten.select.int(select_scatter_96, 1, 97)
        copy_97 = torch.ops.aten.copy.default(select_876, argmin_97);  select_876 = argmin_97 = None
        select_scatter_97 = torch.ops.aten.select_scatter.default(select_scatter_96, copy_97, 1, 97);  select_scatter_96 = copy_97 = None
        sub_97 = torch.ops.aten.sub.Tensor(select_873, view_389);  select_873 = view_389 = None
        div_97 = torch.ops.aten.div.Tensor(sub_97, select_872);  sub_97 = select_872 = None
        select_878 = torch.ops.aten.select.int(arg1_1, 0, 97)
        slice_386 = torch.ops.aten.slice.Tensor(select_878, 0, 97, 9223372036854775807);  select_878 = None
        slice_387 = torch.ops.aten.slice.Tensor(slice_scatter_95, 1, 97, 9223372036854775807)
        expand_195 = torch.ops.aten.expand.default(slice_387, [2048, 31]);  slice_387 = None
        mul_291 = torch.ops.aten.mul.Tensor(expand_195, 1);  expand_195 = None
        view_390 = torch.ops.aten.view.default(div_97, [2048, 1]);  div_97 = None
        mul_292 = torch.ops.aten.mul.Tensor(view_390, slice_386);  view_390 = slice_386 = None
        mul_293 = torch.ops.aten.mul.Tensor(mul_292, -1.0);  mul_292 = None
        add_97 = torch.ops.aten.add.Tensor(mul_291, mul_293);  mul_291 = mul_293 = None
        slice_scatter_96 = torch.ops.aten.slice_scatter.default(slice_scatter_95, add_97, 1, 97, 9223372036854775807);  slice_scatter_95 = add_97 = None
        select_880 = torch.ops.aten.select.int(arg1_1, 0, 98)
        select_881 = torch.ops.aten.select.int(select_880, 0, 98);  select_880 = None
        select_882 = torch.ops.aten.select.int(slice_scatter_96, 1, 98)
        unsqueeze_196 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_98 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_883 = torch.ops.aten.select.int(slice_scatter_96, 1, 98)
        view_392 = torch.ops.aten.view.default(select_883, [8, 128, 2]);  select_883 = None
        baddbmm_98 = torch.ops.aten.baddbmm.default(unsqueeze_196, view_392, permute_98, alpha = -2.0);  unsqueeze_196 = view_392 = permute_98 = None
        argmin_98 = torch.ops.aten.argmin.default(baddbmm_98, -1);  baddbmm_98 = None
        unsqueeze_197 = torch.ops.aten.unsqueeze.default(argmin_98, -1)
        expand_196 = torch.ops.aten.expand.default(unsqueeze_197, [8, 128, 2]);  unsqueeze_197 = None
        gather_98 = torch.ops.aten.gather.default(arg3_1, 1, expand_196);  expand_196 = None
        view_393 = torch.ops.aten.view.default(gather_98, [-1]);  gather_98 = None
        select_885 = torch.ops.aten.select.int(select_scatter_97, 1, 98)
        copy_98 = torch.ops.aten.copy.default(select_885, argmin_98);  select_885 = argmin_98 = None
        select_scatter_98 = torch.ops.aten.select_scatter.default(select_scatter_97, copy_98, 1, 98);  select_scatter_97 = copy_98 = None
        sub_98 = torch.ops.aten.sub.Tensor(select_882, view_393);  select_882 = view_393 = None
        div_98 = torch.ops.aten.div.Tensor(sub_98, select_881);  sub_98 = select_881 = None
        select_887 = torch.ops.aten.select.int(arg1_1, 0, 98)
        slice_390 = torch.ops.aten.slice.Tensor(select_887, 0, 98, 9223372036854775807);  select_887 = None
        slice_391 = torch.ops.aten.slice.Tensor(slice_scatter_96, 1, 98, 9223372036854775807)
        expand_197 = torch.ops.aten.expand.default(slice_391, [2048, 30]);  slice_391 = None
        mul_294 = torch.ops.aten.mul.Tensor(expand_197, 1);  expand_197 = None
        view_394 = torch.ops.aten.view.default(div_98, [2048, 1]);  div_98 = None
        mul_295 = torch.ops.aten.mul.Tensor(view_394, slice_390);  view_394 = slice_390 = None
        mul_296 = torch.ops.aten.mul.Tensor(mul_295, -1.0);  mul_295 = None
        add_98 = torch.ops.aten.add.Tensor(mul_294, mul_296);  mul_294 = mul_296 = None
        slice_scatter_97 = torch.ops.aten.slice_scatter.default(slice_scatter_96, add_98, 1, 98, 9223372036854775807);  slice_scatter_96 = add_98 = None
        select_889 = torch.ops.aten.select.int(arg1_1, 0, 99)
        select_890 = torch.ops.aten.select.int(select_889, 0, 99);  select_889 = None
        select_891 = torch.ops.aten.select.int(slice_scatter_97, 1, 99)
        unsqueeze_198 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_99 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_892 = torch.ops.aten.select.int(slice_scatter_97, 1, 99)
        view_396 = torch.ops.aten.view.default(select_892, [8, 128, 2]);  select_892 = None
        baddbmm_99 = torch.ops.aten.baddbmm.default(unsqueeze_198, view_396, permute_99, alpha = -2.0);  unsqueeze_198 = view_396 = permute_99 = None
        argmin_99 = torch.ops.aten.argmin.default(baddbmm_99, -1);  baddbmm_99 = None
        unsqueeze_199 = torch.ops.aten.unsqueeze.default(argmin_99, -1)
        expand_198 = torch.ops.aten.expand.default(unsqueeze_199, [8, 128, 2]);  unsqueeze_199 = None
        gather_99 = torch.ops.aten.gather.default(arg3_1, 1, expand_198);  expand_198 = None
        view_397 = torch.ops.aten.view.default(gather_99, [-1]);  gather_99 = None
        select_894 = torch.ops.aten.select.int(select_scatter_98, 1, 99)
        copy_99 = torch.ops.aten.copy.default(select_894, argmin_99);  select_894 = argmin_99 = None
        select_scatter_99 = torch.ops.aten.select_scatter.default(select_scatter_98, copy_99, 1, 99);  select_scatter_98 = copy_99 = None
        sub_99 = torch.ops.aten.sub.Tensor(select_891, view_397);  select_891 = view_397 = None
        div_99 = torch.ops.aten.div.Tensor(sub_99, select_890);  sub_99 = select_890 = None
        select_896 = torch.ops.aten.select.int(arg1_1, 0, 99)
        slice_394 = torch.ops.aten.slice.Tensor(select_896, 0, 99, 9223372036854775807);  select_896 = None
        slice_395 = torch.ops.aten.slice.Tensor(slice_scatter_97, 1, 99, 9223372036854775807)
        expand_199 = torch.ops.aten.expand.default(slice_395, [2048, 29]);  slice_395 = None
        mul_297 = torch.ops.aten.mul.Tensor(expand_199, 1);  expand_199 = None
        view_398 = torch.ops.aten.view.default(div_99, [2048, 1]);  div_99 = None
        mul_298 = torch.ops.aten.mul.Tensor(view_398, slice_394);  view_398 = slice_394 = None
        mul_299 = torch.ops.aten.mul.Tensor(mul_298, -1.0);  mul_298 = None
        add_99 = torch.ops.aten.add.Tensor(mul_297, mul_299);  mul_297 = mul_299 = None
        slice_scatter_98 = torch.ops.aten.slice_scatter.default(slice_scatter_97, add_99, 1, 99, 9223372036854775807);  slice_scatter_97 = add_99 = None
        select_898 = torch.ops.aten.select.int(arg1_1, 0, 100)
        select_899 = torch.ops.aten.select.int(select_898, 0, 100);  select_898 = None
        select_900 = torch.ops.aten.select.int(slice_scatter_98, 1, 100)
        unsqueeze_200 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_100 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_901 = torch.ops.aten.select.int(slice_scatter_98, 1, 100)
        view_400 = torch.ops.aten.view.default(select_901, [8, 128, 2]);  select_901 = None
        baddbmm_100 = torch.ops.aten.baddbmm.default(unsqueeze_200, view_400, permute_100, alpha = -2.0);  unsqueeze_200 = view_400 = permute_100 = None
        argmin_100 = torch.ops.aten.argmin.default(baddbmm_100, -1);  baddbmm_100 = None
        unsqueeze_201 = torch.ops.aten.unsqueeze.default(argmin_100, -1)
        expand_200 = torch.ops.aten.expand.default(unsqueeze_201, [8, 128, 2]);  unsqueeze_201 = None
        gather_100 = torch.ops.aten.gather.default(arg3_1, 1, expand_200);  expand_200 = None
        view_401 = torch.ops.aten.view.default(gather_100, [-1]);  gather_100 = None
        select_903 = torch.ops.aten.select.int(select_scatter_99, 1, 100)
        copy_100 = torch.ops.aten.copy.default(select_903, argmin_100);  select_903 = argmin_100 = None
        select_scatter_100 = torch.ops.aten.select_scatter.default(select_scatter_99, copy_100, 1, 100);  select_scatter_99 = copy_100 = None
        sub_100 = torch.ops.aten.sub.Tensor(select_900, view_401);  select_900 = view_401 = None
        div_100 = torch.ops.aten.div.Tensor(sub_100, select_899);  sub_100 = select_899 = None
        select_905 = torch.ops.aten.select.int(arg1_1, 0, 100)
        slice_398 = torch.ops.aten.slice.Tensor(select_905, 0, 100, 9223372036854775807);  select_905 = None
        slice_399 = torch.ops.aten.slice.Tensor(slice_scatter_98, 1, 100, 9223372036854775807)
        expand_201 = torch.ops.aten.expand.default(slice_399, [2048, 28]);  slice_399 = None
        mul_300 = torch.ops.aten.mul.Tensor(expand_201, 1);  expand_201 = None
        view_402 = torch.ops.aten.view.default(div_100, [2048, 1]);  div_100 = None
        mul_301 = torch.ops.aten.mul.Tensor(view_402, slice_398);  view_402 = slice_398 = None
        mul_302 = torch.ops.aten.mul.Tensor(mul_301, -1.0);  mul_301 = None
        add_100 = torch.ops.aten.add.Tensor(mul_300, mul_302);  mul_300 = mul_302 = None
        slice_scatter_99 = torch.ops.aten.slice_scatter.default(slice_scatter_98, add_100, 1, 100, 9223372036854775807);  slice_scatter_98 = add_100 = None
        select_907 = torch.ops.aten.select.int(arg1_1, 0, 101)
        select_908 = torch.ops.aten.select.int(select_907, 0, 101);  select_907 = None
        select_909 = torch.ops.aten.select.int(slice_scatter_99, 1, 101)
        unsqueeze_202 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_101 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_910 = torch.ops.aten.select.int(slice_scatter_99, 1, 101)
        view_404 = torch.ops.aten.view.default(select_910, [8, 128, 2]);  select_910 = None
        baddbmm_101 = torch.ops.aten.baddbmm.default(unsqueeze_202, view_404, permute_101, alpha = -2.0);  unsqueeze_202 = view_404 = permute_101 = None
        argmin_101 = torch.ops.aten.argmin.default(baddbmm_101, -1);  baddbmm_101 = None
        unsqueeze_203 = torch.ops.aten.unsqueeze.default(argmin_101, -1)
        expand_202 = torch.ops.aten.expand.default(unsqueeze_203, [8, 128, 2]);  unsqueeze_203 = None
        gather_101 = torch.ops.aten.gather.default(arg3_1, 1, expand_202);  expand_202 = None
        view_405 = torch.ops.aten.view.default(gather_101, [-1]);  gather_101 = None
        select_912 = torch.ops.aten.select.int(select_scatter_100, 1, 101)
        copy_101 = torch.ops.aten.copy.default(select_912, argmin_101);  select_912 = argmin_101 = None
        select_scatter_101 = torch.ops.aten.select_scatter.default(select_scatter_100, copy_101, 1, 101);  select_scatter_100 = copy_101 = None
        sub_101 = torch.ops.aten.sub.Tensor(select_909, view_405);  select_909 = view_405 = None
        div_101 = torch.ops.aten.div.Tensor(sub_101, select_908);  sub_101 = select_908 = None
        select_914 = torch.ops.aten.select.int(arg1_1, 0, 101)
        slice_402 = torch.ops.aten.slice.Tensor(select_914, 0, 101, 9223372036854775807);  select_914 = None
        slice_403 = torch.ops.aten.slice.Tensor(slice_scatter_99, 1, 101, 9223372036854775807)
        expand_203 = torch.ops.aten.expand.default(slice_403, [2048, 27]);  slice_403 = None
        mul_303 = torch.ops.aten.mul.Tensor(expand_203, 1);  expand_203 = None
        view_406 = torch.ops.aten.view.default(div_101, [2048, 1]);  div_101 = None
        mul_304 = torch.ops.aten.mul.Tensor(view_406, slice_402);  view_406 = slice_402 = None
        mul_305 = torch.ops.aten.mul.Tensor(mul_304, -1.0);  mul_304 = None
        add_101 = torch.ops.aten.add.Tensor(mul_303, mul_305);  mul_303 = mul_305 = None
        slice_scatter_100 = torch.ops.aten.slice_scatter.default(slice_scatter_99, add_101, 1, 101, 9223372036854775807);  slice_scatter_99 = add_101 = None
        select_916 = torch.ops.aten.select.int(arg1_1, 0, 102)
        select_917 = torch.ops.aten.select.int(select_916, 0, 102);  select_916 = None
        select_918 = torch.ops.aten.select.int(slice_scatter_100, 1, 102)
        unsqueeze_204 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_102 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_919 = torch.ops.aten.select.int(slice_scatter_100, 1, 102)
        view_408 = torch.ops.aten.view.default(select_919, [8, 128, 2]);  select_919 = None
        baddbmm_102 = torch.ops.aten.baddbmm.default(unsqueeze_204, view_408, permute_102, alpha = -2.0);  unsqueeze_204 = view_408 = permute_102 = None
        argmin_102 = torch.ops.aten.argmin.default(baddbmm_102, -1);  baddbmm_102 = None
        unsqueeze_205 = torch.ops.aten.unsqueeze.default(argmin_102, -1)
        expand_204 = torch.ops.aten.expand.default(unsqueeze_205, [8, 128, 2]);  unsqueeze_205 = None
        gather_102 = torch.ops.aten.gather.default(arg3_1, 1, expand_204);  expand_204 = None
        view_409 = torch.ops.aten.view.default(gather_102, [-1]);  gather_102 = None
        select_921 = torch.ops.aten.select.int(select_scatter_101, 1, 102)
        copy_102 = torch.ops.aten.copy.default(select_921, argmin_102);  select_921 = argmin_102 = None
        select_scatter_102 = torch.ops.aten.select_scatter.default(select_scatter_101, copy_102, 1, 102);  select_scatter_101 = copy_102 = None
        sub_102 = torch.ops.aten.sub.Tensor(select_918, view_409);  select_918 = view_409 = None
        div_102 = torch.ops.aten.div.Tensor(sub_102, select_917);  sub_102 = select_917 = None
        select_923 = torch.ops.aten.select.int(arg1_1, 0, 102)
        slice_406 = torch.ops.aten.slice.Tensor(select_923, 0, 102, 9223372036854775807);  select_923 = None
        slice_407 = torch.ops.aten.slice.Tensor(slice_scatter_100, 1, 102, 9223372036854775807)
        expand_205 = torch.ops.aten.expand.default(slice_407, [2048, 26]);  slice_407 = None
        mul_306 = torch.ops.aten.mul.Tensor(expand_205, 1);  expand_205 = None
        view_410 = torch.ops.aten.view.default(div_102, [2048, 1]);  div_102 = None
        mul_307 = torch.ops.aten.mul.Tensor(view_410, slice_406);  view_410 = slice_406 = None
        mul_308 = torch.ops.aten.mul.Tensor(mul_307, -1.0);  mul_307 = None
        add_102 = torch.ops.aten.add.Tensor(mul_306, mul_308);  mul_306 = mul_308 = None
        slice_scatter_101 = torch.ops.aten.slice_scatter.default(slice_scatter_100, add_102, 1, 102, 9223372036854775807);  slice_scatter_100 = add_102 = None
        select_925 = torch.ops.aten.select.int(arg1_1, 0, 103)
        select_926 = torch.ops.aten.select.int(select_925, 0, 103);  select_925 = None
        select_927 = torch.ops.aten.select.int(slice_scatter_101, 1, 103)
        unsqueeze_206 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_103 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_928 = torch.ops.aten.select.int(slice_scatter_101, 1, 103)
        view_412 = torch.ops.aten.view.default(select_928, [8, 128, 2]);  select_928 = None
        baddbmm_103 = torch.ops.aten.baddbmm.default(unsqueeze_206, view_412, permute_103, alpha = -2.0);  unsqueeze_206 = view_412 = permute_103 = None
        argmin_103 = torch.ops.aten.argmin.default(baddbmm_103, -1);  baddbmm_103 = None
        unsqueeze_207 = torch.ops.aten.unsqueeze.default(argmin_103, -1)
        expand_206 = torch.ops.aten.expand.default(unsqueeze_207, [8, 128, 2]);  unsqueeze_207 = None
        gather_103 = torch.ops.aten.gather.default(arg3_1, 1, expand_206);  expand_206 = None
        view_413 = torch.ops.aten.view.default(gather_103, [-1]);  gather_103 = None
        select_930 = torch.ops.aten.select.int(select_scatter_102, 1, 103)
        copy_103 = torch.ops.aten.copy.default(select_930, argmin_103);  select_930 = argmin_103 = None
        select_scatter_103 = torch.ops.aten.select_scatter.default(select_scatter_102, copy_103, 1, 103);  select_scatter_102 = copy_103 = None
        sub_103 = torch.ops.aten.sub.Tensor(select_927, view_413);  select_927 = view_413 = None
        div_103 = torch.ops.aten.div.Tensor(sub_103, select_926);  sub_103 = select_926 = None
        select_932 = torch.ops.aten.select.int(arg1_1, 0, 103)
        slice_410 = torch.ops.aten.slice.Tensor(select_932, 0, 103, 9223372036854775807);  select_932 = None
        slice_411 = torch.ops.aten.slice.Tensor(slice_scatter_101, 1, 103, 9223372036854775807)
        expand_207 = torch.ops.aten.expand.default(slice_411, [2048, 25]);  slice_411 = None
        mul_309 = torch.ops.aten.mul.Tensor(expand_207, 1);  expand_207 = None
        view_414 = torch.ops.aten.view.default(div_103, [2048, 1]);  div_103 = None
        mul_310 = torch.ops.aten.mul.Tensor(view_414, slice_410);  view_414 = slice_410 = None
        mul_311 = torch.ops.aten.mul.Tensor(mul_310, -1.0);  mul_310 = None
        add_103 = torch.ops.aten.add.Tensor(mul_309, mul_311);  mul_309 = mul_311 = None
        slice_scatter_102 = torch.ops.aten.slice_scatter.default(slice_scatter_101, add_103, 1, 103, 9223372036854775807);  slice_scatter_101 = add_103 = None
        select_934 = torch.ops.aten.select.int(arg1_1, 0, 104)
        select_935 = torch.ops.aten.select.int(select_934, 0, 104);  select_934 = None
        select_936 = torch.ops.aten.select.int(slice_scatter_102, 1, 104)
        unsqueeze_208 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_104 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_937 = torch.ops.aten.select.int(slice_scatter_102, 1, 104)
        view_416 = torch.ops.aten.view.default(select_937, [8, 128, 2]);  select_937 = None
        baddbmm_104 = torch.ops.aten.baddbmm.default(unsqueeze_208, view_416, permute_104, alpha = -2.0);  unsqueeze_208 = view_416 = permute_104 = None
        argmin_104 = torch.ops.aten.argmin.default(baddbmm_104, -1);  baddbmm_104 = None
        unsqueeze_209 = torch.ops.aten.unsqueeze.default(argmin_104, -1)
        expand_208 = torch.ops.aten.expand.default(unsqueeze_209, [8, 128, 2]);  unsqueeze_209 = None
        gather_104 = torch.ops.aten.gather.default(arg3_1, 1, expand_208);  expand_208 = None
        view_417 = torch.ops.aten.view.default(gather_104, [-1]);  gather_104 = None
        select_939 = torch.ops.aten.select.int(select_scatter_103, 1, 104)
        copy_104 = torch.ops.aten.copy.default(select_939, argmin_104);  select_939 = argmin_104 = None
        select_scatter_104 = torch.ops.aten.select_scatter.default(select_scatter_103, copy_104, 1, 104);  select_scatter_103 = copy_104 = None
        sub_104 = torch.ops.aten.sub.Tensor(select_936, view_417);  select_936 = view_417 = None
        div_104 = torch.ops.aten.div.Tensor(sub_104, select_935);  sub_104 = select_935 = None
        select_941 = torch.ops.aten.select.int(arg1_1, 0, 104)
        slice_414 = torch.ops.aten.slice.Tensor(select_941, 0, 104, 9223372036854775807);  select_941 = None
        slice_415 = torch.ops.aten.slice.Tensor(slice_scatter_102, 1, 104, 9223372036854775807)
        expand_209 = torch.ops.aten.expand.default(slice_415, [2048, 24]);  slice_415 = None
        mul_312 = torch.ops.aten.mul.Tensor(expand_209, 1);  expand_209 = None
        view_418 = torch.ops.aten.view.default(div_104, [2048, 1]);  div_104 = None
        mul_313 = torch.ops.aten.mul.Tensor(view_418, slice_414);  view_418 = slice_414 = None
        mul_314 = torch.ops.aten.mul.Tensor(mul_313, -1.0);  mul_313 = None
        add_104 = torch.ops.aten.add.Tensor(mul_312, mul_314);  mul_312 = mul_314 = None
        slice_scatter_103 = torch.ops.aten.slice_scatter.default(slice_scatter_102, add_104, 1, 104, 9223372036854775807);  slice_scatter_102 = add_104 = None
        select_943 = torch.ops.aten.select.int(arg1_1, 0, 105)
        select_944 = torch.ops.aten.select.int(select_943, 0, 105);  select_943 = None
        select_945 = torch.ops.aten.select.int(slice_scatter_103, 1, 105)
        unsqueeze_210 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_105 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_946 = torch.ops.aten.select.int(slice_scatter_103, 1, 105)
        view_420 = torch.ops.aten.view.default(select_946, [8, 128, 2]);  select_946 = None
        baddbmm_105 = torch.ops.aten.baddbmm.default(unsqueeze_210, view_420, permute_105, alpha = -2.0);  unsqueeze_210 = view_420 = permute_105 = None
        argmin_105 = torch.ops.aten.argmin.default(baddbmm_105, -1);  baddbmm_105 = None
        unsqueeze_211 = torch.ops.aten.unsqueeze.default(argmin_105, -1)
        expand_210 = torch.ops.aten.expand.default(unsqueeze_211, [8, 128, 2]);  unsqueeze_211 = None
        gather_105 = torch.ops.aten.gather.default(arg3_1, 1, expand_210);  expand_210 = None
        view_421 = torch.ops.aten.view.default(gather_105, [-1]);  gather_105 = None
        select_948 = torch.ops.aten.select.int(select_scatter_104, 1, 105)
        copy_105 = torch.ops.aten.copy.default(select_948, argmin_105);  select_948 = argmin_105 = None
        select_scatter_105 = torch.ops.aten.select_scatter.default(select_scatter_104, copy_105, 1, 105);  select_scatter_104 = copy_105 = None
        sub_105 = torch.ops.aten.sub.Tensor(select_945, view_421);  select_945 = view_421 = None
        div_105 = torch.ops.aten.div.Tensor(sub_105, select_944);  sub_105 = select_944 = None
        select_950 = torch.ops.aten.select.int(arg1_1, 0, 105)
        slice_418 = torch.ops.aten.slice.Tensor(select_950, 0, 105, 9223372036854775807);  select_950 = None
        slice_419 = torch.ops.aten.slice.Tensor(slice_scatter_103, 1, 105, 9223372036854775807)
        expand_211 = torch.ops.aten.expand.default(slice_419, [2048, 23]);  slice_419 = None
        mul_315 = torch.ops.aten.mul.Tensor(expand_211, 1);  expand_211 = None
        view_422 = torch.ops.aten.view.default(div_105, [2048, 1]);  div_105 = None
        mul_316 = torch.ops.aten.mul.Tensor(view_422, slice_418);  view_422 = slice_418 = None
        mul_317 = torch.ops.aten.mul.Tensor(mul_316, -1.0);  mul_316 = None
        add_105 = torch.ops.aten.add.Tensor(mul_315, mul_317);  mul_315 = mul_317 = None
        slice_scatter_104 = torch.ops.aten.slice_scatter.default(slice_scatter_103, add_105, 1, 105, 9223372036854775807);  slice_scatter_103 = add_105 = None
        select_952 = torch.ops.aten.select.int(arg1_1, 0, 106)
        select_953 = torch.ops.aten.select.int(select_952, 0, 106);  select_952 = None
        select_954 = torch.ops.aten.select.int(slice_scatter_104, 1, 106)
        unsqueeze_212 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_106 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_955 = torch.ops.aten.select.int(slice_scatter_104, 1, 106)
        view_424 = torch.ops.aten.view.default(select_955, [8, 128, 2]);  select_955 = None
        baddbmm_106 = torch.ops.aten.baddbmm.default(unsqueeze_212, view_424, permute_106, alpha = -2.0);  unsqueeze_212 = view_424 = permute_106 = None
        argmin_106 = torch.ops.aten.argmin.default(baddbmm_106, -1);  baddbmm_106 = None
        unsqueeze_213 = torch.ops.aten.unsqueeze.default(argmin_106, -1)
        expand_212 = torch.ops.aten.expand.default(unsqueeze_213, [8, 128, 2]);  unsqueeze_213 = None
        gather_106 = torch.ops.aten.gather.default(arg3_1, 1, expand_212);  expand_212 = None
        view_425 = torch.ops.aten.view.default(gather_106, [-1]);  gather_106 = None
        select_957 = torch.ops.aten.select.int(select_scatter_105, 1, 106)
        copy_106 = torch.ops.aten.copy.default(select_957, argmin_106);  select_957 = argmin_106 = None
        select_scatter_106 = torch.ops.aten.select_scatter.default(select_scatter_105, copy_106, 1, 106);  select_scatter_105 = copy_106 = None
        sub_106 = torch.ops.aten.sub.Tensor(select_954, view_425);  select_954 = view_425 = None
        div_106 = torch.ops.aten.div.Tensor(sub_106, select_953);  sub_106 = select_953 = None
        select_959 = torch.ops.aten.select.int(arg1_1, 0, 106)
        slice_422 = torch.ops.aten.slice.Tensor(select_959, 0, 106, 9223372036854775807);  select_959 = None
        slice_423 = torch.ops.aten.slice.Tensor(slice_scatter_104, 1, 106, 9223372036854775807)
        expand_213 = torch.ops.aten.expand.default(slice_423, [2048, 22]);  slice_423 = None
        mul_318 = torch.ops.aten.mul.Tensor(expand_213, 1);  expand_213 = None
        view_426 = torch.ops.aten.view.default(div_106, [2048, 1]);  div_106 = None
        mul_319 = torch.ops.aten.mul.Tensor(view_426, slice_422);  view_426 = slice_422 = None
        mul_320 = torch.ops.aten.mul.Tensor(mul_319, -1.0);  mul_319 = None
        add_106 = torch.ops.aten.add.Tensor(mul_318, mul_320);  mul_318 = mul_320 = None
        slice_scatter_105 = torch.ops.aten.slice_scatter.default(slice_scatter_104, add_106, 1, 106, 9223372036854775807);  slice_scatter_104 = add_106 = None
        select_961 = torch.ops.aten.select.int(arg1_1, 0, 107)
        select_962 = torch.ops.aten.select.int(select_961, 0, 107);  select_961 = None
        select_963 = torch.ops.aten.select.int(slice_scatter_105, 1, 107)
        unsqueeze_214 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_107 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_964 = torch.ops.aten.select.int(slice_scatter_105, 1, 107)
        view_428 = torch.ops.aten.view.default(select_964, [8, 128, 2]);  select_964 = None
        baddbmm_107 = torch.ops.aten.baddbmm.default(unsqueeze_214, view_428, permute_107, alpha = -2.0);  unsqueeze_214 = view_428 = permute_107 = None
        argmin_107 = torch.ops.aten.argmin.default(baddbmm_107, -1);  baddbmm_107 = None
        unsqueeze_215 = torch.ops.aten.unsqueeze.default(argmin_107, -1)
        expand_214 = torch.ops.aten.expand.default(unsqueeze_215, [8, 128, 2]);  unsqueeze_215 = None
        gather_107 = torch.ops.aten.gather.default(arg3_1, 1, expand_214);  expand_214 = None
        view_429 = torch.ops.aten.view.default(gather_107, [-1]);  gather_107 = None
        select_966 = torch.ops.aten.select.int(select_scatter_106, 1, 107)
        copy_107 = torch.ops.aten.copy.default(select_966, argmin_107);  select_966 = argmin_107 = None
        select_scatter_107 = torch.ops.aten.select_scatter.default(select_scatter_106, copy_107, 1, 107);  select_scatter_106 = copy_107 = None
        sub_107 = torch.ops.aten.sub.Tensor(select_963, view_429);  select_963 = view_429 = None
        div_107 = torch.ops.aten.div.Tensor(sub_107, select_962);  sub_107 = select_962 = None
        select_968 = torch.ops.aten.select.int(arg1_1, 0, 107)
        slice_426 = torch.ops.aten.slice.Tensor(select_968, 0, 107, 9223372036854775807);  select_968 = None
        slice_427 = torch.ops.aten.slice.Tensor(slice_scatter_105, 1, 107, 9223372036854775807)
        expand_215 = torch.ops.aten.expand.default(slice_427, [2048, 21]);  slice_427 = None
        mul_321 = torch.ops.aten.mul.Tensor(expand_215, 1);  expand_215 = None
        view_430 = torch.ops.aten.view.default(div_107, [2048, 1]);  div_107 = None
        mul_322 = torch.ops.aten.mul.Tensor(view_430, slice_426);  view_430 = slice_426 = None
        mul_323 = torch.ops.aten.mul.Tensor(mul_322, -1.0);  mul_322 = None
        add_107 = torch.ops.aten.add.Tensor(mul_321, mul_323);  mul_321 = mul_323 = None
        slice_scatter_106 = torch.ops.aten.slice_scatter.default(slice_scatter_105, add_107, 1, 107, 9223372036854775807);  slice_scatter_105 = add_107 = None
        select_970 = torch.ops.aten.select.int(arg1_1, 0, 108)
        select_971 = torch.ops.aten.select.int(select_970, 0, 108);  select_970 = None
        select_972 = torch.ops.aten.select.int(slice_scatter_106, 1, 108)
        unsqueeze_216 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_108 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_973 = torch.ops.aten.select.int(slice_scatter_106, 1, 108)
        view_432 = torch.ops.aten.view.default(select_973, [8, 128, 2]);  select_973 = None
        baddbmm_108 = torch.ops.aten.baddbmm.default(unsqueeze_216, view_432, permute_108, alpha = -2.0);  unsqueeze_216 = view_432 = permute_108 = None
        argmin_108 = torch.ops.aten.argmin.default(baddbmm_108, -1);  baddbmm_108 = None
        unsqueeze_217 = torch.ops.aten.unsqueeze.default(argmin_108, -1)
        expand_216 = torch.ops.aten.expand.default(unsqueeze_217, [8, 128, 2]);  unsqueeze_217 = None
        gather_108 = torch.ops.aten.gather.default(arg3_1, 1, expand_216);  expand_216 = None
        view_433 = torch.ops.aten.view.default(gather_108, [-1]);  gather_108 = None
        select_975 = torch.ops.aten.select.int(select_scatter_107, 1, 108)
        copy_108 = torch.ops.aten.copy.default(select_975, argmin_108);  select_975 = argmin_108 = None
        select_scatter_108 = torch.ops.aten.select_scatter.default(select_scatter_107, copy_108, 1, 108);  select_scatter_107 = copy_108 = None
        sub_108 = torch.ops.aten.sub.Tensor(select_972, view_433);  select_972 = view_433 = None
        div_108 = torch.ops.aten.div.Tensor(sub_108, select_971);  sub_108 = select_971 = None
        select_977 = torch.ops.aten.select.int(arg1_1, 0, 108)
        slice_430 = torch.ops.aten.slice.Tensor(select_977, 0, 108, 9223372036854775807);  select_977 = None
        slice_431 = torch.ops.aten.slice.Tensor(slice_scatter_106, 1, 108, 9223372036854775807)
        expand_217 = torch.ops.aten.expand.default(slice_431, [2048, 20]);  slice_431 = None
        mul_324 = torch.ops.aten.mul.Tensor(expand_217, 1);  expand_217 = None
        view_434 = torch.ops.aten.view.default(div_108, [2048, 1]);  div_108 = None
        mul_325 = torch.ops.aten.mul.Tensor(view_434, slice_430);  view_434 = slice_430 = None
        mul_326 = torch.ops.aten.mul.Tensor(mul_325, -1.0);  mul_325 = None
        add_108 = torch.ops.aten.add.Tensor(mul_324, mul_326);  mul_324 = mul_326 = None
        slice_scatter_107 = torch.ops.aten.slice_scatter.default(slice_scatter_106, add_108, 1, 108, 9223372036854775807);  slice_scatter_106 = add_108 = None
        select_979 = torch.ops.aten.select.int(arg1_1, 0, 109)
        select_980 = torch.ops.aten.select.int(select_979, 0, 109);  select_979 = None
        select_981 = torch.ops.aten.select.int(slice_scatter_107, 1, 109)
        unsqueeze_218 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_109 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_982 = torch.ops.aten.select.int(slice_scatter_107, 1, 109)
        view_436 = torch.ops.aten.view.default(select_982, [8, 128, 2]);  select_982 = None
        baddbmm_109 = torch.ops.aten.baddbmm.default(unsqueeze_218, view_436, permute_109, alpha = -2.0);  unsqueeze_218 = view_436 = permute_109 = None
        argmin_109 = torch.ops.aten.argmin.default(baddbmm_109, -1);  baddbmm_109 = None
        unsqueeze_219 = torch.ops.aten.unsqueeze.default(argmin_109, -1)
        expand_218 = torch.ops.aten.expand.default(unsqueeze_219, [8, 128, 2]);  unsqueeze_219 = None
        gather_109 = torch.ops.aten.gather.default(arg3_1, 1, expand_218);  expand_218 = None
        view_437 = torch.ops.aten.view.default(gather_109, [-1]);  gather_109 = None
        select_984 = torch.ops.aten.select.int(select_scatter_108, 1, 109)
        copy_109 = torch.ops.aten.copy.default(select_984, argmin_109);  select_984 = argmin_109 = None
        select_scatter_109 = torch.ops.aten.select_scatter.default(select_scatter_108, copy_109, 1, 109);  select_scatter_108 = copy_109 = None
        sub_109 = torch.ops.aten.sub.Tensor(select_981, view_437);  select_981 = view_437 = None
        div_109 = torch.ops.aten.div.Tensor(sub_109, select_980);  sub_109 = select_980 = None
        select_986 = torch.ops.aten.select.int(arg1_1, 0, 109)
        slice_434 = torch.ops.aten.slice.Tensor(select_986, 0, 109, 9223372036854775807);  select_986 = None
        slice_435 = torch.ops.aten.slice.Tensor(slice_scatter_107, 1, 109, 9223372036854775807)
        expand_219 = torch.ops.aten.expand.default(slice_435, [2048, 19]);  slice_435 = None
        mul_327 = torch.ops.aten.mul.Tensor(expand_219, 1);  expand_219 = None
        view_438 = torch.ops.aten.view.default(div_109, [2048, 1]);  div_109 = None
        mul_328 = torch.ops.aten.mul.Tensor(view_438, slice_434);  view_438 = slice_434 = None
        mul_329 = torch.ops.aten.mul.Tensor(mul_328, -1.0);  mul_328 = None
        add_109 = torch.ops.aten.add.Tensor(mul_327, mul_329);  mul_327 = mul_329 = None
        slice_scatter_108 = torch.ops.aten.slice_scatter.default(slice_scatter_107, add_109, 1, 109, 9223372036854775807);  slice_scatter_107 = add_109 = None
        select_988 = torch.ops.aten.select.int(arg1_1, 0, 110)
        select_989 = torch.ops.aten.select.int(select_988, 0, 110);  select_988 = None
        select_990 = torch.ops.aten.select.int(slice_scatter_108, 1, 110)
        unsqueeze_220 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_110 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_991 = torch.ops.aten.select.int(slice_scatter_108, 1, 110)
        view_440 = torch.ops.aten.view.default(select_991, [8, 128, 2]);  select_991 = None
        baddbmm_110 = torch.ops.aten.baddbmm.default(unsqueeze_220, view_440, permute_110, alpha = -2.0);  unsqueeze_220 = view_440 = permute_110 = None
        argmin_110 = torch.ops.aten.argmin.default(baddbmm_110, -1);  baddbmm_110 = None
        unsqueeze_221 = torch.ops.aten.unsqueeze.default(argmin_110, -1)
        expand_220 = torch.ops.aten.expand.default(unsqueeze_221, [8, 128, 2]);  unsqueeze_221 = None
        gather_110 = torch.ops.aten.gather.default(arg3_1, 1, expand_220);  expand_220 = None
        view_441 = torch.ops.aten.view.default(gather_110, [-1]);  gather_110 = None
        select_993 = torch.ops.aten.select.int(select_scatter_109, 1, 110)
        copy_110 = torch.ops.aten.copy.default(select_993, argmin_110);  select_993 = argmin_110 = None
        select_scatter_110 = torch.ops.aten.select_scatter.default(select_scatter_109, copy_110, 1, 110);  select_scatter_109 = copy_110 = None
        sub_110 = torch.ops.aten.sub.Tensor(select_990, view_441);  select_990 = view_441 = None
        div_110 = torch.ops.aten.div.Tensor(sub_110, select_989);  sub_110 = select_989 = None
        select_995 = torch.ops.aten.select.int(arg1_1, 0, 110)
        slice_438 = torch.ops.aten.slice.Tensor(select_995, 0, 110, 9223372036854775807);  select_995 = None
        slice_439 = torch.ops.aten.slice.Tensor(slice_scatter_108, 1, 110, 9223372036854775807)
        expand_221 = torch.ops.aten.expand.default(slice_439, [2048, 18]);  slice_439 = None
        mul_330 = torch.ops.aten.mul.Tensor(expand_221, 1);  expand_221 = None
        view_442 = torch.ops.aten.view.default(div_110, [2048, 1]);  div_110 = None
        mul_331 = torch.ops.aten.mul.Tensor(view_442, slice_438);  view_442 = slice_438 = None
        mul_332 = torch.ops.aten.mul.Tensor(mul_331, -1.0);  mul_331 = None
        add_110 = torch.ops.aten.add.Tensor(mul_330, mul_332);  mul_330 = mul_332 = None
        slice_scatter_109 = torch.ops.aten.slice_scatter.default(slice_scatter_108, add_110, 1, 110, 9223372036854775807);  slice_scatter_108 = add_110 = None
        select_997 = torch.ops.aten.select.int(arg1_1, 0, 111)
        select_998 = torch.ops.aten.select.int(select_997, 0, 111);  select_997 = None
        select_999 = torch.ops.aten.select.int(slice_scatter_109, 1, 111)
        unsqueeze_222 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_111 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_1000 = torch.ops.aten.select.int(slice_scatter_109, 1, 111)
        view_444 = torch.ops.aten.view.default(select_1000, [8, 128, 2]);  select_1000 = None
        baddbmm_111 = torch.ops.aten.baddbmm.default(unsqueeze_222, view_444, permute_111, alpha = -2.0);  unsqueeze_222 = view_444 = permute_111 = None
        argmin_111 = torch.ops.aten.argmin.default(baddbmm_111, -1);  baddbmm_111 = None
        unsqueeze_223 = torch.ops.aten.unsqueeze.default(argmin_111, -1)
        expand_222 = torch.ops.aten.expand.default(unsqueeze_223, [8, 128, 2]);  unsqueeze_223 = None
        gather_111 = torch.ops.aten.gather.default(arg3_1, 1, expand_222);  expand_222 = None
        view_445 = torch.ops.aten.view.default(gather_111, [-1]);  gather_111 = None
        select_1002 = torch.ops.aten.select.int(select_scatter_110, 1, 111)
        copy_111 = torch.ops.aten.copy.default(select_1002, argmin_111);  select_1002 = argmin_111 = None
        select_scatter_111 = torch.ops.aten.select_scatter.default(select_scatter_110, copy_111, 1, 111);  select_scatter_110 = copy_111 = None
        sub_111 = torch.ops.aten.sub.Tensor(select_999, view_445);  select_999 = view_445 = None
        div_111 = torch.ops.aten.div.Tensor(sub_111, select_998);  sub_111 = select_998 = None
        select_1004 = torch.ops.aten.select.int(arg1_1, 0, 111)
        slice_442 = torch.ops.aten.slice.Tensor(select_1004, 0, 111, 9223372036854775807);  select_1004 = None
        slice_443 = torch.ops.aten.slice.Tensor(slice_scatter_109, 1, 111, 9223372036854775807)
        expand_223 = torch.ops.aten.expand.default(slice_443, [2048, 17]);  slice_443 = None
        mul_333 = torch.ops.aten.mul.Tensor(expand_223, 1);  expand_223 = None
        view_446 = torch.ops.aten.view.default(div_111, [2048, 1]);  div_111 = None
        mul_334 = torch.ops.aten.mul.Tensor(view_446, slice_442);  view_446 = slice_442 = None
        mul_335 = torch.ops.aten.mul.Tensor(mul_334, -1.0);  mul_334 = None
        add_111 = torch.ops.aten.add.Tensor(mul_333, mul_335);  mul_333 = mul_335 = None
        slice_scatter_110 = torch.ops.aten.slice_scatter.default(slice_scatter_109, add_111, 1, 111, 9223372036854775807);  slice_scatter_109 = add_111 = None
        select_1006 = torch.ops.aten.select.int(arg1_1, 0, 112)
        select_1007 = torch.ops.aten.select.int(select_1006, 0, 112);  select_1006 = None
        select_1008 = torch.ops.aten.select.int(slice_scatter_110, 1, 112)
        unsqueeze_224 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_112 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_1009 = torch.ops.aten.select.int(slice_scatter_110, 1, 112)
        view_448 = torch.ops.aten.view.default(select_1009, [8, 128, 2]);  select_1009 = None
        baddbmm_112 = torch.ops.aten.baddbmm.default(unsqueeze_224, view_448, permute_112, alpha = -2.0);  unsqueeze_224 = view_448 = permute_112 = None
        argmin_112 = torch.ops.aten.argmin.default(baddbmm_112, -1);  baddbmm_112 = None
        unsqueeze_225 = torch.ops.aten.unsqueeze.default(argmin_112, -1)
        expand_224 = torch.ops.aten.expand.default(unsqueeze_225, [8, 128, 2]);  unsqueeze_225 = None
        gather_112 = torch.ops.aten.gather.default(arg3_1, 1, expand_224);  expand_224 = None
        view_449 = torch.ops.aten.view.default(gather_112, [-1]);  gather_112 = None
        select_1011 = torch.ops.aten.select.int(select_scatter_111, 1, 112)
        copy_112 = torch.ops.aten.copy.default(select_1011, argmin_112);  select_1011 = argmin_112 = None
        select_scatter_112 = torch.ops.aten.select_scatter.default(select_scatter_111, copy_112, 1, 112);  select_scatter_111 = copy_112 = None
        sub_112 = torch.ops.aten.sub.Tensor(select_1008, view_449);  select_1008 = view_449 = None
        div_112 = torch.ops.aten.div.Tensor(sub_112, select_1007);  sub_112 = select_1007 = None
        select_1013 = torch.ops.aten.select.int(arg1_1, 0, 112)
        slice_446 = torch.ops.aten.slice.Tensor(select_1013, 0, 112, 9223372036854775807);  select_1013 = None
        slice_447 = torch.ops.aten.slice.Tensor(slice_scatter_110, 1, 112, 9223372036854775807)
        expand_225 = torch.ops.aten.expand.default(slice_447, [2048, 16]);  slice_447 = None
        mul_336 = torch.ops.aten.mul.Tensor(expand_225, 1);  expand_225 = None
        view_450 = torch.ops.aten.view.default(div_112, [2048, 1]);  div_112 = None
        mul_337 = torch.ops.aten.mul.Tensor(view_450, slice_446);  view_450 = slice_446 = None
        mul_338 = torch.ops.aten.mul.Tensor(mul_337, -1.0);  mul_337 = None
        add_112 = torch.ops.aten.add.Tensor(mul_336, mul_338);  mul_336 = mul_338 = None
        slice_scatter_111 = torch.ops.aten.slice_scatter.default(slice_scatter_110, add_112, 1, 112, 9223372036854775807);  slice_scatter_110 = add_112 = None
        select_1015 = torch.ops.aten.select.int(arg1_1, 0, 113)
        select_1016 = torch.ops.aten.select.int(select_1015, 0, 113);  select_1015 = None
        select_1017 = torch.ops.aten.select.int(slice_scatter_111, 1, 113)
        unsqueeze_226 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_113 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_1018 = torch.ops.aten.select.int(slice_scatter_111, 1, 113)
        view_452 = torch.ops.aten.view.default(select_1018, [8, 128, 2]);  select_1018 = None
        baddbmm_113 = torch.ops.aten.baddbmm.default(unsqueeze_226, view_452, permute_113, alpha = -2.0);  unsqueeze_226 = view_452 = permute_113 = None
        argmin_113 = torch.ops.aten.argmin.default(baddbmm_113, -1);  baddbmm_113 = None
        unsqueeze_227 = torch.ops.aten.unsqueeze.default(argmin_113, -1)
        expand_226 = torch.ops.aten.expand.default(unsqueeze_227, [8, 128, 2]);  unsqueeze_227 = None
        gather_113 = torch.ops.aten.gather.default(arg3_1, 1, expand_226);  expand_226 = None
        view_453 = torch.ops.aten.view.default(gather_113, [-1]);  gather_113 = None
        select_1020 = torch.ops.aten.select.int(select_scatter_112, 1, 113)
        copy_113 = torch.ops.aten.copy.default(select_1020, argmin_113);  select_1020 = argmin_113 = None
        select_scatter_113 = torch.ops.aten.select_scatter.default(select_scatter_112, copy_113, 1, 113);  select_scatter_112 = copy_113 = None
        sub_113 = torch.ops.aten.sub.Tensor(select_1017, view_453);  select_1017 = view_453 = None
        div_113 = torch.ops.aten.div.Tensor(sub_113, select_1016);  sub_113 = select_1016 = None
        select_1022 = torch.ops.aten.select.int(arg1_1, 0, 113)
        slice_450 = torch.ops.aten.slice.Tensor(select_1022, 0, 113, 9223372036854775807);  select_1022 = None
        slice_451 = torch.ops.aten.slice.Tensor(slice_scatter_111, 1, 113, 9223372036854775807)
        expand_227 = torch.ops.aten.expand.default(slice_451, [2048, 15]);  slice_451 = None
        mul_339 = torch.ops.aten.mul.Tensor(expand_227, 1);  expand_227 = None
        view_454 = torch.ops.aten.view.default(div_113, [2048, 1]);  div_113 = None
        mul_340 = torch.ops.aten.mul.Tensor(view_454, slice_450);  view_454 = slice_450 = None
        mul_341 = torch.ops.aten.mul.Tensor(mul_340, -1.0);  mul_340 = None
        add_113 = torch.ops.aten.add.Tensor(mul_339, mul_341);  mul_339 = mul_341 = None
        slice_scatter_112 = torch.ops.aten.slice_scatter.default(slice_scatter_111, add_113, 1, 113, 9223372036854775807);  slice_scatter_111 = add_113 = None
        select_1024 = torch.ops.aten.select.int(arg1_1, 0, 114)
        select_1025 = torch.ops.aten.select.int(select_1024, 0, 114);  select_1024 = None
        select_1026 = torch.ops.aten.select.int(slice_scatter_112, 1, 114)
        unsqueeze_228 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_114 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_1027 = torch.ops.aten.select.int(slice_scatter_112, 1, 114)
        view_456 = torch.ops.aten.view.default(select_1027, [8, 128, 2]);  select_1027 = None
        baddbmm_114 = torch.ops.aten.baddbmm.default(unsqueeze_228, view_456, permute_114, alpha = -2.0);  unsqueeze_228 = view_456 = permute_114 = None
        argmin_114 = torch.ops.aten.argmin.default(baddbmm_114, -1);  baddbmm_114 = None
        unsqueeze_229 = torch.ops.aten.unsqueeze.default(argmin_114, -1)
        expand_228 = torch.ops.aten.expand.default(unsqueeze_229, [8, 128, 2]);  unsqueeze_229 = None
        gather_114 = torch.ops.aten.gather.default(arg3_1, 1, expand_228);  expand_228 = None
        view_457 = torch.ops.aten.view.default(gather_114, [-1]);  gather_114 = None
        select_1029 = torch.ops.aten.select.int(select_scatter_113, 1, 114)
        copy_114 = torch.ops.aten.copy.default(select_1029, argmin_114);  select_1029 = argmin_114 = None
        select_scatter_114 = torch.ops.aten.select_scatter.default(select_scatter_113, copy_114, 1, 114);  select_scatter_113 = copy_114 = None
        sub_114 = torch.ops.aten.sub.Tensor(select_1026, view_457);  select_1026 = view_457 = None
        div_114 = torch.ops.aten.div.Tensor(sub_114, select_1025);  sub_114 = select_1025 = None
        select_1031 = torch.ops.aten.select.int(arg1_1, 0, 114)
        slice_454 = torch.ops.aten.slice.Tensor(select_1031, 0, 114, 9223372036854775807);  select_1031 = None
        slice_455 = torch.ops.aten.slice.Tensor(slice_scatter_112, 1, 114, 9223372036854775807)
        expand_229 = torch.ops.aten.expand.default(slice_455, [2048, 14]);  slice_455 = None
        mul_342 = torch.ops.aten.mul.Tensor(expand_229, 1);  expand_229 = None
        view_458 = torch.ops.aten.view.default(div_114, [2048, 1]);  div_114 = None
        mul_343 = torch.ops.aten.mul.Tensor(view_458, slice_454);  view_458 = slice_454 = None
        mul_344 = torch.ops.aten.mul.Tensor(mul_343, -1.0);  mul_343 = None
        add_114 = torch.ops.aten.add.Tensor(mul_342, mul_344);  mul_342 = mul_344 = None
        slice_scatter_113 = torch.ops.aten.slice_scatter.default(slice_scatter_112, add_114, 1, 114, 9223372036854775807);  slice_scatter_112 = add_114 = None
        select_1033 = torch.ops.aten.select.int(arg1_1, 0, 115)
        select_1034 = torch.ops.aten.select.int(select_1033, 0, 115);  select_1033 = None
        select_1035 = torch.ops.aten.select.int(slice_scatter_113, 1, 115)
        unsqueeze_230 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_115 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_1036 = torch.ops.aten.select.int(slice_scatter_113, 1, 115)
        view_460 = torch.ops.aten.view.default(select_1036, [8, 128, 2]);  select_1036 = None
        baddbmm_115 = torch.ops.aten.baddbmm.default(unsqueeze_230, view_460, permute_115, alpha = -2.0);  unsqueeze_230 = view_460 = permute_115 = None
        argmin_115 = torch.ops.aten.argmin.default(baddbmm_115, -1);  baddbmm_115 = None
        unsqueeze_231 = torch.ops.aten.unsqueeze.default(argmin_115, -1)
        expand_230 = torch.ops.aten.expand.default(unsqueeze_231, [8, 128, 2]);  unsqueeze_231 = None
        gather_115 = torch.ops.aten.gather.default(arg3_1, 1, expand_230);  expand_230 = None
        view_461 = torch.ops.aten.view.default(gather_115, [-1]);  gather_115 = None
        select_1038 = torch.ops.aten.select.int(select_scatter_114, 1, 115)
        copy_115 = torch.ops.aten.copy.default(select_1038, argmin_115);  select_1038 = argmin_115 = None
        select_scatter_115 = torch.ops.aten.select_scatter.default(select_scatter_114, copy_115, 1, 115);  select_scatter_114 = copy_115 = None
        sub_115 = torch.ops.aten.sub.Tensor(select_1035, view_461);  select_1035 = view_461 = None
        div_115 = torch.ops.aten.div.Tensor(sub_115, select_1034);  sub_115 = select_1034 = None
        select_1040 = torch.ops.aten.select.int(arg1_1, 0, 115)
        slice_458 = torch.ops.aten.slice.Tensor(select_1040, 0, 115, 9223372036854775807);  select_1040 = None
        slice_459 = torch.ops.aten.slice.Tensor(slice_scatter_113, 1, 115, 9223372036854775807)
        expand_231 = torch.ops.aten.expand.default(slice_459, [2048, 13]);  slice_459 = None
        mul_345 = torch.ops.aten.mul.Tensor(expand_231, 1);  expand_231 = None
        view_462 = torch.ops.aten.view.default(div_115, [2048, 1]);  div_115 = None
        mul_346 = torch.ops.aten.mul.Tensor(view_462, slice_458);  view_462 = slice_458 = None
        mul_347 = torch.ops.aten.mul.Tensor(mul_346, -1.0);  mul_346 = None
        add_115 = torch.ops.aten.add.Tensor(mul_345, mul_347);  mul_345 = mul_347 = None
        slice_scatter_114 = torch.ops.aten.slice_scatter.default(slice_scatter_113, add_115, 1, 115, 9223372036854775807);  slice_scatter_113 = add_115 = None
        select_1042 = torch.ops.aten.select.int(arg1_1, 0, 116)
        select_1043 = torch.ops.aten.select.int(select_1042, 0, 116);  select_1042 = None
        select_1044 = torch.ops.aten.select.int(slice_scatter_114, 1, 116)
        unsqueeze_232 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_116 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_1045 = torch.ops.aten.select.int(slice_scatter_114, 1, 116)
        view_464 = torch.ops.aten.view.default(select_1045, [8, 128, 2]);  select_1045 = None
        baddbmm_116 = torch.ops.aten.baddbmm.default(unsqueeze_232, view_464, permute_116, alpha = -2.0);  unsqueeze_232 = view_464 = permute_116 = None
        argmin_116 = torch.ops.aten.argmin.default(baddbmm_116, -1);  baddbmm_116 = None
        unsqueeze_233 = torch.ops.aten.unsqueeze.default(argmin_116, -1)
        expand_232 = torch.ops.aten.expand.default(unsqueeze_233, [8, 128, 2]);  unsqueeze_233 = None
        gather_116 = torch.ops.aten.gather.default(arg3_1, 1, expand_232);  expand_232 = None
        view_465 = torch.ops.aten.view.default(gather_116, [-1]);  gather_116 = None
        select_1047 = torch.ops.aten.select.int(select_scatter_115, 1, 116)
        copy_116 = torch.ops.aten.copy.default(select_1047, argmin_116);  select_1047 = argmin_116 = None
        select_scatter_116 = torch.ops.aten.select_scatter.default(select_scatter_115, copy_116, 1, 116);  select_scatter_115 = copy_116 = None
        sub_116 = torch.ops.aten.sub.Tensor(select_1044, view_465);  select_1044 = view_465 = None
        div_116 = torch.ops.aten.div.Tensor(sub_116, select_1043);  sub_116 = select_1043 = None
        select_1049 = torch.ops.aten.select.int(arg1_1, 0, 116)
        slice_462 = torch.ops.aten.slice.Tensor(select_1049, 0, 116, 9223372036854775807);  select_1049 = None
        slice_463 = torch.ops.aten.slice.Tensor(slice_scatter_114, 1, 116, 9223372036854775807)
        expand_233 = torch.ops.aten.expand.default(slice_463, [2048, 12]);  slice_463 = None
        mul_348 = torch.ops.aten.mul.Tensor(expand_233, 1);  expand_233 = None
        view_466 = torch.ops.aten.view.default(div_116, [2048, 1]);  div_116 = None
        mul_349 = torch.ops.aten.mul.Tensor(view_466, slice_462);  view_466 = slice_462 = None
        mul_350 = torch.ops.aten.mul.Tensor(mul_349, -1.0);  mul_349 = None
        add_116 = torch.ops.aten.add.Tensor(mul_348, mul_350);  mul_348 = mul_350 = None
        slice_scatter_115 = torch.ops.aten.slice_scatter.default(slice_scatter_114, add_116, 1, 116, 9223372036854775807);  slice_scatter_114 = add_116 = None
        select_1051 = torch.ops.aten.select.int(arg1_1, 0, 117)
        select_1052 = torch.ops.aten.select.int(select_1051, 0, 117);  select_1051 = None
        select_1053 = torch.ops.aten.select.int(slice_scatter_115, 1, 117)
        unsqueeze_234 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_117 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_1054 = torch.ops.aten.select.int(slice_scatter_115, 1, 117)
        view_468 = torch.ops.aten.view.default(select_1054, [8, 128, 2]);  select_1054 = None
        baddbmm_117 = torch.ops.aten.baddbmm.default(unsqueeze_234, view_468, permute_117, alpha = -2.0);  unsqueeze_234 = view_468 = permute_117 = None
        argmin_117 = torch.ops.aten.argmin.default(baddbmm_117, -1);  baddbmm_117 = None
        unsqueeze_235 = torch.ops.aten.unsqueeze.default(argmin_117, -1)
        expand_234 = torch.ops.aten.expand.default(unsqueeze_235, [8, 128, 2]);  unsqueeze_235 = None
        gather_117 = torch.ops.aten.gather.default(arg3_1, 1, expand_234);  expand_234 = None
        view_469 = torch.ops.aten.view.default(gather_117, [-1]);  gather_117 = None
        select_1056 = torch.ops.aten.select.int(select_scatter_116, 1, 117)
        copy_117 = torch.ops.aten.copy.default(select_1056, argmin_117);  select_1056 = argmin_117 = None
        select_scatter_117 = torch.ops.aten.select_scatter.default(select_scatter_116, copy_117, 1, 117);  select_scatter_116 = copy_117 = None
        sub_117 = torch.ops.aten.sub.Tensor(select_1053, view_469);  select_1053 = view_469 = None
        div_117 = torch.ops.aten.div.Tensor(sub_117, select_1052);  sub_117 = select_1052 = None
        select_1058 = torch.ops.aten.select.int(arg1_1, 0, 117)
        slice_466 = torch.ops.aten.slice.Tensor(select_1058, 0, 117, 9223372036854775807);  select_1058 = None
        slice_467 = torch.ops.aten.slice.Tensor(slice_scatter_115, 1, 117, 9223372036854775807)
        expand_235 = torch.ops.aten.expand.default(slice_467, [2048, 11]);  slice_467 = None
        mul_351 = torch.ops.aten.mul.Tensor(expand_235, 1);  expand_235 = None
        view_470 = torch.ops.aten.view.default(div_117, [2048, 1]);  div_117 = None
        mul_352 = torch.ops.aten.mul.Tensor(view_470, slice_466);  view_470 = slice_466 = None
        mul_353 = torch.ops.aten.mul.Tensor(mul_352, -1.0);  mul_352 = None
        add_117 = torch.ops.aten.add.Tensor(mul_351, mul_353);  mul_351 = mul_353 = None
        slice_scatter_116 = torch.ops.aten.slice_scatter.default(slice_scatter_115, add_117, 1, 117, 9223372036854775807);  slice_scatter_115 = add_117 = None
        select_1060 = torch.ops.aten.select.int(arg1_1, 0, 118)
        select_1061 = torch.ops.aten.select.int(select_1060, 0, 118);  select_1060 = None
        select_1062 = torch.ops.aten.select.int(slice_scatter_116, 1, 118)
        unsqueeze_236 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_118 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_1063 = torch.ops.aten.select.int(slice_scatter_116, 1, 118)
        view_472 = torch.ops.aten.view.default(select_1063, [8, 128, 2]);  select_1063 = None
        baddbmm_118 = torch.ops.aten.baddbmm.default(unsqueeze_236, view_472, permute_118, alpha = -2.0);  unsqueeze_236 = view_472 = permute_118 = None
        argmin_118 = torch.ops.aten.argmin.default(baddbmm_118, -1);  baddbmm_118 = None
        unsqueeze_237 = torch.ops.aten.unsqueeze.default(argmin_118, -1)
        expand_236 = torch.ops.aten.expand.default(unsqueeze_237, [8, 128, 2]);  unsqueeze_237 = None
        gather_118 = torch.ops.aten.gather.default(arg3_1, 1, expand_236);  expand_236 = None
        view_473 = torch.ops.aten.view.default(gather_118, [-1]);  gather_118 = None
        select_1065 = torch.ops.aten.select.int(select_scatter_117, 1, 118)
        copy_118 = torch.ops.aten.copy.default(select_1065, argmin_118);  select_1065 = argmin_118 = None
        select_scatter_118 = torch.ops.aten.select_scatter.default(select_scatter_117, copy_118, 1, 118);  select_scatter_117 = copy_118 = None
        sub_118 = torch.ops.aten.sub.Tensor(select_1062, view_473);  select_1062 = view_473 = None
        div_118 = torch.ops.aten.div.Tensor(sub_118, select_1061);  sub_118 = select_1061 = None
        select_1067 = torch.ops.aten.select.int(arg1_1, 0, 118)
        slice_470 = torch.ops.aten.slice.Tensor(select_1067, 0, 118, 9223372036854775807);  select_1067 = None
        slice_471 = torch.ops.aten.slice.Tensor(slice_scatter_116, 1, 118, 9223372036854775807)
        expand_237 = torch.ops.aten.expand.default(slice_471, [2048, 10]);  slice_471 = None
        mul_354 = torch.ops.aten.mul.Tensor(expand_237, 1);  expand_237 = None
        view_474 = torch.ops.aten.view.default(div_118, [2048, 1]);  div_118 = None
        mul_355 = torch.ops.aten.mul.Tensor(view_474, slice_470);  view_474 = slice_470 = None
        mul_356 = torch.ops.aten.mul.Tensor(mul_355, -1.0);  mul_355 = None
        add_118 = torch.ops.aten.add.Tensor(mul_354, mul_356);  mul_354 = mul_356 = None
        slice_scatter_117 = torch.ops.aten.slice_scatter.default(slice_scatter_116, add_118, 1, 118, 9223372036854775807);  slice_scatter_116 = add_118 = None
        select_1069 = torch.ops.aten.select.int(arg1_1, 0, 119)
        select_1070 = torch.ops.aten.select.int(select_1069, 0, 119);  select_1069 = None
        select_1071 = torch.ops.aten.select.int(slice_scatter_117, 1, 119)
        unsqueeze_238 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_119 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_1072 = torch.ops.aten.select.int(slice_scatter_117, 1, 119)
        view_476 = torch.ops.aten.view.default(select_1072, [8, 128, 2]);  select_1072 = None
        baddbmm_119 = torch.ops.aten.baddbmm.default(unsqueeze_238, view_476, permute_119, alpha = -2.0);  unsqueeze_238 = view_476 = permute_119 = None
        argmin_119 = torch.ops.aten.argmin.default(baddbmm_119, -1);  baddbmm_119 = None
        unsqueeze_239 = torch.ops.aten.unsqueeze.default(argmin_119, -1)
        expand_238 = torch.ops.aten.expand.default(unsqueeze_239, [8, 128, 2]);  unsqueeze_239 = None
        gather_119 = torch.ops.aten.gather.default(arg3_1, 1, expand_238);  expand_238 = None
        view_477 = torch.ops.aten.view.default(gather_119, [-1]);  gather_119 = None
        select_1074 = torch.ops.aten.select.int(select_scatter_118, 1, 119)
        copy_119 = torch.ops.aten.copy.default(select_1074, argmin_119);  select_1074 = argmin_119 = None
        select_scatter_119 = torch.ops.aten.select_scatter.default(select_scatter_118, copy_119, 1, 119);  select_scatter_118 = copy_119 = None
        sub_119 = torch.ops.aten.sub.Tensor(select_1071, view_477);  select_1071 = view_477 = None
        div_119 = torch.ops.aten.div.Tensor(sub_119, select_1070);  sub_119 = select_1070 = None
        select_1076 = torch.ops.aten.select.int(arg1_1, 0, 119)
        slice_474 = torch.ops.aten.slice.Tensor(select_1076, 0, 119, 9223372036854775807);  select_1076 = None
        slice_475 = torch.ops.aten.slice.Tensor(slice_scatter_117, 1, 119, 9223372036854775807)
        expand_239 = torch.ops.aten.expand.default(slice_475, [2048, 9]);  slice_475 = None
        mul_357 = torch.ops.aten.mul.Tensor(expand_239, 1);  expand_239 = None
        view_478 = torch.ops.aten.view.default(div_119, [2048, 1]);  div_119 = None
        mul_358 = torch.ops.aten.mul.Tensor(view_478, slice_474);  view_478 = slice_474 = None
        mul_359 = torch.ops.aten.mul.Tensor(mul_358, -1.0);  mul_358 = None
        add_119 = torch.ops.aten.add.Tensor(mul_357, mul_359);  mul_357 = mul_359 = None
        slice_scatter_118 = torch.ops.aten.slice_scatter.default(slice_scatter_117, add_119, 1, 119, 9223372036854775807);  slice_scatter_117 = add_119 = None
        select_1078 = torch.ops.aten.select.int(arg1_1, 0, 120)
        select_1079 = torch.ops.aten.select.int(select_1078, 0, 120);  select_1078 = None
        select_1080 = torch.ops.aten.select.int(slice_scatter_118, 1, 120)
        unsqueeze_240 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_120 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_1081 = torch.ops.aten.select.int(slice_scatter_118, 1, 120)
        view_480 = torch.ops.aten.view.default(select_1081, [8, 128, 2]);  select_1081 = None
        baddbmm_120 = torch.ops.aten.baddbmm.default(unsqueeze_240, view_480, permute_120, alpha = -2.0);  unsqueeze_240 = view_480 = permute_120 = None
        argmin_120 = torch.ops.aten.argmin.default(baddbmm_120, -1);  baddbmm_120 = None
        unsqueeze_241 = torch.ops.aten.unsqueeze.default(argmin_120, -1)
        expand_240 = torch.ops.aten.expand.default(unsqueeze_241, [8, 128, 2]);  unsqueeze_241 = None
        gather_120 = torch.ops.aten.gather.default(arg3_1, 1, expand_240);  expand_240 = None
        view_481 = torch.ops.aten.view.default(gather_120, [-1]);  gather_120 = None
        select_1083 = torch.ops.aten.select.int(select_scatter_119, 1, 120)
        copy_120 = torch.ops.aten.copy.default(select_1083, argmin_120);  select_1083 = argmin_120 = None
        select_scatter_120 = torch.ops.aten.select_scatter.default(select_scatter_119, copy_120, 1, 120);  select_scatter_119 = copy_120 = None
        sub_120 = torch.ops.aten.sub.Tensor(select_1080, view_481);  select_1080 = view_481 = None
        div_120 = torch.ops.aten.div.Tensor(sub_120, select_1079);  sub_120 = select_1079 = None
        select_1085 = torch.ops.aten.select.int(arg1_1, 0, 120)
        slice_478 = torch.ops.aten.slice.Tensor(select_1085, 0, 120, 9223372036854775807);  select_1085 = None
        slice_479 = torch.ops.aten.slice.Tensor(slice_scatter_118, 1, 120, 9223372036854775807)
        expand_241 = torch.ops.aten.expand.default(slice_479, [2048, 8]);  slice_479 = None
        mul_360 = torch.ops.aten.mul.Tensor(expand_241, 1);  expand_241 = None
        view_482 = torch.ops.aten.view.default(div_120, [2048, 1]);  div_120 = None
        mul_361 = torch.ops.aten.mul.Tensor(view_482, slice_478);  view_482 = slice_478 = None
        mul_362 = torch.ops.aten.mul.Tensor(mul_361, -1.0);  mul_361 = None
        add_120 = torch.ops.aten.add.Tensor(mul_360, mul_362);  mul_360 = mul_362 = None
        slice_scatter_119 = torch.ops.aten.slice_scatter.default(slice_scatter_118, add_120, 1, 120, 9223372036854775807);  slice_scatter_118 = add_120 = None
        select_1087 = torch.ops.aten.select.int(arg1_1, 0, 121)
        select_1088 = torch.ops.aten.select.int(select_1087, 0, 121);  select_1087 = None
        select_1089 = torch.ops.aten.select.int(slice_scatter_119, 1, 121)
        unsqueeze_242 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_121 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_1090 = torch.ops.aten.select.int(slice_scatter_119, 1, 121)
        view_484 = torch.ops.aten.view.default(select_1090, [8, 128, 2]);  select_1090 = None
        baddbmm_121 = torch.ops.aten.baddbmm.default(unsqueeze_242, view_484, permute_121, alpha = -2.0);  unsqueeze_242 = view_484 = permute_121 = None
        argmin_121 = torch.ops.aten.argmin.default(baddbmm_121, -1);  baddbmm_121 = None
        unsqueeze_243 = torch.ops.aten.unsqueeze.default(argmin_121, -1)
        expand_242 = torch.ops.aten.expand.default(unsqueeze_243, [8, 128, 2]);  unsqueeze_243 = None
        gather_121 = torch.ops.aten.gather.default(arg3_1, 1, expand_242);  expand_242 = None
        view_485 = torch.ops.aten.view.default(gather_121, [-1]);  gather_121 = None
        select_1092 = torch.ops.aten.select.int(select_scatter_120, 1, 121)
        copy_121 = torch.ops.aten.copy.default(select_1092, argmin_121);  select_1092 = argmin_121 = None
        select_scatter_121 = torch.ops.aten.select_scatter.default(select_scatter_120, copy_121, 1, 121);  select_scatter_120 = copy_121 = None
        sub_121 = torch.ops.aten.sub.Tensor(select_1089, view_485);  select_1089 = view_485 = None
        div_121 = torch.ops.aten.div.Tensor(sub_121, select_1088);  sub_121 = select_1088 = None
        select_1094 = torch.ops.aten.select.int(arg1_1, 0, 121)
        slice_482 = torch.ops.aten.slice.Tensor(select_1094, 0, 121, 9223372036854775807);  select_1094 = None
        slice_483 = torch.ops.aten.slice.Tensor(slice_scatter_119, 1, 121, 9223372036854775807)
        expand_243 = torch.ops.aten.expand.default(slice_483, [2048, 7]);  slice_483 = None
        mul_363 = torch.ops.aten.mul.Tensor(expand_243, 1);  expand_243 = None
        view_486 = torch.ops.aten.view.default(div_121, [2048, 1]);  div_121 = None
        mul_364 = torch.ops.aten.mul.Tensor(view_486, slice_482);  view_486 = slice_482 = None
        mul_365 = torch.ops.aten.mul.Tensor(mul_364, -1.0);  mul_364 = None
        add_121 = torch.ops.aten.add.Tensor(mul_363, mul_365);  mul_363 = mul_365 = None
        slice_scatter_120 = torch.ops.aten.slice_scatter.default(slice_scatter_119, add_121, 1, 121, 9223372036854775807);  slice_scatter_119 = add_121 = None
        select_1096 = torch.ops.aten.select.int(arg1_1, 0, 122)
        select_1097 = torch.ops.aten.select.int(select_1096, 0, 122);  select_1096 = None
        select_1098 = torch.ops.aten.select.int(slice_scatter_120, 1, 122)
        unsqueeze_244 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_122 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_1099 = torch.ops.aten.select.int(slice_scatter_120, 1, 122)
        view_488 = torch.ops.aten.view.default(select_1099, [8, 128, 2]);  select_1099 = None
        baddbmm_122 = torch.ops.aten.baddbmm.default(unsqueeze_244, view_488, permute_122, alpha = -2.0);  unsqueeze_244 = view_488 = permute_122 = None
        argmin_122 = torch.ops.aten.argmin.default(baddbmm_122, -1);  baddbmm_122 = None
        unsqueeze_245 = torch.ops.aten.unsqueeze.default(argmin_122, -1)
        expand_244 = torch.ops.aten.expand.default(unsqueeze_245, [8, 128, 2]);  unsqueeze_245 = None
        gather_122 = torch.ops.aten.gather.default(arg3_1, 1, expand_244);  expand_244 = None
        view_489 = torch.ops.aten.view.default(gather_122, [-1]);  gather_122 = None
        select_1101 = torch.ops.aten.select.int(select_scatter_121, 1, 122)
        copy_122 = torch.ops.aten.copy.default(select_1101, argmin_122);  select_1101 = argmin_122 = None
        select_scatter_122 = torch.ops.aten.select_scatter.default(select_scatter_121, copy_122, 1, 122);  select_scatter_121 = copy_122 = None
        sub_122 = torch.ops.aten.sub.Tensor(select_1098, view_489);  select_1098 = view_489 = None
        div_122 = torch.ops.aten.div.Tensor(sub_122, select_1097);  sub_122 = select_1097 = None
        select_1103 = torch.ops.aten.select.int(arg1_1, 0, 122)
        slice_486 = torch.ops.aten.slice.Tensor(select_1103, 0, 122, 9223372036854775807);  select_1103 = None
        slice_487 = torch.ops.aten.slice.Tensor(slice_scatter_120, 1, 122, 9223372036854775807)
        expand_245 = torch.ops.aten.expand.default(slice_487, [2048, 6]);  slice_487 = None
        mul_366 = torch.ops.aten.mul.Tensor(expand_245, 1);  expand_245 = None
        view_490 = torch.ops.aten.view.default(div_122, [2048, 1]);  div_122 = None
        mul_367 = torch.ops.aten.mul.Tensor(view_490, slice_486);  view_490 = slice_486 = None
        mul_368 = torch.ops.aten.mul.Tensor(mul_367, -1.0);  mul_367 = None
        add_122 = torch.ops.aten.add.Tensor(mul_366, mul_368);  mul_366 = mul_368 = None
        slice_scatter_121 = torch.ops.aten.slice_scatter.default(slice_scatter_120, add_122, 1, 122, 9223372036854775807);  slice_scatter_120 = add_122 = None
        select_1105 = torch.ops.aten.select.int(arg1_1, 0, 123)
        select_1106 = torch.ops.aten.select.int(select_1105, 0, 123);  select_1105 = None
        select_1107 = torch.ops.aten.select.int(slice_scatter_121, 1, 123)
        unsqueeze_246 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_123 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_1108 = torch.ops.aten.select.int(slice_scatter_121, 1, 123)
        view_492 = torch.ops.aten.view.default(select_1108, [8, 128, 2]);  select_1108 = None
        baddbmm_123 = torch.ops.aten.baddbmm.default(unsqueeze_246, view_492, permute_123, alpha = -2.0);  unsqueeze_246 = view_492 = permute_123 = None
        argmin_123 = torch.ops.aten.argmin.default(baddbmm_123, -1);  baddbmm_123 = None
        unsqueeze_247 = torch.ops.aten.unsqueeze.default(argmin_123, -1)
        expand_246 = torch.ops.aten.expand.default(unsqueeze_247, [8, 128, 2]);  unsqueeze_247 = None
        gather_123 = torch.ops.aten.gather.default(arg3_1, 1, expand_246);  expand_246 = None
        view_493 = torch.ops.aten.view.default(gather_123, [-1]);  gather_123 = None
        select_1110 = torch.ops.aten.select.int(select_scatter_122, 1, 123)
        copy_123 = torch.ops.aten.copy.default(select_1110, argmin_123);  select_1110 = argmin_123 = None
        select_scatter_123 = torch.ops.aten.select_scatter.default(select_scatter_122, copy_123, 1, 123);  select_scatter_122 = copy_123 = None
        sub_123 = torch.ops.aten.sub.Tensor(select_1107, view_493);  select_1107 = view_493 = None
        div_123 = torch.ops.aten.div.Tensor(sub_123, select_1106);  sub_123 = select_1106 = None
        select_1112 = torch.ops.aten.select.int(arg1_1, 0, 123)
        slice_490 = torch.ops.aten.slice.Tensor(select_1112, 0, 123, 9223372036854775807);  select_1112 = None
        slice_491 = torch.ops.aten.slice.Tensor(slice_scatter_121, 1, 123, 9223372036854775807)
        expand_247 = torch.ops.aten.expand.default(slice_491, [2048, 5]);  slice_491 = None
        mul_369 = torch.ops.aten.mul.Tensor(expand_247, 1);  expand_247 = None
        view_494 = torch.ops.aten.view.default(div_123, [2048, 1]);  div_123 = None
        mul_370 = torch.ops.aten.mul.Tensor(view_494, slice_490);  view_494 = slice_490 = None
        mul_371 = torch.ops.aten.mul.Tensor(mul_370, -1.0);  mul_370 = None
        add_123 = torch.ops.aten.add.Tensor(mul_369, mul_371);  mul_369 = mul_371 = None
        slice_scatter_122 = torch.ops.aten.slice_scatter.default(slice_scatter_121, add_123, 1, 123, 9223372036854775807);  slice_scatter_121 = add_123 = None
        select_1114 = torch.ops.aten.select.int(arg1_1, 0, 124)
        select_1115 = torch.ops.aten.select.int(select_1114, 0, 124);  select_1114 = None
        select_1116 = torch.ops.aten.select.int(slice_scatter_122, 1, 124)
        unsqueeze_248 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_124 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_1117 = torch.ops.aten.select.int(slice_scatter_122, 1, 124)
        view_496 = torch.ops.aten.view.default(select_1117, [8, 128, 2]);  select_1117 = None
        baddbmm_124 = torch.ops.aten.baddbmm.default(unsqueeze_248, view_496, permute_124, alpha = -2.0);  unsqueeze_248 = view_496 = permute_124 = None
        argmin_124 = torch.ops.aten.argmin.default(baddbmm_124, -1);  baddbmm_124 = None
        unsqueeze_249 = torch.ops.aten.unsqueeze.default(argmin_124, -1)
        expand_248 = torch.ops.aten.expand.default(unsqueeze_249, [8, 128, 2]);  unsqueeze_249 = None
        gather_124 = torch.ops.aten.gather.default(arg3_1, 1, expand_248);  expand_248 = None
        view_497 = torch.ops.aten.view.default(gather_124, [-1]);  gather_124 = None
        select_1119 = torch.ops.aten.select.int(select_scatter_123, 1, 124)
        copy_124 = torch.ops.aten.copy.default(select_1119, argmin_124);  select_1119 = argmin_124 = None
        select_scatter_124 = torch.ops.aten.select_scatter.default(select_scatter_123, copy_124, 1, 124);  select_scatter_123 = copy_124 = None
        sub_124 = torch.ops.aten.sub.Tensor(select_1116, view_497);  select_1116 = view_497 = None
        div_124 = torch.ops.aten.div.Tensor(sub_124, select_1115);  sub_124 = select_1115 = None
        select_1121 = torch.ops.aten.select.int(arg1_1, 0, 124)
        slice_494 = torch.ops.aten.slice.Tensor(select_1121, 0, 124, 9223372036854775807);  select_1121 = None
        slice_495 = torch.ops.aten.slice.Tensor(slice_scatter_122, 1, 124, 9223372036854775807)
        expand_249 = torch.ops.aten.expand.default(slice_495, [2048, 4]);  slice_495 = None
        mul_372 = torch.ops.aten.mul.Tensor(expand_249, 1);  expand_249 = None
        view_498 = torch.ops.aten.view.default(div_124, [2048, 1]);  div_124 = None
        mul_373 = torch.ops.aten.mul.Tensor(view_498, slice_494);  view_498 = slice_494 = None
        mul_374 = torch.ops.aten.mul.Tensor(mul_373, -1.0);  mul_373 = None
        add_124 = torch.ops.aten.add.Tensor(mul_372, mul_374);  mul_372 = mul_374 = None
        slice_scatter_123 = torch.ops.aten.slice_scatter.default(slice_scatter_122, add_124, 1, 124, 9223372036854775807);  slice_scatter_122 = add_124 = None
        select_1123 = torch.ops.aten.select.int(arg1_1, 0, 125)
        select_1124 = torch.ops.aten.select.int(select_1123, 0, 125);  select_1123 = None
        select_1125 = torch.ops.aten.select.int(slice_scatter_123, 1, 125)
        unsqueeze_250 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_125 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_1126 = torch.ops.aten.select.int(slice_scatter_123, 1, 125)
        view_500 = torch.ops.aten.view.default(select_1126, [8, 128, 2]);  select_1126 = None
        baddbmm_125 = torch.ops.aten.baddbmm.default(unsqueeze_250, view_500, permute_125, alpha = -2.0);  unsqueeze_250 = view_500 = permute_125 = None
        argmin_125 = torch.ops.aten.argmin.default(baddbmm_125, -1);  baddbmm_125 = None
        unsqueeze_251 = torch.ops.aten.unsqueeze.default(argmin_125, -1)
        expand_250 = torch.ops.aten.expand.default(unsqueeze_251, [8, 128, 2]);  unsqueeze_251 = None
        gather_125 = torch.ops.aten.gather.default(arg3_1, 1, expand_250);  expand_250 = None
        view_501 = torch.ops.aten.view.default(gather_125, [-1]);  gather_125 = None
        select_1128 = torch.ops.aten.select.int(select_scatter_124, 1, 125)
        copy_125 = torch.ops.aten.copy.default(select_1128, argmin_125);  select_1128 = argmin_125 = None
        select_scatter_125 = torch.ops.aten.select_scatter.default(select_scatter_124, copy_125, 1, 125);  select_scatter_124 = copy_125 = None
        sub_125 = torch.ops.aten.sub.Tensor(select_1125, view_501);  select_1125 = view_501 = None
        div_125 = torch.ops.aten.div.Tensor(sub_125, select_1124);  sub_125 = select_1124 = None
        select_1130 = torch.ops.aten.select.int(arg1_1, 0, 125)
        slice_498 = torch.ops.aten.slice.Tensor(select_1130, 0, 125, 9223372036854775807);  select_1130 = None
        slice_499 = torch.ops.aten.slice.Tensor(slice_scatter_123, 1, 125, 9223372036854775807)
        expand_251 = torch.ops.aten.expand.default(slice_499, [2048, 3]);  slice_499 = None
        mul_375 = torch.ops.aten.mul.Tensor(expand_251, 1);  expand_251 = None
        view_502 = torch.ops.aten.view.default(div_125, [2048, 1]);  div_125 = None
        mul_376 = torch.ops.aten.mul.Tensor(view_502, slice_498);  view_502 = slice_498 = None
        mul_377 = torch.ops.aten.mul.Tensor(mul_376, -1.0);  mul_376 = None
        add_125 = torch.ops.aten.add.Tensor(mul_375, mul_377);  mul_375 = mul_377 = None
        slice_scatter_124 = torch.ops.aten.slice_scatter.default(slice_scatter_123, add_125, 1, 125, 9223372036854775807);  slice_scatter_123 = add_125 = None
        select_1132 = torch.ops.aten.select.int(arg1_1, 0, 126)
        select_1133 = torch.ops.aten.select.int(select_1132, 0, 126);  select_1132 = None
        select_1134 = torch.ops.aten.select.int(slice_scatter_124, 1, 126)
        unsqueeze_252 = torch.ops.aten.unsqueeze.default(arg2_1, 1)
        permute_126 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_1135 = torch.ops.aten.select.int(slice_scatter_124, 1, 126)
        view_504 = torch.ops.aten.view.default(select_1135, [8, 128, 2]);  select_1135 = None
        baddbmm_126 = torch.ops.aten.baddbmm.default(unsqueeze_252, view_504, permute_126, alpha = -2.0);  unsqueeze_252 = view_504 = permute_126 = None
        argmin_126 = torch.ops.aten.argmin.default(baddbmm_126, -1);  baddbmm_126 = None
        unsqueeze_253 = torch.ops.aten.unsqueeze.default(argmin_126, -1)
        expand_252 = torch.ops.aten.expand.default(unsqueeze_253, [8, 128, 2]);  unsqueeze_253 = None
        gather_126 = torch.ops.aten.gather.default(arg3_1, 1, expand_252);  expand_252 = None
        view_505 = torch.ops.aten.view.default(gather_126, [-1]);  gather_126 = None
        select_1137 = torch.ops.aten.select.int(select_scatter_125, 1, 126)
        copy_126 = torch.ops.aten.copy.default(select_1137, argmin_126);  select_1137 = argmin_126 = None
        select_scatter_126 = torch.ops.aten.select_scatter.default(select_scatter_125, copy_126, 1, 126);  select_scatter_125 = copy_126 = None
        sub_126 = torch.ops.aten.sub.Tensor(select_1134, view_505);  select_1134 = view_505 = None
        div_126 = torch.ops.aten.div.Tensor(sub_126, select_1133);  sub_126 = select_1133 = None
        select_1139 = torch.ops.aten.select.int(arg1_1, 0, 126)
        slice_502 = torch.ops.aten.slice.Tensor(select_1139, 0, 126, 9223372036854775807);  select_1139 = None
        slice_503 = torch.ops.aten.slice.Tensor(slice_scatter_124, 1, 126, 9223372036854775807)
        expand_253 = torch.ops.aten.expand.default(slice_503, [2048, 2]);  slice_503 = None
        mul_378 = torch.ops.aten.mul.Tensor(expand_253, 1);  expand_253 = None
        view_506 = torch.ops.aten.view.default(div_126, [2048, 1]);  div_126 = None
        mul_379 = torch.ops.aten.mul.Tensor(view_506, slice_502);  view_506 = slice_502 = None
        mul_380 = torch.ops.aten.mul.Tensor(mul_379, -1.0);  mul_379 = None
        add_126 = torch.ops.aten.add.Tensor(mul_378, mul_380);  mul_378 = mul_380 = None
        slice_scatter_125 = torch.ops.aten.slice_scatter.default(slice_scatter_124, add_126, 1, 126, 9223372036854775807);  slice_scatter_124 = add_126 = None
        select_1141 = torch.ops.aten.select.int(arg1_1, 0, 127)
        select_1142 = torch.ops.aten.select.int(select_1141, 0, 127);  select_1141 = None
        select_1143 = torch.ops.aten.select.int(slice_scatter_125, 1, 127)
        unsqueeze_254 = torch.ops.aten.unsqueeze.default(arg2_1, 1);  arg2_1 = None
        permute_127 = torch.ops.aten.permute.default(arg3_1, [0, 2, 1])
        select_1144 = torch.ops.aten.select.int(slice_scatter_125, 1, 127)
        view_508 = torch.ops.aten.view.default(select_1144, [8, 128, 2]);  select_1144 = None
        baddbmm_127 = torch.ops.aten.baddbmm.default(unsqueeze_254, view_508, permute_127, alpha = -2.0);  unsqueeze_254 = view_508 = permute_127 = None
        argmin_127 = torch.ops.aten.argmin.default(baddbmm_127, -1);  baddbmm_127 = None
        unsqueeze_255 = torch.ops.aten.unsqueeze.default(argmin_127, -1)
        expand_254 = torch.ops.aten.expand.default(unsqueeze_255, [8, 128, 2]);  unsqueeze_255 = None
        gather_127 = torch.ops.aten.gather.default(arg3_1, 1, expand_254);  arg3_1 = expand_254 = None
        view_509 = torch.ops.aten.view.default(gather_127, [-1]);  gather_127 = None
        select_1146 = torch.ops.aten.select.int(select_scatter_126, 1, 127)
        copy_127 = torch.ops.aten.copy.default(select_1146, argmin_127);  select_1146 = argmin_127 = None
        select_scatter_127 = torch.ops.aten.select_scatter.default(select_scatter_126, copy_127, 1, 127);  select_scatter_126 = copy_127 = None
        sub_127 = torch.ops.aten.sub.Tensor(select_1143, view_509);  select_1143 = view_509 = None
        div_127 = torch.ops.aten.div.Tensor(sub_127, select_1142);  sub_127 = select_1142 = None
        select_1148 = torch.ops.aten.select.int(arg1_1, 0, 127);  arg1_1 = None
        slice_506 = torch.ops.aten.slice.Tensor(select_1148, 0, 127, 9223372036854775807);  select_1148 = None
        slice_507 = torch.ops.aten.slice.Tensor(slice_scatter_125, 1, 127, 9223372036854775807)
        expand_255 = torch.ops.aten.expand.default(slice_507, [2048, 1]);  slice_507 = None
        mul_381 = torch.ops.aten.mul.Tensor(expand_255, 1);  expand_255 = None
        view_510 = torch.ops.aten.view.default(div_127, [2048, 1]);  div_127 = None
        mul_382 = torch.ops.aten.mul.Tensor(view_510, slice_506);  view_510 = slice_506 = None
        mul_383 = torch.ops.aten.mul.Tensor(mul_382, -1.0);  mul_382 = None
        add_127 = torch.ops.aten.add.Tensor(mul_381, mul_383);  mul_381 = mul_383 = None
        slice_scatter_126 = torch.ops.aten.slice_scatter.default(slice_scatter_125, add_127, 1, 127, 9223372036854775807);  slice_scatter_125 = add_127 = None
        copy_ = torch.ops.aten.copy_.default(arg0_1, slice_scatter_126);  arg0_1 = slice_scatter_126 = copy_ = None
        return (select_scatter_127,)
        
def load_args(reader):
    buf0 = reader.storage(None, 1048576, device=device(type='npu', index=0))
    reader.tensor(buf0, (2048, 128), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 65536, device=device(type='npu', index=0))
    reader.tensor(buf1, (128, 128), is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 512, device=device(type='npu', index=0))
    reader.tensor(buf2, (8, 16), is_leaf=True)  # arg2_1
    buf3 = reader.storage(None, 1024, device=device(type='npu', index=0))
    reader.tensor(buf3, (8, 16, 2), is_leaf=True)  # arg3_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)