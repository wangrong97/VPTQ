# DeepSeek-V4 NPU/CPU 适配与 Hessian 采集

本目录适配 **DeepSeek-V4-Flash-BF16**（`deepseek_v4` 架构，272B，MLA +
DSA indexer + Hyper-Connections + 256 专家 MoE），使其无需 GPU 专属
依赖（tilelang、fast_hadamard_transform）即可在 **Ascend NPU / CPU**
上运行，用于 VPTQ 的 Hessian 采集。

## 组成

| 文件 | 说明 |
|---|---|
| `kernels_torch.py` | tilelang kernel 的纯 PyTorch 等价实现：`sparse_attn`（top-k 索引注意力 + attention sink）、`hc_split_sinkhorn`（Hyper-Connections 混合）、`act_quant` / `fp4_act_quant`（QAT 模拟）、`hadamard_transform` |
| `model.py` | 从 `inference/model.py` vendor 而来（Apache-2.0, DeepSeek-AI），补丁：kernel 导入替换、Hadamard 替换、设备安全（masked_fill、索引上设备）、`wo_a_input` Identity 钩子点（wo_a 经 einsum 消费，无模块调用） |
| `loader.py` | `ModelArgs` 配置映射（强制 BF16）、分片 safetensors 加载（按参数 dtype 转换）、层流式前向驱动（`cpu` / `stream` / `map` 三种驻留模式） |
| `collect_hessian.py` | Hessian 采集 CLI，逐层挂钩、逐层保存释放 |

## Hessian 采集

```bash
python -m vptq.tools.deepseek_v4.collect_hessian \
    --ckpt /mnt/share/weight/DeepSeek-V4-Flash-BF16 \
    --data-dir /mnt/share/rr08002/weights/RedPajama-Data-1T-Sample \
    --output-dir ./hessians/dsv4-flash \
    --nsamples 32 --seqlen 4096 --residency cpu
```

- 每层输出 `layer_XXXX.pt`：注意力各投影（`attn.wq_a/wq_b/wkv/wo_b`）、
  compressor/indexer、`attn.wo_a_input.g0..g7`（wo_a 逐组输入 Hessian）、
  **每个路由专家独立**的 `ffn.experts.N.mlp_in`（w1/w3 共享输入）与
  `ffn.experts.N.w2`、共享专家与 MTP 之外的全部线性输入。
- 路由门控 `ffn.gate` 无模块边界（函数式 linear），其输入与共享专家
  相同，复用 `ffn.shared_experts.mlp_in` 即可。
- `--residency stream`：层驻留 CPU、逐层上 NPU 计算（单卡 ~15GB 空闲
  即可）；`--residency map`：预先把层分布到多卡。
- `--n-layer-limit N`：只建/载前 N 层，用于流水线冒烟。

## 资源估算（272B BF16）

- CPU 驻留：权重 543GB 内存；43 层全量约 40 小时（32×4096 tokens）。
- 每层 Hessian 约 21GB fp32（256 专家 × (4096²+2048²) + 注意力），
  全模型约 900GB 磁盘；`--save-inv` 默认关闭（11008 个 Cholesky 代价高）。
