# VPTQ Quantizer（②③ 阶段）

VPTQ 官方未开源的量化器实现：Hessian 加权码本初始化 + 逐列二阶
向量量化 + 残差 VQ。输入为 `vptq/tools/hessian` 采集的
`H` / `H⁻¹`（layer_XXXX.pt），输出可直接装配成推理侧的
`VQuantLinear`。

## 模块

| 文件 | 说明 |
|---|---|
| `config.py` | `QuantConfig`：vector_len / num_centroids / 残差码本 / 分组；含官方预设（`v8_k65536_65536` ≈2.06bit、`v8_k65536_256` ≈3bit、`v16_k65536_65536` = 2bit） |
| `vq.py` | 核心算法：Hessian 加权 K-means（k-means++ 播种，k>2048 自动切换随机播种防 O(k²) 爆炸）→ GPTQ 式分块逐列量化（H⁻¹ 上 Cholesky 误差传播）→ 可选残差 VQ；产出 `QuantizedWeight`（含 proxy_error / mse） |
| `kmeans.py` | 按维加权的 K-means 变体（实验参考） |
| `hessian_io.py` | `HessianStore` 懒加载层文件；`hessian_key_of` 权重名→Hessian 组映射（w1/w3 共享 mlp_in，router 复用共享专家，wo_a 逐组 g0~g7） |
| `export.py` | `QuantizedWeight` → `VQuantLinear`（布局转置 (g,cols,nvec)→(g,nvec,cols)，key 0 outlier 占位）；`reconstruct_weight` 参考重建 |
| `run.py` | 全模型驱动：按层量化 DeepSeek-V4 全部线性（注意力 / router / 共享专家 / 256 路由专家 / wo_a 逐组），输出 quant_layer_XXXX.pt + quant_report.json |

## 用法

```bash
python -m vptq.tools.quantize.run \
    --ckpt /mnt/share/weight/DeepSeek-V4-Flash-BF16 \
    --hessian-dir /mnt/share/rr08002/weights/hessians/DeepSeek-V4-Flash-BF16 \
    --output-dir ./quant_out \
    --vector-len 8 --num-centroids 65536 --num-res-centroids 65536 \
    --layer-range 0:43 --expert-range 0:256 --device npu
```

专家间相互独立，可用 `--expert-range` 起多个进程分片（如 8 进程 ×
32 专家，每进程绑一张 NPU）。

### 分范围量化（--scope，支持混合位宽）

`--scope` 取 `all,attn,router,shared,experts` 的逗方子集：

```bash
# 仅量化 MoE 路由专家（注意力/router/共享专家保持 BF16）
python -m vptq.tools.quantize.run ... --scope experts

# 混合位宽：先跑专家 2bit，再跑注意力 4bit（结果合并进同一文件，
# 已有权重自动跳过，天然支持断点续跑）
python -m vptq.tools.quantize.run ... --scope experts \
    --vector-len 16 --num-centroids 65536 --num-res-centroids 65536
python -m vptq.tools.quantize.run ... --scope attn,router,shared \
    --vector-len 8 --num-centroids 65536 --num-res-centroids 65536
```

## 实测（k=65536, v8, NPU）

- 速度：~5.7s / 矩阵（2048×4096）→ 全模型 ~34k 矩阵，8 进程约 7h
- 质量：主量化 proxy_error ≈ 0.2%（Hessian 度量）；残差 VQ 再降一半
- proxy_error ≪ mse（0.2% vs 14%）——二阶优化把误差集中到
  Hessian 认为不重要的方向，这是 VPTQ 优于朴素 VQ 的核心证据

## 已修复的集成问题

- `vq.py` indices 堆叠多维（(g,1,cols,nvec)→(g,cols,nvec)）
- `hessian_key_of`：`.weight` 后缀与无前缀名的双重匹配 bug
- `vqlinear.py`：残差+非打包构造时 `num_indices` 未定义（仓库既有 bug）
- `init_parameters` 约定 dict key 0 为 outlier 槽（无 outlier 需占位）
