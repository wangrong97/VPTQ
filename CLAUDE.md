# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

VPTQ（Vector Post-Training Quantization）：基于向量量化（VQ）的 LLM 极低 bit（<2bit）后训练量化，EMNLP 2024。上游为 microsoft/VPTQ；本仓库是工作 fork，当前分支 `pr_w00608002_vptq_deepseek_moe_w2a8` 在 **Ascend NPU** 上对 DeepSeek-V4-Flash-BF16（272B，256 路由专家 MoE）做专家 W2 量化。

**运行环境（aarch64 + Ascend NPU，无 CUDA）**：`libvptq.so` CUDA 内核无法在本机构建；量化工具链为纯 PyTorch 实现，通过 `--device npu` + `ASCEND_RT_VISIBLE_DEVICES` 运行。修改 `csrc/` 需在有 GPU 的环境验证。注意：本环境存在周期性 killer 进程（约 20-25 分钟杀一次任务），长任务依赖根目录 shell 脚本的看门狗 + 幂等落盘设计。

## 常用命令

```bash
# 构建（需要 CUDA toolkit + nvcc；CMake 编译 csrc/ 产出 vptq/libvptq.so）
python setup.py build bdist_wheel
pip install dist/vptq-*.whl
pip install -e .          # 开发模式（Develop 命令会把 libvptq.so 拷回 vptq/）
python setup.py clean

# 测试（test_quant_gemv.py 需要 GPU + 已编译内核；量化工具链测试纯 CPU 可跑）
pytest tests/ -v -s
pytest tests/test_quantize.py -v -s        # 单个文件
pytest tests/test_quantize.py::test_name   # 单个测试

# Lint / 格式化（black line-length=80；isort profile=black；clang-format 18）
pre-commit run --all-files
pre-commit install

# 推理 CLI（HuggingFace VPTQ 量化模型）
python -m vptq --model=VPTQ-community/... --prompt="..."
python -m vptq --model=... --chat
```

## 架构

仓库分三块：**推理侧**（上游核心，CUDA）、**量化侧**（本 fork 补齐的官方未开源量化器，纯 torch）、**DeepSeek-V4 NPU 适配**（本 fork 新增）。

### 推理侧（CUDA kernel + transformers 集成）

- `vptq/layers/vqlinear.py`：`VQuantLinear` —— 向量量化 Linear 层。`vector_lens` / `num_centroids` / `num_res_centroids` 均为二元组：**[0] 是 outlier 分量，[1] 是主分量**；`num_res_centroids=-1` 表示无残差量化。码本 dict 的 key 0 恒为 outlier 占位槽（无 outlier 也要占位）。
- `vptq/layers/model_base.py`：`vptq.AutoModelForCausalLM.from_pretrained` 以 meta/init_empty 加载 HF 模型后，按 config 把 Linear 替换为 `VQuantLinear`（`make_quant_linear` / `set_op_by_name`）。
- `vptq/ops/quant_gemm.py`：算子分派层。token 数 < 3 走 `quant_gemv`（decode 路径），否则 `dequant` 重建权重 + `F.linear`（prefill 路径）。成功 `import vptq.libvptq` 才用 CUDA 内核，否则回退纯 torch 实现（会打印警告，极慢）。
- `csrc/`：CUTLASS 风格 CUDA 内核（`quant_gemv.cu` / `quant_gemv_v2.cu` / `dequant.cu`，kernel 模板在 `csrc/kernels/`），pybind11 注册为 `libvptq` 模块（三个入口：`dequant` / `quant_gemv` / `quant_gemv_v2`）。依赖 `third_party/cutlass` submodule（构建时自动 `git submodule update --init`）。

### 量化侧流水线（vptq/tools/，纯 torch，NPU/CPU 可跑）

流程：**Hessian 采集 → 量化 → 导出装配**，各阶段间以磁盘 `.pt` 文件解耦。

1. `vptq/tools/hessian/`（`python -m vptq.tools.hessian`）：hook 式采集各 Linear 输入的 `H = E[xxᵀ]`（运行均值 + token 数）与阻尼 Cholesky 逆（GPTQ 式 damping）。共享输入的 Linear 共享一个 Hessian（q/k/v → `attn_in`，gate/up → `mlp_in`）。输出 `layer_XXXX.pt`，可按 `--layer-range` 分片采集后合并。
2. `vptq/tools/quantize/`（`python -m vptq.tools.quantize.run`）：
   - `vq.py` 核心算法：Hessian 对角加权 K-means 码本初始化（k>2048 自动随机播种防 O(k²)）→ GPTQ 式分块逐列量化（H⁻¹ 上 Cholesky 误差传播）→ 可选残差 VQ。产物 `QuantizedWeight`（含 proxy_error，Hessian 度量；proxy_error ≪ mse 是二阶优化有效的核心证据）。
   - `config.py`：`QuantConfig` 预设（如 `v8_k65536_65536` ≈2.06bit、`v16_k65536_65536` = 2bit）。
   - `hessian_io.py`：权重名 → Hessian 组映射（w1/w3 共享 `mlp_in`，router 复用共享专家，wo_a 逐组 g0~g7）。
   - `export.py`：`QuantizedWeight` → `VQuantLinear`（布局转置 (g,cols,nvec)→(g,nvec,cols)）；`rebuild_bf16.py` 重建 BF16 部署权重。
   - `run.py` 关键参数：`--scope all,attn,router,shared,experts`（支持混合位宽分批量化）、`--layer-range` / `--expert-range`（多进程分片，每进程绑一卡）、已有权重自动跳过（天然断点续跑）。逐矩阵落盘到 `.parts/` 目录，全部完成后合并。
3. `vptq/tools/deepseek_v4/`：DeepSeek-V4-Flash 的 NPU/CPU 适配 —— `kernels_torch.py`（tilelang kernel 的纯 torch 等价：sparse_attn / hc_split_sinkhorn / hadamard 等）、`model.py`（vendored 自 DeepSeek-AI inference/model.py，Apache-2.0）、`loader.py`（层驻留 `cpu`/`stream`/`map` 三模式）、`collect_hessian.py`。256 个路由专家的 Hessian 相互独立（`ffn.experts.N.mlp_in` / `.w2`）。

### 根目录 shell 脚本（当前分支的实验工作流）

`quantize_dsv4_experts*.sh` 系列：DeepSeek-V4 专家量化一键启动 —— 8 卡分片（每卡 32 专家 × 43 层）+ 看门狗（进程被杀后自动重启）+ `.parts` 逐矩阵落盘（重启零损失）。`start|status|stop` 子命令。脚本幂等，直接重跑即续跑。各变体对应不同量化方案（tile16 / tile16_rot / tile16_dualnorm / nores 等），关键参数写在脚本头注释。`topup_rpmix.sh` / `calib_collect.sh` 为 Hessian 补采与校准采集；`watchdog_check.sh` 是 cron 调用的元看门狗。

会话级工作状态与实验线索记录在根目录 `SESSION_CONTEXT_*.md`（非代码，交接用）。

## 关键约定

- **索引/perm 存储类型**：一律 uint16 存储，但 view 成 float16 或 int16 使用（规避 NCCL 与 safetensors 对 uint16 的检查）；解码时再 view 回 uint16。
- **Python 代码风格**：black，line-length=80；isort profile=black。C++/CUDA 走 `.clang-format`（CI 对 `csrc/` 全量 dry-run 检查）。
- **模型位宽估算**：bits/weight = (log2(主码本) + log2(残差码本)) / vector_len，不含码本自身与索引 padding 开销。
- 量化输出目录下 `quant_report.json` 汇总各矩阵 proxy_error，是质量对比的首要指标。
