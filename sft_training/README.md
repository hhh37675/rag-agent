# 业务大模型 SFT 微调记录

本项目基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 框架对 Qwen2.5:1.5B 进行了指令微调 (SFT)，以提升其在垂直业务场景（如企业规章制度、报销流程答疑）下的回复专业度和准确性。

## 1. 训练环境与显存占用
- **框架**: LLaMA-Factory
- **微调方法**: LoRA (Low-Rank Adaptation)
- **精度**: FP16 (混合精度训练)
- **显存占用评估**: 
  - 1.5B 模型在 FP16 下裸模型占用约 3GB 显存。
  - 采用 LoRA 并设置 `cutoff_len=1024`, `batch_size=4`，峰值显存控制在 6GB-8GB 之间，普通消费级单卡（如 RTX 3060/4060）即可流畅运行。

## 2. 训练启动命令
在 LLaMA-Factory 根目录下执行以下命令启动训练：
```bash
llamafactory-cli train qwen2.5_lora_sft.yaml