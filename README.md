# slipstream

本地 LLM 推理框架,核心是 [mlx-lm](https://github.com/ml-explore/mlx-lm)。

目标:**在保住 MTP 投机解码加速的同时,做真批量同时解码 + 序列随进随出** —— 这三者的组合现有引擎(mlx-lm / MTPLX / oMLX / llama.cpp)都没打通:它们一旦 batch_size>1 就关掉 MTP。

> 名字待定,当前包名 `slipstream` 是临时的。

## 设计原则

- **直筒,不留分支。** 需求是一条直路就只实现这一条:不写用不到的路径、不写 fallback、不写"以防万一"的兜底。用到什么就断言什么,拿不到直接报错。
- **要深,不要宽。** 把真正要的那条路做到底、做对、做透,而非为多种情况各留一条浅路。

## 分层

由核心到外围。每层单独一个文档(`docs/<layer>.md`,随开发补齐)。

| 层 | 职责 | 状态 |
|---|---|---|
| L1 引擎 | 批量前向:prefill / next-k forward / rollback。不采样、不调度。 | ✅ 可用 |
| L2 投机核 | 批量 MTP:draft → verify → 取最小 commit → 回滚 | 未做 |
| L3 执行 | 批量状态机,一次 step 推进整批一轮 | 未做 |
| L4 调度 | prefill/decode 协调,序列随进随出 | 未做 |
| L5 接口 | Python API / CLI / (later) HTTP | 体验用 CLI |

## 引擎层(L1)现状

一个纯粹的批量前向原语,只做三件事,不多:

- **batch** — B 条序列一次 forward。
- **next-k** — 一次喂 k 个 token/行,返回全部 k 个位置的 logits(AR 是 k=1,投机 verify 是 k>1,引擎不关心)。
- **rollback** — trim 掉被拒的投机 token(attention 层;SSM 层回滚待 L2 解决)。

已验证:批量解码与单序列逐 token 一致(等长);不等长用右 padding + mask,并修复了 mlx-lm 让 SSM padding 未被屏蔽的 bug。批量输出与单序列的浮点级微小差异是批量推理固有性质,非 bug。

```python
from slipstream import Engine
import mlx.core as mx

eng = Engine("/path/to/model")
state, logits = eng.prefill([ids_a, ids_b])   # 批量 prefill
tok = eng.forward(state, mx.array([[x], [y]]))  # 批量 next-1
```

## 试用

```bash
python try_engine.py           # 交互:每行一个 prompt,空行运行,:q 退出
python smoke_engine.py         # 引擎回归测试(batch + next-k)
```
