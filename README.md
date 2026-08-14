# AI-MODEL

## 云端训练

安装依赖：

```bash
python -m pip install -r requirements.txt
```

有限训练：

```bash
python train_cloud.py --output-dir /persistent/ai-model --max-epochs 1000
```

持续训练：

```bash
python train_cloud.py --output-dir /persistent/ai-model --continuous
```

`/persistent/ai-model` 必须替换为云平台的持久化目录。训练会每个 epoch 更新
`checkpoints/last.ckpt`；同一条命令重新启动时会自动恢复模型、优化器、epoch 和
global step。首次运行还会在 `artifacts/` 保存与检查点绑定的词表和 Word2Vec；
后续启动直接加载，不会重新训练 Word2Vec。使用 `--no-resume` 才会从头开始。

可用 `--max-time 00:12:00:00` 限制单次任务最多运行 12 小时。有限训练中的
`--max-epochs` 是总 epoch 目标，不是每次重启后额外增加的 epoch 数。

## 本地聊天

```bash
python chat.py --output-dir /persistent/ai-model
```

输入 `exit`、`quit` 或 `退出` 结束聊天。训练任务不会进入交互输入。
