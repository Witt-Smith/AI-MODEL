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

默认数据集是 Hugging Face `silver/lccc` 的 `base` 配置，读取字段为 `dialog`。
如果数据集结构不同，需要明确指定，例如：

```bash
python train_cloud.py --dataset-name owner/dataset --dataset-config default --dialog-field messages
```

也可以把 `--dataset-name` 指向本地 JSON。训练记录只要求 `question` 和
`answer`；`intent`、`record_id`、`answer_id` 是可选元数据，不会自动变成模型的
意图路由能力：

```json
{
  "train": [
    {"question": "你好", "answer": "你好，有什么可以帮你？"}
  ],
  "validation": [],
  "test": [],
  "unknown": ["一个训练域外的问题"]
}
```

持续训练：

```bash
python train_cloud.py --output-dir /persistent/ai-model --continuous
```

`/persistent/ai-model` 必须替换为云平台的持久化目录。训练会每个 epoch 更新
`checkpoints/last.ckpt`；同一条命令重新启动时会自动恢复模型、优化器、epoch 和
global step。首次运行还会在 `artifacts/` 保存与检查点绑定的词表和 Word2Vec；
后续启动直接加载，不会重新训练 Word2Vec。`--no-resume` 只重新初始化网络权重，
仍会复用同一输出目录里的词表和 Word2Vec；更换数据集时必须使用新的输出目录，
否则词表与数据会失去对应关系。

可用 `--max-time 00:12:00:00` 限制单次任务最多运行 12 小时。有限训练中的
`--max-epochs` 是总 epoch 目标，不是每次重启后额外增加的 epoch 数。

## 本地聊天

```bash
python chat.py --output-dir /persistent/ai-model
```

输入 `exit`、`quit` 或 `退出` 结束聊天。训练任务不会进入交互输入。
可用 `--device cpu` 固定运行设备、`--max-new-tokens 50` 控制最长生成长度，
以及 `--typing-delay 0` 关闭逐字输出延迟。

## 离线检查

```bash
python -m unittest discover -s tests -v
```

这组测试不会下载远程数据，覆盖本地/远程数据字段、固定特殊 token、EOS 停止、
单次编码问题和 `last.ckpt` 恢复优先级。根目录的 `test.py` 是个人试验文件，
不会被这条命令收集。

训练指标位于 `logs/train/version_*/metrics.csv`，可用 `Log(metrics_path)` 的
`show_dataframe()` 查看，或用 `show_plot()` 绘制 `normal_loss` 和 `lr`。
