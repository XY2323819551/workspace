# optimizer

1. 指数衰减学习率公式:
   lr = initial_lr * decay_rate^(global_step / decay_steps)

   其中:
   - lr: 当前学习率
   - initial_lr: 初始学习率
   - decay_rate: 衰减率 (通常小于1)
   - global_step: 当前训练步数
   - decay_steps: 衰减周期 (每经过多少步进行一次衰减)

   这个公式可以用以下PyTorch代码实现:

   ```python
   import torch.optim as optim

   initial_lr = 0.1
   decay_rate = 0.96
   decay_steps = 1000

   optimizer = optim.SGD(model.parameters(), lr=initial_lr)
   scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: decay_rate**(step / decay_steps))

   for epoch in range(num_epochs):
       for batch in dataloader:
           # 训练步骤
           optimizer.step()
           scheduler.step()
   ```

   这段代码使用`LambdaLR`调度器来实现指数衰减学习率。每次调用`scheduler.step()`时，学习率都会按照公式进行更新。这种方式使学习率随着训练步数的增加而逐渐减小，有助于在训练初期快速收敛，后期微调参数。

2. 学习率衰减的策略

   - 指数衰减
   - 分段常数衰减
   - 余弦退火衰减
   - 自适应学习率
   - 学习率预热
   - 学习率退火

3. 学习率预热

   在训练初期,使用较小的学习率,逐渐增加到初始学习率,有助于模型快速稳定地收敛。

4. 学习率退火

   在训练过程中,当验证集的性能不再提升时,降低学习率,有助于模型在新的起点上继续训练。
