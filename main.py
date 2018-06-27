import torch.optim as optim
import torch.nn as nn
import torch
import models
import data

epochs = 400
train_size = 3917
batch_size = 8
steps_per_epoch = round(train_size / batch_size)
display_step = 8

net = models.SqueezeNet(3, 8)

criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters())

data_iter = data.data_iterator(batch_size=batch_size)

for epoch in range(epochs):

    running_loss = 0
    running_acc = 0
    for step in range(steps_per_epoch):
        inputs, labels = next(data_iter)
        inputs = torch.Tensor(inputs)
        labels = torch.Tensor(labels)
        optimizer.zero_grad()

        outputs = net(inputs)
        # print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += (labels.max(1)[1] == outputs.max(1)[1]).float().sum() / outputs.size(0)

        if step % display_step == display_step - 1:
            loss = running_loss / display_step
            acc = running_acc / display_step
            print(f'epoch : {epoch} step : {step} loss : {loss} acc : {acc}')

            running_loss = 0
            running_acc = 0

torch.save(net, 'net.pt')
