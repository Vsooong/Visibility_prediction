import torch
import torchvision
from torchsummary import summary
import torch.nn as nn
from un_supervised_model import *
from util_project import args
import numpy as np
from video_process import VideoDataset, get_transform


class Visibility(nn.Module):
    def __init__(self, simclr, predict_dim=1):
        super(Visibility, self).__init__()
        self.simclr = simclr
        self.n_features = simclr.n_features
        self.fc = nn.Linear(self.n_features, predict_dim)

    def forward(self, x_i):
        h_i = self.fc(self.simclr(x_i))
        return h_i,


def get_pretrained_model(load_states=True):
    load_res = True
    if load_states == True:
        load_res = False
    encoder = get_resnet('resnet18', load_res)
    n_features = encoder.fc.in_features
    smodel = SimCLR(encoder, n_features)
    predict_model = Visibility(smodel).to(args.device)
    if load_states:
        predict_model.load_state_dict(torch.load(args.model_save1, map_location=args.device))
        print('load single fame model from:', args.model_save1)
    return predict_model


def train_one_epoch(model, optimizer, data_loader, criterion):
    model.train()
    loss_epoch = 0
    # num = 0
    for images, targets in data_loader:
        optimizer.zero_grad()
        images = images.to(args.device)
        targets = targets.to(args.device)
        pred = model(images)[0].squeeze()
        # print(pred)
        # print(targets)
        loss = criterion(targets, pred)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
        # num += len(targets)
        # if num % 5000 == 0:
        #     print(loss.item())
    return loss_epoch


def main(train_process=False):
    device = args.device
    dataset = VideoDataset(get_transform(train=False))
    dataset_test = VideoDataset(get_transform(train=False))

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.test_batch_size,
        shuffle=False, num_workers=args.workers,
    )
    model = get_pretrained_model(True)

    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.8)
    criterion = nn.MSELoss().to(device)
    best_loss = 9999999
    if train_process == True:
        for epoch in range(args.epochs):
            loss_epoch = train_one_epoch(model, optimizer, train_loader, criterion)
            lr_scheduler.step()
            if loss_epoch < best_loss:
                best_loss = loss_epoch
                torch.save(model.state_dict(), args.model_save1)
            # loss1, loss2 = evaluate(model, test_loader)
            # print (loss1, loss2)
            # torch.save(model.state_dict(), save_path)
            print(epoch, loss_epoch)
        print('training finish ')
    else:
        loss1, loss2 = evaluate(model, test_loader)
        print(loss1, loss2)


evaluateL1 = nn.L1Loss(reduction='sum')
evaluateL2 = nn.MSELoss(reduction='sum')


def evaluate(model, test_loader):
    model.eval()
    n_samples = 0
    total_loss1 = 0
    total_loss2 = 0

    for images, targets in test_loader:
        images = images.to(args.device)
        targets = targets.to(args.device)
        pred = model(images)[0].squeeze()
        total_loss1 += evaluateL1(targets, pred).data.item()
        total_loss2 += evaluateL2(targets, pred).data.item()
        # print(total_loss1,total_loss2)
        n_samples += len(targets)

    return total_loss1 / n_samples, np.sqrt(total_loss2 / n_samples)


if __name__ == '__main__':
    main(train_process=True)

    # # (time step,batch size, channel, height, length)
    # input = torch.rand(8, 3, 360, 640).to(args.device)
    #
    # model = get_pretrained_model().to(args.device)
    # nParams = sum([p.nelement() for p in model.parameters()])
    # print('number of parameters: %d' % nParams)
    # h_i = model(input)
    # # summary(model, (3, 360, 640))
