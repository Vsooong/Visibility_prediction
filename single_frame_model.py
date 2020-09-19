import torch
import torchvision
from torchsummary import summary
import torch.nn as nn
from un_supervised_model import *
from util_project import args
import torch.nn.functional as F
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
    encoder = get_resnet('resnet18', True).to(args.device)
    n_features = encoder.fc.in_features
    smodel = SimCLR(encoder, n_features).to(args.device)
    predict_model = Visibility(smodel)
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


def evaluate(model, data_loader_test, device):
    model.eval()


def main():
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
        dataset_test, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
    )
    model = get_pretrained_model().to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.8)
    criterion = nn.MSELoss().to(device)
    best_loss = 9999999

    for epoch in range(args.epochs):
        loss_epoch = train_one_epoch(model, optimizer, train_loader, criterion)
        lr_scheduler.step()
        if loss_epoch < best_loss:
            best_loss = loss_epoch
            torch.save(model.state_dict(), args.model_save1)
        # evaluate on the test dataset
        evaluate(model, test_loader, device=device)
        # torch.save(model.state_dict(), save_path)
        print(epoch, loss_epoch)


if __name__ == '__main__':
    train()
    # args = get_config()
    # # (time step,batch size, channel, height, length)
    # input = torch.rand(8, 3, 360, 640).to(args.device)
    #
    # model = get_pretrained_model().to(args.device)
    # nParams = sum([p.nelement() for p in model.parameters()])
    # print('number of parameters: %d' % nParams)
    # h_i = model(input)
    # # summary(model, (3, 360, 640))
