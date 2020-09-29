import yaml
import argparse
import torch
import os
import random
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas
import matplotlib.pyplot as plt

def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file, 'rb') as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg


def get_config():
    parser = argparse.ArgumentParser(description="visibility")
    config = yaml_config_hook("config_vis.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes
    return args


args = get_config()

# labels=np.asarray(labels)
# preds=np.asarray(preds)+labels
# x = list(np.arange(50, max_value+1, 50))
# loss_dict = {}
# metric_dict = {}
# for i in x:
#     loss_dict[i] = []
#
# for index in range(len(labels)):
#     lb = labels[index]
#     pd = preds[index]
#     inv = int(lb / 50) * 50
#     loss_dict[inv].append([lb, pd])
# for k, v in loss_dict.items():
#     if len(v) > 0:
#         lbs, pds = zip(*v)
#         mae = mean_absolute_error(lbs, pds)
#         rmse = np.sqrt(mean_squared_error(lbs, pds))
#         r = np.corrcoef(lbs, pds)[0][1]
#         metric_dict[k] = [mae, rmse, r]
# mae_list = []
# mse_list = []
# r_list = []
# for i in x:
#     if i in metric_dict.keys():
#         mae_list.append(metric_dict[i][0])
#         mse_list.append(metric_dict[i][1])
#         r_list.append(metric_dict[i][2])
#     else:
#         mae_list.append(0)
#         mse_list.append(0)
#         r_list.append(0)
# d = {'MOR': x, 'MAE': mae_list, 'RMSE': mse_list, 'R': r_list}
# data = pandas.DataFrame(data=d)
# import seaborn as sns
#
# sns.set_style("darkgrid")
# # sns.lineplot(data=data, x='MOR', y='R')
# sns.lineplot(x='MOR', y='value', hue='variable', data=pandas.melt(data, ['MOR']))
#
# plt.show()

# special_transform = transforms.Compose([
#     transforms.Resize(size=(360,640)),
#     transforms.RandomResizedCrop(size=(360,640)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ToTensor(),
# ])
# special_transform = transforms.Compose([
#     transforms.CenterCrop(size=(260, 520)),
#     transforms.Resize(size=(360,640)),
#     # transforms.RandomResizedCrop(size=(360,640)),
#     # transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ToTensor(),
# ])
# pil_transform = transforms.ToPILImage()
#
# # dic_path='F:/data/visibility/output/'
# dic_path = '/home/dl/data/visibility/test'
# out_path = '/home/dl/data/visibility/test1'
# for name in os.listdir(dic_path):
#     file_path = os.path.join(dic_path, name)
#     img = Image.open(file_path).convert('RGB')
#     image = special_transform(img)
#     save_img = pil_transform(image)
#     save_img.save(os.path.join(out_path, name))
