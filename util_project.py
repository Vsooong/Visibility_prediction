import os
import yaml
import argparse
import torch
from torchvision import transforms
from PIL import Image


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file,'rb') as f:
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

    args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes
    return args

args=get_config()
import os
dic_path='F:/data/visibility/output/'
out_path='F:/data/visibility/train/'

file_path=os.path.join(dic_path,'frame_000001_1100.0.jpg')
img=Image.open(file_path).convert('RGB')
special_transform = transforms.Compose([
    transforms.CenterCrop(size=(260,520)),
    # transforms.Resize(size=(360,640)),
    # transforms.RandomResizedCrop(size=(360,640)),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])
# special_transform = transforms.Compose([
#     transforms.Resize(size=(360,640)),
#     transforms.RandomResizedCrop(size=(360,640)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ToTensor(),
# ])
pil_transform= transforms.ToPILImage()
image=special_transform(img)
print(image.size())
save_img=pil_transform(image)
save_img.save(os.path.join(out_path,'1.jpg'))