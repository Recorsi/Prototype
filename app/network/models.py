import os
import torch
import torch.nn as nn
import torchvision
from network.xception import xception
import urllib.request
import ssl


def download_pretrained_weights(url, dest_path):
    print(f"Downloading pretrained weights from {url} to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    context = ssl._create_unverified_context()  # Disable SSL verification
    with urllib.request.urlopen(url, context=context) as response, open(dest_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
        
    print("Download complete.")


def return_pytorch04_xception(pretrained=True):
    model = xception(pretrained=False)
    if pretrained:
        weights_url = 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth'
        dest_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/xception-b5690688.pth")
        
        if not os.path.exists(dest_path):
            download_pretrained_weights(weights_url, dest_path)
            
        model.fc = model.last_linear
        del model.last_linear
        state_dict = torch.load(dest_path)
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        model.load_state_dict(state_dict)
        model.last_linear = model.fc
        del model.fc
        
    return model


class TransferModel(nn.Module):
    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        if modelchoice == 'xception':
            self.model = return_pytorch04_xception()
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        elif modelchoice == 'resnet50' or modelchoice == 'resnet18':
            if modelchoice == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=True)
            if modelchoice == 'resnet18':
                self.model = torchvision.models.resnet18(pretrained=True)
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
            else:
                self.model.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def set_trainable_up_to(self, boolean, layername="Conv2d_4a_3x3"):
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if boolean:
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise Exception('Layer not found, cant finetune!'.format(layername))
        else:
            if self.modelchoice == 'xception':
                for param in self.model.last_linear.parameters():
                    param.requires_grad = True
            else:
                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


def model_selection(modelname, num_out_classes, dropout=None):
    if modelname == 'xception':
        return TransferModel(modelchoice='xception', num_out_classes=num_out_classes), 299, True, ['image'], None
    elif modelname == 'resnet18':
        return TransferModel(modelchoice='resnet18', dropout=dropout, num_out_classes=num_out_classes), 224, True, ['image'], None
    else:
        raise NotImplementedError(modelname)