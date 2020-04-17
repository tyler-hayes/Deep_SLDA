import os
import json
import numpy as np
import torchvision.transforms as transforms
import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset


def accuracy(output, target, topk=(1,), output_has_class_ids=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if not output_has_class_ids:
        output = torch.Tensor(output)
    else:
        output = torch.LongTensor(output)
    target = torch.LongTensor(target)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = output.shape[0]
        if not output_has_class_ids:
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
        else:
            pred = output[:, :maxk].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def filter_by_class(labels, min_class, max_class):
    return list(np.where(np.logical_and(labels >= min_class, labels < max_class))[0])


class ImagenetDataset(Dataset):
    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x, y = self.data[index]
        return x, y


def setup_imagenet_dataloader(dirname, training, idxs, batch_size=256, augment=False, shuffle=False,
                              sampler=None, batch_sampler=None, num_workers=8):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if training and augment:
        augmentation_transforms = []
        if augment:
            print('\nUsing standard data augmentation...')
            augmentation_transforms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        dataset = datasets.ImageFolder(
            dirname,
            transforms.Compose(augmentation_transforms))
    else:
        dataset = datasets.ImageFolder(dirname, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    if idxs is None:
        idxs = range(len(dataset))

    if batch_sampler is None and sampler is None:

        if shuffle:
            sampler = torch.utils.data.sampler.SubsetRandomSampler(idxs)
        else:
            sampler = IndexSampler(idxs)

    dataset = ImagenetDataset(dataset, idxs)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=True, batch_sampler=batch_sampler, sampler=sampler)
    return loader


class IndexSampler(torch.utils.data.Sampler):
    """Samples elements sequentially, always in the same order.
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_imagenet_loader(images_path, min_class, max_class, training, batch_size=256, shuffle=False):
    train_labels = np.load('./imagenet_files/imagenet_train_labels.npy')
    val_labels = np.load('./imagenet_files/imagenet_val_labels.npy')
    if training:
        curr_idx = filter_by_class(train_labels, min_class=min_class, max_class=max_class)
        images_path += '/train'
    else:
        curr_idx = filter_by_class(val_labels, min_class=min_class, max_class=max_class)
        images_path += '/val'

    loader = setup_imagenet_dataloader(images_path, training, curr_idx, batch_size=batch_size, shuffle=shuffle)
    return loader


def save_predictions(y_pred, min_class_trained, max_class_trained, save_path, suffix=''):
    name = 'preds_min_trained_' + str(min_class_trained) + '_max_trained_' + str(max_class_trained) + suffix
    torch.save(y_pred, save_path + '/' + name + '.pt')


def save_accuracies(accuracies, min_class_trained, max_class_trained, save_path, suffix=''):
    name = 'accuracies_min_trained_' + str(min_class_trained) + '_max_trained_' + str(
        max_class_trained) + suffix + '.json'
    json.dump(accuracies, open(os.path.join(save_path, name), 'w'))


def safe_load_dict(model, new_model_state):
    """
    Safe loading of previous ckpt file.
    """
    old_model_state = model.state_dict()

    c = 0
    for name, param in new_model_state.items():
        n = name.split('.')
        beg = n[0]
        end = n[1:]
        if beg == 'model':
            name = '.'.join(end)
        if name not in old_model_state:
            # print('%s not found in old model.' % name)
            continue
        c += 1
        if old_model_state[name].shape != param.shape:
            print('Shape mismatch...ignoring %s' % name)
            continue
        else:
            old_model_state[name].copy_(param)
    if c == 0:
        raise AssertionError('No previous ckpt names matched and the ckpt was not loaded properly.')
