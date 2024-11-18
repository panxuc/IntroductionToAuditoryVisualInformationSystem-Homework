import os
import json
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from functools import partial


class_to_idx = {
  'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4,
  'bus':5, 'car':6, 'cat':7, 'chair':8, 'cow':9, 'diningtable':10,
  'dog':11, 'horse':12, 'motorbike':13, 'person':14, 'pottedplant':15,
  'sheep':16, 'sofa':17, 'train':18, 'tvmonitor':19
}
idx_to_class = {i:c for c, i in class_to_idx.items()}


def get_pascal_voc2007_data(image_root, split='train'):
    """
    Use torchvision.datasets
    https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.VOCDetection
    """

    train_dataset = datasets.VOCDetection(image_root, year='2007', image_set=split,
                                    download=False)

    return train_dataset


def pascal_voc2007_loader(dataset, batch_size, num_workers=0, shuffle=False, proposal_path=None):
    """
    Data loader for Pascal VOC 2007.
    https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    """
    collate_fn = partial(voc_collate_fn, proposal_path=proposal_path)
    train_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle, pin_memory=True,
                            num_workers=num_workers,
                            collate_fn=collate_fn)
    return train_loader


def voc_collate_fn(batch_lst, reshape_size=224, proposal_path=None):
    preprocess = transforms.Compose([
      transforms.Resize((reshape_size, reshape_size)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])
    
    batch_size = len(batch_lst)
    
    img_batch = torch.zeros(batch_size, 3, reshape_size, reshape_size)
    
    box_list = []
    box_batch_idx = []
    w_list = []
    h_list = []
    img_id_list = []
    proposal_list = []
    proposal_batch_idx = []
    
    for i in range(batch_size):
      img, ann = batch_lst[i]
      w_list.append(img.size[0]) # image width
      h_list.append(img.size[1]) # image height
      img_id_list.append(ann['annotation']['filename'])
      img_batch[i] = preprocess(img)
      all_bbox = ann['annotation']['object']
      if type(all_bbox) == dict: # inconsistency in the annotation file
        all_bbox = [all_bbox]
      for bbox_idx, one_bbox in enumerate(all_bbox):
        bbox = one_bbox['bndbox']
        obj_cls = one_bbox['name']
        box_list.append(torch.Tensor([float(bbox['xmin']), float(bbox['ymin']),
          float(bbox['xmax']), float(bbox['ymax']), class_to_idx[obj_cls]]))
        box_batch_idx.append(i)
      if proposal_path is not None:
        proposal_fn = ann['annotation']['filename'].replace('.jpg', '.json')
        with open(os.path.join(proposal_path, proposal_fn), 'r') as f:
          proposal = json.load(f)
        for p in proposal:
          proposal_list.append([p['x_min'], p['y_min'], p['x_max'], p['y_max']])
          proposal_batch_idx.append(i)
    
    h_batch = torch.tensor(h_list)
    w_batch = torch.tensor(w_list)
    box_batch = torch.stack(box_list)
    box_batch_ids = torch.tensor(box_batch_idx, dtype=torch.long)
    proposals = torch.tensor(proposal_list, dtype=box_batch.dtype)
    proposal_batch_ids = torch.tensor(proposal_batch_idx, dtype=torch.long)
    assert len(box_batch) == len(box_batch_ids)
    assert len(proposals) == len(proposal_batch_ids)

    return img_batch, box_batch, box_batch_ids, proposals, proposal_batch_ids, w_batch, h_batch, img_id_list