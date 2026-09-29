import torch
import torch.nn.functional as F
from statistics import mean
from medpy.metric.binary import hd95

class Eval:
    def __init__(self, config, classes=1):
        self.iou_list = []
        self.dice_list = []
        self.hd95_list = []
        self.num_classes=classes
        self.config = config
        self.class_dice = None
        self.class_iou = None

    def reset_eval(self, clear_class=True):
        self.iou_list.clear()
        self.dice_list.clear()
        self.hd95_list.clear()
        # DICE/IoU por clase: se reinician al empezar cada época, no al calcular la media
        # (class_metric() se llama después de mean_metric() y los necesita)
        if clear_class:
            self.class_dice = None
            self.class_iou = None

    def compute_metrics(self, pred, gt, print_val=False):
        pred = pred.detach()
        gt = gt.detach()

        if torch.cuda.is_available():
            pred = pred.cuda()
            gt = gt.cuda()

        if self.config.data_loader.num_classes > 1:
            iou, dice = self.multiclass_iou_and_dice(pred, gt)
            
            self.iou_list.append(iou.item())
            self.dice_list.append(dice.item())

            hd95_val = self.multiclass_hd95(pred, gt)
            if hd95_val is not None:
                self.hd95_list.append(hd95_val)
            
        else:
            pred = pred.to(torch.uint8)
            gt = gt.to(torch.uint8)

            pred = pred[None, ...] if pred.ndim == 3 else pred
            gt = gt[None, ...] if gt.ndim == 3 else gt

            pred_count = torch.sum(pred == 1, dim=list(range(1, pred.ndim)))
            gt_count = torch.sum(gt == 1, dim=list(range(1, gt.ndim)))
            iou, dice = self.iou_and_dice(pred, gt)
            
            self.iou_list.append(iou)
            self.dice_list.append(dice)

            hd95_val = 0
            if torch.sum(pred_count) > 0 and torch.sum(gt_count) > 0:
                hd95_val = self.compute_hd95(pred.squeeze(), gt.squeeze())
                self.hd95_list.append(hd95_val)

        if print_val:
            print(f'iou={iou}, dice={dice}, h95={hd95_val}')
        
    def iou_and_dice(self, pred, gt):
        eps = 1e-6
        intersection = (pred & gt).sum()
        dice_union = pred.sum() + gt.sum()
        iou_union = dice_union - intersection

        iou = (intersection + eps) / (iou_union + eps)
        dice = (2 * intersection + eps) / (dice_union + eps)

        return iou.item(), dice.item()

    def compute_hd95(self, pred, gt):
        return hd95(pred.cpu().numpy(), gt.cpu().numpy())
    
    def multiclass_iou_and_dice(self, pred, gt):
        eps = 1e-6
        target_one_hot = F.one_hot(gt.squeeze(1).long(), num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        # predicción dura (clase más probable), igual que el umbral 0.5 del caso binario
        pred = F.one_hot(pred.argmax(dim=1), num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()

        intersection = torch.sum(pred * target_one_hot, dim=(0, 2, 3, 4))
        cardinality = torch.sum(pred + target_one_hot, dim=(0, 2, 3, 4))

        dice_scores = (2.0 * intersection + eps) / (cardinality + eps)
        average_dice = torch.mean(dice_scores)
        
        # IoU calculation
        union = cardinality - intersection
        iou_scores = (intersection + eps) / (union + eps)
        average_iou = torch.mean(iou_scores)

        if self.class_dice is None:
            self.class_dice = dice_scores.unsqueeze(0)
        else:
            self.class_dice = torch.cat((self.class_dice, dice_scores.unsqueeze(0)), 0)

        if self.class_iou is None:
            self.class_iou = iou_scores.unsqueeze(0)
        else:
            self.class_iou = torch.cat((self.class_iou, iou_scores.unsqueeze(0)), 0)
        
        return average_iou, average_dice

    def multiclass_hd95(self, pred, gt):
        # HD95 real (distancia entre superficies, MedPy) por clase, sin contar el fondo (clase 0).
        # Solo se evalúan las clases presentes a la vez en predicción y GT, igual que en el caso binario.
        # Devuelve la media sobre clases y volúmenes, o None si no hay ninguna clase evaluable.
        pred_labels = pred.argmax(dim=1).cpu().numpy()
        gt_labels = gt.squeeze(1).long().cpu().numpy()

        values = []
        for b in range(pred_labels.shape[0]):
            for c in range(1, self.num_classes):
                pred_c = pred_labels[b] == c
                gt_c = gt_labels[b] == c
                if pred_c.any() and gt_c.any():
                    values.append(hd95(pred_c, gt_c))

        return mean(values) if values else None

    def mean_metric(self):
        iou = 0 if len(self.iou_list) == 0 else mean(self.iou_list)
        dice = 0 if len(self.dice_list) == 0 else mean(self.dice_list)
        hd95_val = 0 if len(self.hd95_list) == 0 else mean(self.hd95_list)

        self.reset_eval(clear_class=False)
        return iou, dice, hd95_val

    def class_metric(self):
        if self.class_dice is None:
            print('DICE/IOU por clase: sin datos en esta época')
            return
        print(f'DICE: {torch.mean(self.class_dice, dim=0)}')
        print(f'IOU: {torch.mean(self.class_iou, dim=0)}')
