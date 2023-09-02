import numpy as np
import torch
import torch.nn.functional as F

class Metrics:
    """Tracking mean metrics
    """

    def __init__(self, labels):
        """Creates an new `Metrics` instance.

        Args:
          labels: the labels for all classes.
        """

        self.labels = labels

        self.tn = [0]*self.labels
        self.fn = [0]*self.labels
        self.fp = [0]*self.labels
        self.tp = [0]*self.labels
        self.inf = [1e-6]*self.labels

    def add_(self, actual, predicted):
        """Adds an observation to the tracker.

        Args:
          actual: the ground truth labels.
          predicted: the predicted labels.
        """

        confusion = predicted.view(-1).float() / actual.view(-1).float()

        self.tn += torch.sum(torch.isnan(confusion)).item()
        self.fn += torch.sum(confusion == float("inf")).item()
        self.fp += torch.sum(confusion == 0).item()
        self.tp += torch.sum(confusion == 1).item()

    def add(self, actual, predicted):
        """Adds an observation to the tracker.

        Args:
          actual: the ground truth labels.
          predicted: the predicted labels.
        """
        mask_unique = np.unique(actual)
        # self.mask_unique = mask_unique
        # print(mask_unique)
        for mask_unique_label in mask_unique:
            mask1 = np.where(actual == mask_unique_label, 1, 0)
            pred1 = np.where(predicted == mask_unique_label, 1, 0)

        # confusion = predicted.view(-1).float() / actual.view(-1).float()
            confusion = torch.from_numpy(pred1.reshape(-1)/mask1.reshape(-1))

            self.tn[mask_unique_label] += torch.sum(torch.isnan(confusion)).item()
            self.fn[mask_unique_label] += torch.sum(confusion == float("inf")).item()
            self.fp[mask_unique_label] += torch.sum(confusion == 0).item()
            self.tp[mask_unique_label] += torch.sum(confusion == 1).item()

    # 获取得到每个epoch对应的精确度
    def get_precision_epoch(self ,actual, predicted):

        stn = [0]*self.labels
        sfn = [0]*self.labels
        sfp = [0]*self.labels
        stp = [0]*self.labels
        sinf = [1e-6]*self.labels

        mask_unique = np.unique(actual)
        # self.mask_unique = mask_unique
        # print(mask_unique)
        for mask_unique_label in mask_unique:
            mask1 = np.where(actual == mask_unique_label, 1, 0)
            pred1 = np.where(predicted == mask_unique_label, 1, 0)

            # confusion = predicted.view(-1).float() / actual.view(-1).float()
            confusion = torch.from_numpy(pred1.reshape(-1) / mask1.reshape(-1))

            stn[mask_unique_label] += torch.sum(torch.isnan(confusion)).item()
            sfn[mask_unique_label] += torch.sum(confusion == float("inf")).item()
            sfp[mask_unique_label] += torch.sum(confusion == 0).item()
            stp[mask_unique_label] += torch.sum(confusion == 1).item()

        precision = [0] * self.labels
        for i in range(self.labels):
            try:
                precision[i] = stp[i] / (stp[i] + sfp[i])
            except ZeroDivisionError:
                precision[i] = sinf[i]
        return precision

    def get_precision(self):
        val = [0]*self.labels
        for i in range(self.labels):
            try:
                val[i] = self.tp[i] / (self.tp[i] + self.fp[i])
            except ZeroDivisionError:
                val[i] = self.inf[i]
        return val

    def get_recall(self):
        val = [0]*self.labels
        for i in range(self.labels):
            try:
                val[i] = self.tp[i] / (self.tp[i] + self.fn[i])
            except ZeroDivisionError:
                val[i] = self.inf[i]
        return val

    def get_f_score(self):
        f1 = [0]*self.labels
        for i in range(self.labels):
            pr = 2 * (self.tp[i] / (self.tp[i] + self.fp[i] + self.inf[i])) * (self.tp[i] / (self.tp[i] + self.fn[i] + self.inf[i]))
            p_r = (self.tp[i] / (self.tp[i] + self.fp[i] + self.inf[i])) + (self.tp[i] / (self.tp[i] + self.fn[i] + self.inf[i]))
            f1[i] =  pr / (p_r + self.inf[0])

        return f1
    def get_oa(self):

        t_pn = self.tp + self.tn
        t_tpn = self.tp + self.tn + self.fp + self.fn + self.inf
        return t_pn / (t_tpn + self.inf)

    def get_miou(self):
        """Retrieves the mean Intersection over Union score.

        Returns:
          The mean Intersection over Union score for all observations seen so far.
        """
        return np.nanmean([self.tn / (self.tn + self.fn + self.fp), self.tp / (self.tp + self.fn + self.fp)])

    def get_fg_iou(self):
        """Retrieves the foreground Intersection over Union score.

        Returns:
          The foreground Intersection over Union score for all observations seen so far.
        """
        iou = [0]*self.labels
        for i in range(self.labels):

            try:
                iou[i] = self.tp[i] / (self.tp[i] + self.fn[i] + self.fp[i])
            except ZeroDivisionError:
                iou[i] = self.inf[i]

        return iou

    def get_mcc(self):
        """Retrieves the Matthew's Coefficient Correlation score.

        Returns:
          The Matthew's Coefficient Correlation score for all observations seen so far.
        """
        mcc = [0]*self.labels
        for i in self.mask_unique:
            if i==0:
                continue
            try:
                mcc[i] = (self.tp[i] * self.tn[i] - self.fp[i] * self.fn[i]) / math.sqrt(
                    (self.tp[i] + self.fp[i]) * (self.tp[i] + self.fn[i]) * (self.tn[i] + self.fp[i]) * (self.tn[i] + self.fn[i])
                )
            except ZeroDivisionError:
                mcc[i] = self.inf

        return mcc

def iou_score(output, target, flag=None):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = output.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
        if len(target.shape) == 3:
            target = target[:, np.newaxis, :, :]
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)

def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label.astype('int'), minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)



