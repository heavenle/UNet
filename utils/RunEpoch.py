import os
import os.path as osp
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import torch
from tqdm import tqdm
from utils.metrics import Metrics, Evaluator, iou_score
from utils.AverageMeter import AverageMeter


class IterRunningBase():
    def __init__(self, epoch, config, train_loader, val_loader, model, loss, optimizer, scheduler, device, logger):
        self.epoch = epoch
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.logger = logger
        self.scheduler = scheduler
        self.device = device
        self.best_iou = 0

    def run(self):
        for epoch in range(0, self.config['epoch']):
            self.logger.info(f"EPOCH is {epoch+1}")
            self.train()
            self.val(epoch)
            self.scheduler.step()

    def train(self):

        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter()}
        for i in self.config['loss']:
            avg_meters[i] = AverageMeter()

        evaluator = Evaluator(self.config['num_classes'])
        train_metrics = Metrics(self.config['num_classes'])

        self.model.train()
        pbar = tqdm(total=len(self.train_loader))
        for img_meta in self.train_loader:
            input = img_meta["img"].to(self.device).to(torch.float32)
            target = img_meta["mask"].to(self.device).to(torch.int64)
            out = self.model(input)
            loss_single = torch.zeros(len(self.config['loss'])).to(self.device)
            for output in out:
                for loss_ids in range(len(self.config['loss'])):
                    loss_single[loss_ids] += self.loss[loss_ids](output, target)
            loss_single = loss_single / len(out)
            loss_all = loss_single.sum()
            iou = iou_score(out[-1], target)

            loss_all.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # 计算指标。
            out[0] = out[0].cpu().detach().numpy()
            target = target.cpu().detach().numpy()
            for batch_num in range(0, out[0].shape[0]):
                pred = np.argmax(out[0][batch_num].transpose(1, 2, 0), 2)
                gt = target[batch_num]
                evaluator.add_batch(gt, pred)
                pred = np.rint(pred).astype(int)
                gt = gt.astype(int)
                train_metrics.add(gt, pred)

            # # 更新指标
            avg_meters['loss'].update(loss_all.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            for ids, name in enumerate(self.config['loss']):
                avg_meters[name].update(loss_single[ids].item(), input.size(0))

            cout_loss = ''
            for ids, name in enumerate(self.config['loss']):
                cout_loss += f" {name} : {loss_single[ids].item():4f}"
            self.logger.info(
                f"[TRAIN]loss: {avg_meters['loss'].avg:4f}, iou_score: {avg_meters['iou'].avg:4f}" + cout_loss)
            pbar.update(1)
        pbar.close()

        train_epoch_pr = train_metrics.get_precision()
        train_epoch_re = train_metrics.get_recall()
        train_epoch_fs = train_metrics.get_f_score()
        train_epoch_iou = train_metrics.get_fg_iou()
        train_epoch_pr_mean = sum(train_epoch_pr) / len(train_epoch_pr)
        train_epoch_re_mean = sum(train_epoch_re) / len(train_epoch_re)
        train_epoch_fs_mean = sum(train_epoch_fs) / len(train_epoch_fs)
        train_epoch_iou_mean = sum(train_epoch_iou) / len(train_epoch_iou)

        Acc = evaluator.Pixel_Accuracy()  # score
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

        self.logger.info(f'[TRAIN]acc={Acc:.4f}, acc_class={Acc_class:.4f}, miou={mIoU:.4f}, fwiou={FWIoU:.4f}')
        self.logger.info(
            f'[TRAIN]pre={train_epoch_pr_mean:.4f}, fs={train_epoch_fs_mean:.4f}, recall={train_epoch_re_mean:.4f}, miou={train_epoch_iou_mean:.4f}')

    def val(self, epoch):
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter()}
        for i in self.config['loss']:
            avg_meters[i] = AverageMeter()

        evaluator = Evaluator(self.config['num_classes'])
        val_metrics = Metrics(self.config['num_classes'])

        # switch to evaluate mode
        self.model.eval()
        # -- 进度条
        pbar = tqdm(total=len(self.val_loader))

        with torch.no_grad():
            for img_meta in self.val_loader:
                input = img_meta["img"].to(self.device).to(torch.float32)
                target = img_meta["mask"].to(self.device).to(torch.int64)
                out = self.model(input)
                loss_single = torch.zeros(len(self.config['loss'])).to(self.device)
                for output in out:
                    for loss_ids in range(len(self.config['loss'])):
                        loss_single[loss_ids] += self.loss[loss_ids](output, target)
                loss_single = loss_single / len(out)
                loss_all = loss_single.sum()
                iou = iou_score(out[-1], target)

                # 计算指标。
                out[0] = out[0].cpu().detach().numpy()
                target = target.cpu().detach().numpy()
                for batch_num in range(0, out[0].shape[0]):
                    pred = np.argmax(out[0][batch_num].transpose(1, 2, 0), 2)
                    gt = target[batch_num]
                    evaluator.add_batch(gt, pred)
                    pred = np.rint(pred).astype(int)
                    gt = gt.astype(int)
                    val_metrics.add(gt, pred)

                # 更新指标
                avg_meters['loss'].update(loss_all.item(), input.size(0))
                avg_meters['iou'].update(iou, input.size(0))
                for ids, name in enumerate(self.config['loss']):
                    avg_meters[name].update(loss_single[ids].item(), input.size(0))

                cout_loss = ''
                for ids, name in enumerate(self.config['loss']):
                    cout_loss += f" {name} : {loss_single[ids].item():4f}"
                self.logger.info(
                    f"[VAL]loss: {avg_meters['loss'].avg:4f}, iou_score: {avg_meters['iou'].avg:4f}" + cout_loss)
                pbar.update(1)
            pbar.close()

            val_epoch_pr = val_metrics.get_precision()
            val_epoch_re = val_metrics.get_recall()
            val_epoch_fs = val_metrics.get_f_score()
            val_epoch_iou = val_metrics.get_fg_iou()
            val_epoch_pr_mean = sum(val_epoch_pr) / len(val_epoch_pr)
            val_epoch_re_mean = sum(val_epoch_re) / len(val_epoch_re)
            val_epoch_fs_mean = sum(val_epoch_fs) / len(val_epoch_fs)
            val_epoch_iou_mean = sum(val_epoch_iou) / len(val_epoch_iou)

            Acc = evaluator.Pixel_Accuracy()
            Acc_class = evaluator.Pixel_Accuracy_Class()
            mIoU = evaluator.Mean_Intersection_over_Union()
            FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

            self.logger.info(f'[VAL]acc={Acc:.4f}, acc_class={Acc_class:.4f}, miou={mIoU:.4f}, fwiou={FWIoU:.4f}')
            self.logger.info(
                f'[VAL]pre={val_epoch_pr_mean:.4f}, fs={val_epoch_fs_mean:.4f}, recall={val_epoch_re_mean:.4f}, miou={val_epoch_iou_mean:.4f}')

            if self.best_iou < mIoU:
                torch.save(self.model.state_dict(), osp.join(self.config['work_dir'], f"model_best.pth"))
            if (epoch + 1) % self.config['checkpoint_iter'] == 0:
                torch.save(self.model.state_dict(), osp.join(self.config['work_dir'], f"model_{epoch+1}.pth"))