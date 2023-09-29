import os
import os.path as osp
import logging
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from glob import glob
import numpy as np
import torch
import cv2
from config import opt
from utils.Registry import ARCH
import NetWorkRegistry
from utils.metrics import Metrics, Evaluator, iou_score
from utils.AverageMeter import AverageMeter

def main():
    # -----------------------------------------------------------------------------
    # 初始化超参数
    # -----------------------------------------------------------------------------
    config_path = "config/config.yml"
    config = opt.opts(config_path)
    if not os.path.exists(config['work_dir']):
        os.mkdir(config['work_dir'])

    # -----------------------------------------------------------------------------
    # 创建日志记录器
    # -----------------------------------------------------------------------------
    logger = logging.getLogger('test_logger')
    logger.setLevel(logging.DEBUG)
    console_Handler = logging.StreamHandler()
    console_Handler.setLevel(logging.DEBUG)
    logger.addHandler(console_Handler)
    for k, v in config.items():
        logger.info("CONFIG - {:15}:{}".format(k, v))

    # -----------------------------------------------------------------------------
    #  加载网络
    # -----------------------------------------------------------------------------
    logger.info("CREAT MODEL - {}".format(config['arch']))
    model = ARCH.get(config['arch'])(**config["arch_params"])
    if config["load_from_to_test"]:
        model.load_state_dict(torch.load(config['load_from_to_test']))

    # -----------------------------------------------------------------------------
    #  设置GPU
    # -----------------------------------------------------------------------------
    device = 'cpu'
    if config["cuda"]:
        logger.info(f'SET GPU ids - cuda:{config["gpu_ids"]}')
        device = torch.device('cuda:' + str(config['gpu_ids']))
        model.to(device)
    else:
        model.to(device)

    # -----------------------------------------------------------------------------
    #  加载测试数据路径
    # -----------------------------------------------------------------------------
    test_path = config["test_with_index_path"]
    img_ids = glob(os.path.join(test_path, 'images', "*" + config['test_img_ext']))
    mask_ids = glob(os.path.join(test_path, 'masks', "*" + config['test_img_ext']))
    zipped = zip(img_ids, mask_ids)
    # -----------------------------------------------------------------------------
    #  初始化计算指标
    # -----------------------------------------------------------------------------
    evaluator = Evaluator(config['num_classes'])
    val_metrics = Metrics(config['num_classes'])
    # -----------------------------------------------------------------------------
    #  开始测试
    # -----------------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        for img_path, mask_path in zipped:
            logger.info(f"DEAL IMG IS - {img_path}")
            img = cv2.imread(img_path)
            img = cv2.resize(img, (config['resize_w'], config['resize_h']))
            img = img.astype(np.uint8) / 255.
            img = img.transpose(2, 0, 1)
            img = torch.tensor(img[np.newaxis, ::])
            if config['cuda']:
                input = img.to(device).to(torch.float32)

            out = model(input)
            output = out[0].detach().cpu().numpy()
            output = np.argmax(output[0], axis=0)
            cv2.imwrite(osp.join(config['work_dir'], osp.basename(img_path)), output.astype(np.uint8))

            # 计算指标。
            out[0] = out[0].cpu().detach().numpy()
            gt = cv2.imread(mask_path, 0)
            pred = np.argmax(out[0][0].transpose(1, 2, 0), 2)
            evaluator.add_batch(gt, pred)
            pred = np.rint(pred).astype(int)
            gt = gt.astype(int)
            val_metrics.add(gt, pred)

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

        logger.info(f'[TEST]acc={Acc:.4f}, acc_class={Acc_class:.4f}, miou={mIoU:.4f}, fwiou={FWIoU:.4f}')
        logger.info(f'[TEST]pre={val_epoch_pr_mean:.4f}, fs={val_epoch_fs_mean:.4f}, recall={val_epoch_re_mean:.4f}, miou={val_epoch_iou_mean:.4f}')


if __name__ == '__main__':
    main()
