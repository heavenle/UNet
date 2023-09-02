import os
import os.path as osp
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import logging
import datetime
import shutil
import torch
from glob import glob
from sklearn.model_selection import train_test_split
from config import opt
from utils.Registry import ARCH, LOSS, LOADER, OPTIMIZER, SCHEDULER
from utils.RunEpoch import IterRunningBase
import NetWorkRegistry


def main():
    # -----------------------------------------------------------------------------
    # 初始化超参数
    # -----------------------------------------------------------------------------
    config_path = "config/config.yml"
    config = opt.opts(config_path)
    if not os.path.exists(config['work_dir']):
        os.mkdir(config['work_dir'])
    shutil.copy(config_path, osp.join(config['work_dir'], "config.yml"))

    # -----------------------------------------------------------------------------
    # 创建日志记录器
    # -----------------------------------------------------------------------------
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    # 创建文件处理器
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logfile_{current_time}.txt"
    file_handler = logging.FileHandler(osp.join(config['work_dir'], log_filename))
    file_handler.setLevel(logging.DEBUG)
    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    # 打印超参数
    for k, v in config.items():
        logger.info("CONFIG - {:15}:{}".format(k, v))

    # -----------------------------------------------------------------------------
    #  加载网络
    # -----------------------------------------------------------------------------
    logger.info("CREAT MODEL - {}".format(config['arch']))
    model = ARCH.get(config['arch'])(**config["arch_params"])
    if config["load_from"]:
        model.load_state_dict(torch.load(config['load_from']))

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
    #  设置Loss(Focal loss需要自己设定alpha)
    # -----------------------------------------------------------------------------
    logger.info("CREAT LOSS - {}".format(config['loss']))
    loss_list = []
    for loss_name in config['loss']:
        loss_list.append(LOSS.get(loss_name)(**config['loss_params'][loss_name]))

    # -----------------------------------------------------------------------------
    #  设置optimizer
    # -----------------------------------------------------------------------------
    logger.info("CREAT OPTIMIZER - {}".format(config['optimizer']))
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = OPTIMIZER.get(config['optimizer'])(params, **config['optimizer_params'])

    # -----------------------------------------------------------------------------
    #  设置scheduler
    # -----------------------------------------------------------------------------
    logger.info("CREAT SCHEDULER - {}".format(config['scheduler']))
    scheduler = SCHEDULER.get(config["scheduler"])(optimizer, **config["scheduler_params"])

    # -----------------------------------------------------------------------------
    #  加载数据
    # -----------------------------------------------------------------------------
    logger.info("CREAT DATABASE - {}".format(config["data_type"]))
    img_ids = glob(osp.join(config['images_path'], '*' + config['img_ext']))
    img_ids = [osp.splitext(os.path.basename(p))[0] for p in img_ids]
    train_img_ids, val_img_ids = train_test_split(img_ids,
                                                  test_size=config['test_size'],
                                                  random_state=41)
    train_dataset = LOADER.get(config["data_type"])(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['images_path']),
        mask_dir=os.path.join(config['masks_path']),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        opt=config)

    val_dataset = LOADER.get(config["data_type"])(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['images_path']),
        mask_dir=os.path.join(config['masks_path']),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        opt=config)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # -----------------------------------------------------------------------------
    #  开始训练
    # -----------------------------------------------------------------------------
    logger.info("START TRAIN - {}".format(config["data_type"]))
    iterRun = IterRunningBase(config["epoch"], config, train_loader, val_loader, model, loss_list, optimizer, scheduler,
                              device,
                              logger)
    iterRun.run()


if __name__ == '__main__':
    main()
