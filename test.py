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
    test_path = config["test_path"]
    img_ids = glob(os.path.join(test_path, "*" + config['test_img_ext']))

    # -----------------------------------------------------------------------------
    #  开始测试
    # -----------------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        for img_path in img_ids:
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


if __name__ == '__main__':
    main()
