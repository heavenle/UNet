import yaml


def opts(config_path):
    with open(config_path, 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['num_classes'] = config['labels'].split(',').__len__()
    config['loss'] = config['loss'].split(',')
    return config


if __name__ == '__main__':
    config_path = 'config.yml'
    config = opts(config_path)
    print(config)
