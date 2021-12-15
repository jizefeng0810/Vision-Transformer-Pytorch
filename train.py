from trainer import Trainer
from utils.util import *
from utils.visualization import TensorboardWriter
from utils.config import get_train_config
from model.model import VisionTransformer
from utils.logger import LoggerRecord
from data_loader.data_loaders import *

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main():
    try:
        config = get_train_config()
    except:
        print("missing or invalid arguments")
        exit(0)

    logName = config.exp_name + '_' + config.dataset
    logR = LoggerRecord(logName + '.txt', level='info')
    loggerR = logR.logger

    # device
    device, device_ids = setup_device(config.n_gpu)

    # create model
    loggerR.info("create model")
    model = VisionTransformer(
        image_size=(config.image_size, config.image_size),
        patch_size=(config.patch_size, config.patch_size),
        emb_dim=config.emb_dim,
        mlp_dim=config.mlp_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        attn_dropout_rate=config.attn_dropout_rate,
        dropout_rate=config.dropout_rate)

    # load checkpoint
    if config.checkpoint_path:
        state_dict = load_checkpoint(config.checkpoint_path)
        if config.num_classes != state_dict['classifier.weight'].size(0):
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            loggerR.info("re-initialize fc layer")
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
        loggerR.info("Load pretrained weights from {}".format(config.checkpoint_path))

    # send model to device
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # create dataloader
    loggerR.info("create dataloaders")
    train_dataloader = eval("{}DataLoader".format(config.dataset))(
        data_dir=os.path.join(config.data_dir, config.dataset),
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        split='train')
    valid_dataloader = eval("{}DataLoader".format(config.dataset))(
        data_dir=os.path.join(config.data_dir, config.dataset),
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        split='val')

    # training criterion
    loggerR.info("create criterion and optimizer")
    criterion = torch.nn.CrossEntropyLoss()

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=config.lr,
        weight_decay=config.wd,
        momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config.lr,
        pct_start=config.warmup_steps / config.train_steps,
        total_steps=config.train_steps)

    # tensorboard
    writer = TensorboardWriter(config.summary_dir, config.tensorboard)

    # metric tracker
    metric_names = ['loss', 'acc1', 'acc5']

    trainer = Trainer(model, criterion, [metric_names, writer], optimizer, loggerR,
                      config=config,
                      device=device,
                      data_loader=train_dataloader,
                      valid_data_loader=valid_dataloader,
                      lr_scheduler=lr_scheduler)

    # start training
    loggerR.info("start training")
    best_acc = 0.0
    epochs = min(1, config.train_steps // len(train_dataloader))
    for epoch in range(1, epochs + 1):
        log = {'epoch': epoch}

        # train the model
        model.train()
        result = trainer._train_epoch(epoch)
        log.update(result)

        # validate the model
        model.eval()
        result = trainer._valid_epoch(epoch)
        log.update(**{'val_' + k: v for k, v in result.items()})

        # best acc
        best = False
        if log['val_acc1'] > best_acc:
            best_acc = log['val_acc1']
            best = True

        # save model
        save_model(config.checkpoint_dir, epoch, model, optimizer, lr_scheduler, device_ids, best)

        # print logged informations to the screen
        for key, value in log.items():
            loggerR.info('    {:15s}: {}'.format(str(key), value))


if __name__ == '__main__':
    main()
