import numpy as np
import torch
from utils.util import inf_loop
from model.metric import top_k_accuracy, MetricTracker


class Trainer():
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, logger, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__()
        self.config = config
        self.model = model
        self.criterion = criterion
        self.metric_ftns, self.writer = metric_ftns
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            acc1, acc5 = top_k_accuracy(output, target, topk=(1, 5))

            self.train_metrics.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('acc1', acc1.item())
            self.train_metrics.update('acc5', acc5.item())

            if batch_idx % 1000 == 0:
                self.logger.info("Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f} Acc@1: {:.2f}, Acc@5: {:.2f}"
                      .format(epoch, batch_idx, len(self.data_loader), loss.item(), acc1.item(), acc5.item()))
        return self.train_metrics.result()

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        losses = []
        acc1s = []
        acc5s = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                acc1, acc5 = top_k_accuracy(output, target, topk=(1, 5))
                losses.append(loss.item())
                acc1s.append(acc1.item())
                acc5s.append(acc5.item())

        loss = np.mean(losses)
        acc1 = np.mean(acc1s)
        acc5 = np.mean(acc5s)
        self.valid_metrics.writer.set_step(epoch, 'valid')
        self.valid_metrics.update('loss', loss)
        self.valid_metrics.update('acc1', acc1)
        self.valid_metrics.update('acc5', acc5)
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
