import torch
from tqdm import tqdm_notebook as tqdm
from cnnlib import Utility
import logging as log

log.basicConfig(filename='model-builder.log', level=log.DEBUG, format='%(asctime)s %(message)s')


class ModelBuilder:
    """
    Builds the model with for the given parameters. Use fit method with appropriate number of epoch after initializing the object.
    """

    def __init__(self, model, data, loss_fn, optimizer, checkpoint=None, model_path=None, scheduler=None,
                 metric_fn=None,
                 train_pred_persister=None, test_pred_persister=None, device=Utility.getDevice()):
        self.model = model
        self.lossFn = loss_fn
        self.optimizer = optimizer
        self.data = data
        self.checkpoint = checkpoint
        self.model_path = model_path
        self.trainer = ModelTrainer(model=model, loss_fn=loss_fn, optimizer=optimizer,
                                    scheduler=optimizer if scheduler is None else scheduler,
                                    metric_fn=metric_fn, persister=train_pred_persister)
        self.tester = ModelTester(model=model, loss_fn=loss_fn, persister=test_pred_persister, metric_fn=metric_fn,
                                  device=device)

    def fit(self, epoch):

        log.info(f"Building model for {epoch}")

        train_metrices = []
        train_losses = []
        test_metrices = []
        test_losses = []
        learning_rates = []
        for e in range(0, epoch):
            print(f'\n\nEpoch: {e + 1}')
            log.info(f"Epoch {e}")

            learning_rate = self.optimizer.param_groups[0]['lr']
            learning_rates.append(learning_rate)

            log.info(f"Starting the training for epoch {e}")
            # Train
            train_result = self.trainer.train_one_epoch(loader=self.data.train, epoch_num=e)
            train_losses.append(train_result.loss.item())
            print(f'Train Loss: {train_result.loss.item()}, Learning Rate: {learning_rate}')

            if train_result.metric:
                train_metrices.append(train_result.metric)
                print(f'Train metrics: {train_result.metric}')

            log.info(f"End of the training for epoch {e}")

            # Test
            if (self.checkpoint is not None and e % self.checkpoint == 0) or (e == epoch - 1):

                log.info(f"Starting the testing for epoch {e}")
                print(f"Predicting on test set.")
                test_result = self.tester.test(loader=self.data.test, epoch_num=e)
                test_losses.append(test_result.loss.item())
                print(f'Test Loss: {test_result.loss.item()}')

                if test_result.metric:
                    test_metrices.append(test_result.metric)
                    print(f'Test metrics: {test_result.metric}')
                log.info(f"End of the training for epoch {e}")

                if self.model_path:
                    torch.save(self.model, self.model_path)
                    print(f"Saved the model to path:{self.model_path}")
                    log.info(f"Saved the model to path:{self.model_path}")

        return ModelBuildResult(train_metrices, train_losses, test_metrices, test_losses, learning_rates)


class ModelTrainer:

    def __init__(self, model, loss_fn, optimizer, scheduler, persister=None, metric_fn=None,
                 device=Utility.getDevice()):
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.model = model
        self.persister = persister
        self.metric_fn = metric_fn

    def __train_one_batch__(self, data, target):
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_fn(output, target)
        loss.backward()
        self.optimizer.step()
        return (loss, output)

    def train_one_epoch(self, loader, epoch_num):

        Utility.cleanup()
        log.info(f"Finished cleanup for epoch {epoch_num}")

        self.model.train()
        pbar = tqdm(loader, ncols=1000)

        total_loss = 0
        metrices = []

        log.info(f"Trainer starting the training for epoch: {epoch_num}")
        for idx, data in enumerate(pbar):

            log.info(f"Obtained the data for batch:{idx}")

            data['bg'] = data['bg'].to(self.device)
            data['fg_bg'] = data['fg_bg'].to(self.device)
            data['fg_bg_mask'] = data['fg_bg_mask'].unsqueeze(dim=1).to(self.device)
            data['fg_bg_depth'] = data['fg_bg_depth'].unsqueeze(dim=1).to(self.device)

            x = torch.cat((data['bg'], data['fg_bg']), dim=1)

            log.info(f"Starting the training for batch:{idx}")
            (loss, pred) = self.__train_one_batch__(x, data['fg_bg_mask'])
            log.info(f"End of the training for batch:{idx}")

            total_loss += loss
            self.scheduler.step()
            log.info(f"Scheduler step for the batch:{idx}")

            if self.persister is not None:
                self.persister(data, pred, epoch_num)
                log.info(f"Persisted the prediction for batch:{idx}")

            if self.metric_fn is not None:
                metric = self.metric_fn(data, pred)
                metrices.append(metric)
                log.info(f"Computed the metric for batch:{idx}")

            log.info(f"Completed the training for batch:{idx}")

        metric = None
        if self.metric_fn is not None:
            metric = self.metric_fn.aggregate(metrices)
        return PredictionResult(total_loss / len(loader.dataset), metric)


class ModelTester:

    def __init__(self, model, loss_fn, persister=None, metric_fn=None, device=Utility.getDevice()):
        self.device = device
        self.loss_fn = loss_fn
        self.model = model
        self.persister = persister
        self.metric_fn = metric_fn

    def __test_one_batch__(self, data, target):
        output = self.model(data)
        loss = self.loss_fn(output, target)
        return (loss, output)

    def test(self, loader, epoch_num):

        Utility.cleanup()
        log.info(f"Finished cleanup for epoch {epoch_num}")

        self.model.eval()

        pbar = tqdm(loader, ncols=1000)
        total_loss = 0
        metrices = []

        log.info(f"Tester starting the testing for epoch: {epoch_num}")

        for idx, data in enumerate(pbar):

            data['bg'] = data['bg'].to(self.device)
            data['fg_bg'] = data['fg_bg'].to(self.device)
            data['fg_bg_mask'] = data['fg_bg_mask'].unsqueeze(dim=1).to(self.device)
            data['fg_bg_depth'] = data['fg_bg_depth'].unsqueeze(dim=1).to(self.device)

            x = torch.cat((data['bg'], data['fg_bg']), dim=1)

            log.info(f"Starting the testing for batch:{idx}")
            (loss, pred) = self.__test_one_batch__(x, data['fg_bg_mask'])
            log.info(f"End of the testing for batch:{idx}")

            total_loss += loss

            if self.persister is not None:
                self.persister(data, pred, epoch_num)
                log.info(f"Persisted the prediction for batch:{idx}")

            if self.metric_fn is not None:
                metric = self.metric_fn(data, pred)
                metrices.append(metric)
                log.info(f"Computed the metric for batch:{idx}")

            log.info(f"Completed the training for batch:{idx}")

        metric = None
        if self.metric_fn is not None:
            metric = self.metric_fn.aggregate(metrices)
        return PredictionResult(total_loss / len(loader.dataset), metric)


class PredictionResult:

    def __init__(self, loss, metric=None):
        self.loss = loss
        self.metric = metric


class ModelBuildResult:

    def __init__(self, train_metrices, train_losses, test_metrices, test_losses, learning_rates=None):
        self.train_metrices = train_metrices
        self.train_losses = train_losses
        self.test_metrices = test_metrices
        self.test_losses = test_losses
        self.learning_rates = learning_rates
