import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from experiment import ThreeDExperiment



@hydra.main(config_path="configs/net_config.yml")
def main(cfg):
    print(cfg.pretty())
    model = ThreeDExperiment(cfg)

    logger = TensorBoardLogger("logs")
    checkpoint_callback = ModelCheckpoint(
        filepath='checkpoints/model_last.ckpt',
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    trainer = Trainer(gpus=1 , max_epochs=cfg.num_epochs, logger=logger,
                          checkpoint_callback=checkpoint_callback,
                          log_every_n_steps=cfg.log_freq, flush_logs_every_n_steps=cfg.log_freq, log_gpu_memory=True, limit_train_batches=cfg.steps_limit,
                          resume_from_checkpoint=cfg.checkpoint_path, check_val_every_n_epoch=cfg.check_val_every_n_epoch,
                          callbacks=[LearningRateMonitor()])

    trainer.fit(model)



if __name__ == '__main__':
    main()
