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
        verbose=True,
        monitor='valid/chamfer_distance_epoch',
        mode='min',
        save_top_k=3,
        save_last=True,
        filepath='model_ep-{epoch}_cd_full-{valid_chamfer_distance}_small_cd-{valid_chamfer_distance_small}'
    )

    trainer = Trainer(gpus=1 , max_epochs=cfg.num_epochs, logger=logger,
                          checkpoint_callback=checkpoint_callback,
                          log_every_n_steps=cfg.log_freq, flush_logs_every_n_steps=cfg.log_freq, log_gpu_memory=True, limit_train_batches=cfg.train_steps_limit, limit_val_batches=cfg.val_steps_limit,
                          resume_from_checkpoint=cfg.checkpoint_path, check_val_every_n_epoch=cfg.check_val_every_n_epoch,
                          callbacks=[LearningRateMonitor()])

    trainer.test(model)



if __name__ == '__main__':
    main()
