'''Pipeline to build the trainer class.'''

# third-party imports
import omegaconf
# local imports
import training.common
import training.callback
import training.dataloading
import training.heads
import training.loss
import training.metrics
import training.optim
import training.trainer

import training.controller

def collect_trainer_comps(
        model: training.common.MultiHeadTrainable,
        data_summary: training.common.DataSummaryFull,
        config: omegaconf.DictConfig
    ) -> training.trainer.TrainerComponents:
    '''Setup the model trainer.'''

    # compile data loaders
    loader_config = training.dataloading.parse_loader_config(
        block_size=config.loader.block_size,
        patch_size=config.loader.patch_size,
        batch_size=config.loader.batch_size,
        stream_cache=config.loader.stream_cache,
        rand_sample_n=config.loader.rand_sample_n,
        rand_seed=config.loader.rand_seed
    )
    data_loaders = training.dataloading.get_dataloaders(
        data_summary=data_summary,
        loader_config=loader_config
    )

    # compile training heads basic specifications
    headspecs = training.heads.build_headspecs(
        data=data_summary,
        alpha_fn=config.loss.alpha_fn,
    )

    # compile training heads loss compute modules
    headlosses = training.loss.build_headlosses(
        headspecs=headspecs,
        config=config.loss.types,
        ignore_index=data_summary.meta.ignore_index,
    )

    # compile training heads metric compute modules
    headmetrics = training.metrics.build_headmetrics(
        headspecs=headspecs,
        ignore_index=data_summary.meta.ignore_index
    )

    # build optimizer and scheduler
    optim_config = training.optim.build_optim_config(
        opt_cls=config.optim.opt_cls,
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
        sched_cls=config.optim.sched_cls,
        sched_args=config.optim.sched_args
    )
    optimization = training.optim.build_optimization(
        model=model,
        config=optim_config
    )

    # generate callback instances
    callbacks = training.callback.build_callbacks()

    # collect components and return
    components = training.trainer.TrainerComponents(
        model=model,
        dataloaders=data_loaders,
        headspecs=headspecs,
        headlosses=headlosses,
        headmetrics=headmetrics,
        optimization=optimization,
        callbacks=callbacks,
    )
    return components

def generate_config(config: omegaconf.DictConfig) -> training.trainer.RuntimeConfig:
    '''Generate trainer runtime config from hydra.'''

    running_config = training.trainer.RuntimeConfig()
    # assign domain config
    running_config.data.dom_ids_name=config.data.domain_ids_name
    running_config.data.dom_vec_name=config.data.domain_vec_name
    # assign schedule config
    running_config.schedule.max_epoch=config.schedule.max_epoch
    running_config.schedule.max_step=config.schedule.max_step
    running_config.schedule.logging_interval=config.schedule.log_every
    running_config.schedule.eval_interval=config.schedule.val_every
    running_config.schedule.checkpoint_interval=5
    running_config.schedule.patience_epochs=5
    running_config.schedule.min_delta=0.01
    # assign monitoring config
    running_config.monitor.enabled=('iou',)
    running_config.monitor.metric=config.monitor.metric_name
    running_config.monitor.head=config.monitor.track_head_name
    running_config.monitor.mode=config.monitor.track_mode
    # assign precision config
    running_config.precision.use_amp=config.precision.use_amp
    # assign optimization config
    running_config.optim.grad_clip_norm=config.optimization.grad_clip_norm
    # return complete runnning config
    return running_config

def build_trainer(
        model: training.common.MultiHeadTrainable,
        data_summary: training.common.DataSummaryFull,
        config: omegaconf.DictConfig
    ) -> training.trainer.MultiHeadTrainer:
    '''Builder trainer.'''

    # collect componenets
    trainer_comps = collect_trainer_comps(model, data_summary, config)
    # generate runtime config
    runtime_config = generate_config(config=config.config)

    # build and return a trainer class
    return training.trainer.MultiHeadTrainer(
        components=trainer_comps,
        config=runtime_config,
        device='cuda'
    )

def build_controller(
        trainer: training.trainer.MultiHeadTrainer,
        config: omegaconf.DictConfig
    ) -> training.controller.Controller:
    '''Setup training controller.'''

    return training.controller.Controller(trainer, config)
