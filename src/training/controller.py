'''Curriculum controller class.'''

# standard imports
import dataclasses
import typing
# third-party imports
import omegaconf
# local imports
import training.common
import training.trainer
import utils.logger

log = utils.logger.Logger(name='phase')

@dataclasses.dataclass
class Phase:
    '''doc'''
    name: str
    num_epochs: int
    active_heads: list[str]
    frozen_heads: list[str] | None = None
    excluded_cls: dict[str, tuple[int, ...]] | None = None
    lr_scale: float = 1.0

    def __repr__(self) -> str:
        indent: int=2
        s = ' ' * indent
        return f'\n{s}'.join([
            f'- Phase Name:\t{self.name}',
            f'- Max Epochs:\t{self.num_epochs}',
            f'- Active Heads:\t{self.active_heads}',
            f'- Frozen Heads:\t{self.frozen_heads}',
            f'- Excld. Class:\t{self.excluded_cls}',
            f'- LR Scale:\t{self.lr_scale}'
        ])

class Controller:
    '''doc'''
    def __init__(
            self,
            trainer: training.trainer.MultiHeadTrainer,
            config: omegaconf.DictConfig
        ):
        '''Initialization'''

        self.trainer = trainer
        self.config = config
        self.phases = self._generate_phases(config)
        self.current_phase_idx = 0
        self.current_phase = self.phases[self.current_phase_idx]

    def fit(self, stopat: str | int| None=None) -> None:
        '''Main entry.'''

        for phase in self.phases:
            print('__Phase details__')
            print(phase)
            print(f'__Phase [{phase.name}] started__')
            self._train_phase()
            print(f'__Phase [{phase.name}] finished__')
            self._save_phase()
            self._next_phase()
            if self.done:
                print('__Experiment Complete__')
                log.log('INFO', 'All training phases finished')
                break
            if isinstance(stopat, str) and stopat == phase.name:
                print('__Experiment Complete__')
                log.log('INFO', f'Training stopped at: {stopat}')
                break
            if isinstance(stopat, int) and stopat == self.current_phase_idx:
                print('__Experiment Complete__')
                log.log('INFO', f'Training stopped at: Phase_{stopat + 1}')
                break

    def _train_phase(self) -> None:
        '''Train the current phase.'''

        num_epoch = self.current_phase.num_epochs
        phase = self.current_phase
        # train
        for epoch in range(1, num_epoch + 1):
            print(f'__Epoch: {epoch}/{num_epoch}__')
            self.trainer.set_head_state(
                active_heads=phase.active_heads,
                frozen_heads=phase.frozen_heads,
                excluded_cls=phase.excluded_cls
            )
            _ = self.trainer.train_one_epoch(epoch)
            # validate at set interval
            if self.trainer.config.schedule.eval_interval is not None and \
                epoch % self.trainer.config.schedule.eval_interval == 0:
                _ = self.trainer.validate()

    def _next_phase(self) -> None:
        '''Move on to the next phase.'''

        # advance phase idx
        self.current_phase_idx += 1
        # if already done stop
        if self.done:
            return
        # else reset trainer state and continue
        self.trainer.reset_head_state()

    def _save_phase(self) -> None:
        '''Save at the current phase.'''

        fpath = f'{self.config.ckpt_dpath}/{self.current_phase.name}.pt'
        ckpt_meta: training.common.CheckpointMetaLike = {
            'best_value': self.trainer.state.metrics.best_value,
            'epoch': self.trainer.state.metrics.best_epoch,
            'step': self.trainer.state.progress.global_step
        }
        training.trainer.save(
            model=self.trainer.comps.model,
            ckpt_meta=ckpt_meta,
            optimizer=self.trainer.comps.optimization.optimizer,
            scheduler=self.trainer.comps.optimization.scheduler,
            fpath=fpath
        )
        log.log('INFO', f'Phase {self.current_phase.name} saved to {fpath}')

    @staticmethod
    def _generate_phases(config: omegaconf.DictConfig) -> list[Phase]:
        '''doc'''

        phases: list[Phase] = []
        cfg_phases = typing.cast(
            typ=dict[str, typing.Any],
            val=omegaconf.OmegaConf.to_container(config.phases, resolve=True)
        )
        # iterate through phases in config (1-based)
        for p in cfg_phases.values():
            phases.append(
                Phase(
                    name=p['name'],
                    num_epochs=p['num_epochs'],
                    active_heads=p['active_heads'],
                    frozen_heads=p['frozen_heads'],
                    excluded_cls=p['masked_classes'],
                    lr_scale=p['lr_scale']
                )
            )

        # return
        return phases

    @property
    def done(self) -> bool:
        '''Returns whether controller has reached the final phase.'''
        return self.current_phase_idx >= len(self.phases)
