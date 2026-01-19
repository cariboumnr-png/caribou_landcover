'''Trainer class.'''

# standard imports
import contextlib
import copy
# third-party imports
import torch
# local imports
import training.trainer

class MultiHeadTrainer:
    '''
    Trainer for multi-head hierarchical segmentation.

    Responsibilities:
    - Wire up model, data loaders, optimizer, schedulers, head specs,
        and metrics.
    - Run training and validation loops.
    - Shift labels from 1-based to 0-based (keep 255) before loss/metrics.
    - Gate child-head loss/metrics under matching parent class regions.
    - Aggregate logs, track best metric, and save/load checkpoints.
    '''

    def __init__(
            self,
            components: training.trainer.TrainerComponents,
            config: training.trainer.RuntimeConfig,
            device: str
        ):
        '''
        Initialize and store all parts; move model to device.

        components: Concrete dataclass containing component protocols
        config: concrete dataclass containing runtime config
        callbacks: list of callback class protocols
        '''

        # parse arguments
        self.comps = components
        self.config = config
        self.device = device
        # move model to device
        self.model.to(self.device)
        # init the runtime state
        self.state = self._init_state()
        # setup callback classes
        self._setup_callbacks()

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        '''
        Train for one epoch.
        '''

        # train phase begin:
        # - set model to .train()
        self._emit('on_train_epoch_begin', epoch)

        # interate through training data batches
        for bidx, batch in enumerate(self.dataloaders.train, start=1):
            print(f'Training Batch_{bidx}', end='\r', flush=True)

            # batch start
            # - get new batch and parse into x, y_dict and domain
            # - reset optimizer gradients
            self._emit('on_train_batch_begin', bidx, batch)

            # batch forward
            # - forward the model with x and domain
            # - autocast context handled
            self._emit('on_train_batch_forward')

            # batch compute loss
            # - compute total and per-head loss
            # - autocast context handled
            self._emit('on_train_batch_compute_loss')

            # batch backward
            self._emit('on_train_backward')

            # gradient clipping
            self._emit('on_train_before_optimizer_step')

            # optimizer step
            self._emit('on_train_optimizer_step')

            # batch end
            # - update train loss and log dict
            self._emit('on_train_batch_end')

        # train phase end
        # - update logs and loss (total/per-head) for the epoch
        self._emit('on_train_epoch_end')
        return self.state.epoch_sum.train_logs

    def validate(self) -> dict:
        '''Validation.'''

        # val phase start
        # - reset head confusion matrices
        self._emit('on_validation_begin')

        # iterate through validation dataset
        for bidx, batch in enumerate(self.dataloaders.val):

            # batch start
            # - get new batch and parse into x, y_dict and domain
            self._emit('on_validation_batch_begin', bidx, batch)

            # batch forward
            # - forward the model with x and domain
            # - autocast context handled
            # - use torch.inference_mode()
            self._emit('on_validation_batch_forward')

            # batch end
            # - update per-head confusion matrix from outputs and y_dict
            self._emit('on_validation_batch_end')

        # val phase end
        # - calculate per-head IoU related metrics for and update the log dict
        self._emit('on_validation_end')
        return self.state.epoch_sum.val_logs

    def set_head_state(
            self,
            active_heads: list[str] | None=None,
            frozen_heads: list[str] | None=None,
            excluded_cls: dict[str, tuple[int, ...]] | None=None
        ) -> None:
        '''Set the head states from input configuration.'''

        # if no active heads provided, make all heads active
        if active_heads is None:
            active_heads = self.state.heads.all_heads

        # set active and frozen heads
        self.state.heads.active_heads = active_heads
        self.state.heads.frozen_heads = frozen_heads

        # set active heads at model
        self.model.set_active_heads(active_heads)
        # set active heads at trainer components
        self.state.heads.active_hspecs = {
            k: copy.deepcopy(self.headspecs[k]) for k in active_heads
        }
        self.state.heads.active_hloss = {
            k: copy.deepcopy(self.headlosses[k]) for k in active_heads
        }
        self.state.heads.active_hmetrics = {
            k: copy.deepcopy(self.headmetrics[k]) for k in active_heads
        }

        # set frozen heads to model if provided
        if frozen_heads is not None:
            self.model.set_frozen_heads(frozen_heads)

        # set excluded classes to active heads
        if excluded_cls is not None:
            for head in active_heads:
                excluded=excluded_cls.get(head)
                if excluded is not None:
                    self.state.heads.active_hspecs[head].set_exclude(excluded)
                    self.headmetrics[head].exclude_class_1b = excluded

    def reset_head_state(self):
        '''Reset the training state back to model defaults.'''

        self.state.heads.active_heads = None
        self.state.heads.frozen_heads = None
        self.state.heads.active_hspecs = None
        self.model.reset_heads()

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        '''
        Inference: run forward pass and return processed predictions.
        '''

        self.model.eval()
        device = self.device
        x = x.to(device)

        with torch.inference_mode(), self._autocast_ctx():
            outputs = self.model.forward(x)  # raw logits per head

        preds = {}
        for head, logits in outputs.items():
            if head.startswith('seg'):  # segmentation head
                preds[head] = torch.argmax(logits, dim=1)  # [B, H, W]
            elif head.startswith('cls'):  # classification head
                probs = torch.softmax(logits, dim=1)
                preds[head] = torch.topk(probs, k=1).indices  # top-1 class
            else:  # regression or other
                preds[head] = logits  # raw values
        return preds

    # ----------------------------internal methods----------------------------
    # runtime state initialization
    def _init_state(self) -> training.trainer.RuntimeState:
        '''Instantiate runtime state in line with trainer config.'''

        # instantiate a state
        state = training.trainer.RuntimeState()
        # state - heads
        state.heads.all_heads = list(self.headspecs.as_dict().keys())
        # state = optimization
        state.optim.scaler = torch.GradScaler(
            device=self.device,
            enabled=self.config.precision.use_amp
        )
        # return
        return state

    def _setup_callbacks(self) -> None:
        '''Setup callback classes (passes `self.trainer` instance).'''

        for callback in self.callbacks:
            callback.setup(self)

    # callback callers
    def _emit(self, hook: str, *args, **kwargs) -> None:
        '''Get callback by name and send trainer and args to execute.'''

        for callback in self.callbacks:
            method = getattr(callback, hook, None)
            if callable(method):
                method(*args, **kwargs)

    # -----------------------helpers called by callbacks----------------------
    # batch extraction
    def _parse_batch(self) -> None:
        '''
        Extract labels for a subset of active heads from a batched tensor.

        Args:
            batch (tuple[Tensor, Tensor]):
            A tuple where
                - `x`: Input image tensor of shape [B, C, H, W].
                - `y`: Stacked label tensor of shape [B, S, H, W], where S
                    corresponds to heads in the order of `all_heads`.
                - 'domain': batch-level domain dict of tensors. can be empty.

            all_heads (list[str]):
                List of all head names and their ordering

            active_heads (list[str]):
                List of head names to extract label slices for.

        Returns:
            tuple:
                - `x` (torch.Tensor): The unchanged input image tensor.
                - `y_dict` (dict[str, torch.Tensor]): A mapping from each
                active head name to its corresponding label tensor of
                shape [B, H, W].
        --------------------------------------------------------------------
        Note:
        Returned `x` tensor and tensors in `y_dict` will be on `device`.
        '''

        # make sure the batch is properly populated
        assert self.state.batch_cxt.batch is not None
        # parse x, y, domain from batch context
        x, y, domain = self.state.batch_cxt.batch
        # make sure it contains data indicated by dims
        assert isinstance(x, torch.Tensor) and x.ndim == 4 # shape [B, S, H, W]
        assert isinstance(y, torch.Tensor) and y.ndim == 4 # shape [B, S, H, W]
        # x and y should have the same batch size and h*w, slice might differ
        assert x.shape[0] == y.shape[0] and x.shape[-2:] == y.shape[-2:]
        # domain can be an empty dict or a dict[str, torch.Tensore]
        assert isinstance(domain, dict)
        if domain:
            # domain names must be str
            # each domain can be a tensor with the same batch size or None
            for k, v in domain.items():
                assert isinstance(k, str)
                if isinstance(v, torch.Tensor):
                    assert v.shape[0] == x.shape[0]

        # move tensors to device
        device = self.device
        x = x.to(device)
        y = y.to(device)
        domain = {k: v.to(device) for k, v in domain.items()} if domain else {}

        # fall back to all heads if activce heads not provided
        if self.state.heads.active_heads is None:
            self.state.heads.active_heads = self.state.heads.all_heads

        # precompute head index mapping
        head_to_idx = {name: i for i, name in enumerate(self.state.heads.all_heads)}
        # validate active heads
        missing = set(self.state.heads.active_heads) - set(self.state.heads.all_heads)
        if missing:
            raise KeyError(f'Active heads not found in all_heads: {missing}')

        # extract y slices
        y_dict = {head: y[:, head_to_idx[head], ...] for head in self.state.heads.active_heads}
        # get running domain
        running_domain = self.__sel_domain(domain)
        # assign to context
        self.state.batch_cxt.x = x
        self.state.batch_cxt.y_dict = y_dict
        self.state.batch_cxt.domain = running_domain

    def __sel_domain(
            self,
            domain: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor | None]:
        '''Get domain tensors - currently one for ids and one for vec.'''

        ids_name = self.config.data.dom_ids_name
        vec_name = self.config.data.dom_vec_name
        ids = domain.get(ids_name) if ids_name is not None else None
        vec = domain.get(vec_name) if vec_name is not None else None
        return {'ids': ids, 'vec': vec}

    # precision context
    def _autocast_ctx(self):
        '''Autocast context for training.'''
        device_type = self.device
        # pick dtype; feel free to prefer bf16 if supported:
        dtype = torch.bfloat16 if device_type == 'cpu' else torch.float16
        if self.config.precision.use_amp:
            return torch.autocast(
                device_type=device_type,
                dtype=dtype,
                enabled=self.config.precision.use_amp
            )
        return contextlib.nullcontext()

    def _val_ctx(self):
        '''Autocast and no gradient context for validation.'''
        stack = contextlib.ExitStack()
        stack.enter_context(torch.inference_mode())
        stack.enter_context(self._autocast_ctx())
        return stack

    # training phase
    def _compute_loss(self) -> None:
        '''Wrapper for compute loss.'''

        # sanity
        assert self.state.heads.active_hspecs is not None
        assert self.state.heads.active_hloss is not None
        # call loss function
        total, perhead = training.trainer.multihead_loss(
            multihead_preds=self.state.batch_out.preds,
            multihead_targets=self.state.batch_cxt.y_dict,
            headspecs=self.state.heads.active_hspecs,
            headlosses=self.state.heads.active_hloss
        )
        # pass to state
        self.state.batch_out.total_loss = total
        self.state.batch_out.head_loss = perhead

    def _clip_grad(self):
        '''Gradient clip'''
        if self.config.optim.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.optim.grad_clip_norm
            )

    def _update_train_logs(self, bidx: int) -> bool:
        '''
        Average losses and update per-head loss logs at intervals.

        Set `bidx = -1` to flush results at end of training epoch.
        '''

        logs = {}
        # update log at interval
        if bidx == -1 or bidx % self.config.schedule.logging_interval == 0:
            # average total loss so far
            avg_loss = self.state.epoch_sum.train_loss / max(1, bidx)
            logs['train_loss_total'] = avg_loss
            # per-head losses
            for h, v in self.state.batch_out.head_loss.items():
                logs[f'train_loss_{h}'] = float(v)
            # extras - lr
            logs['train_lr'] = self.optimization.optimizer.param_groups[0]['lr']

            # assgin to state dict
            self.state.epoch_sum.train_logs = logs
        # if logs are updated: provide a printer friendly text and return flag
        if logs:
            text_list = [f'{k}_loss: {v:.4f}' for k, v in logs.items()]
            text = f'{bidx:05d} | ' + '|'.join(text_list)
            self.state.epoch_sum.train_logs_text = text
            return True
        return False

    # validation phase
    def _update_conf_matrix(self) -> None:
        '''Update confusion matrix in-place at validation batch end.'''

        # sanity
        assert self.state.heads.active_hmetrics is not None
        preds = self.state.batch_out.preds
        targets = self.state.batch_cxt.y_dict
        for head, logits in preds.items():
            parent = self.headspecs[head].parent_head
            parent_1b = targets.get(parent) if parent is not None else None
            # retrieve head metric calculator
            metrics_module = self.state.heads.active_hmetrics[head]
            metrics_module.update(
                p0=logits,                              # 0-based
                t1=targets[head],                       # 1-based
                parent_raw_1b=parent_1b                 # 1-based (keyword arg)
            )

    def _compute_iou(self) -> None:
        '''Compute IoU at the end of validation phase.'''

        # sanity
        assert self.state.heads.active_hmetrics is not None
        val_logs: dict[str, dict] = {}
        val_logs_text: dict[str, str] = {}
        # calculate IoU related metrics for each head
        for head, metrics_module in self.state.heads.active_hmetrics.items():
            metrics_module.compute() # final metrics from batch accumulations
            val_logs[head] = metrics_module.metrics_dict
            val_logs_text[head] = metrics_module.metrics_text
        self.state.epoch_sum.val_logs = val_logs
        self.state.epoch_sum.val_logs_text = val_logs_text

    # -------------------------convenience properties-------------------------
    @property
    def model(self):
        '''Shortcut to model.'''
        return self.comps.model

    @property
    def dataloaders(self):
        '''Shortcut to dataloaders.'''
        return self.comps.dataloaders

    @property
    def headspecs(self):
        '''Shortcut to headspecs.'''
        return self.comps.headspecs

    @property
    def headlosses(self):
        '''Shortcut to headlosses.'''
        return self.comps.headlosses

    @property
    def headmetrics(self):
        '''Shortcut to headmetrics.'''
        return self.comps.headmetrics

    @property
    def optimization(self):
        '''Shortcut to optimization.'''
        return self.comps.optimization

    @property
    def callbacks(self):
        '''Shortcut to callbacks.'''
        return self.comps.callbacks
