'''Multi-head Unet architecture.'''

# third-party imports
import torch
import torch.nn
# local imports
import models.backbones.unet
import models.multihead.base
import models.multihead.concat
import models.multihead.config
import models.multihead.film

class MultiHeadUNet(models.multihead.base.BaseModel):
    '''
    UNet with 4 down/up levels and multi heads and conditioning support.

    Hybrid domain conditioning support:
      - Concatenation at input (if concat_domain_dim > 0 & mode in
      {'concat','hybrid'})
      - FiLM at bottleneck (if domain_embed_dim set & mode in
      {'film','hybrid'})
    '''

    def __init__(
            self,
            config: models.multihead.config.ModelConfig,
            cond: models.multihead.config.CondConfig
            ):
        super().__init__()

        # channels
        in_ch = config.in_ch
        base_ch = config.base_ch

        # convert per head logit adjustment to paramter dict
        self.logit_adjust = torch.nn.ParameterDict({
            h: torch.nn.Parameter(
                data=torch.tensor(v).view(1, -1, 1, 1).to(torch.float32),
                requires_grad=False
            )
            for h, v in config.logit_adjust.items()
        }) if config.enable_logit_adjust else None

        # multihead management
        heads_w_num_cls = {k: len(v) for k, v in config.heads_w_counts.items()}
        self.heads = _HeadManager(in_ch=base_ch, heads=heads_w_num_cls)

        # domain knowledge router
        self.domain_router = _DomainRouter(cond)

        # domain concatenation if proviced
        self.concat = models.multihead.concat.get(cond)
        add = self.concat.output_dim if self.concat is not None else 0

        # core UNet body
        self.body = models.backbones.unet.UNetBackbone(in_ch + add, base_ch)

        # conditioner
        self.film = models.multihead.film.get(cond, base_ch)

        # safety utilities
        self.safety = _NumericSafety(config.enable_clamp, config.clamp_range)

    def forward(
            self,
            x: torch.Tensor,
            **kwargs
        ) -> dict[str, torch.Tensor]:
        '''Forward.'''

        # numeric safty
        assert torch.isfinite(x).all(), "Input has NaN/Inf"

        # get domain if provided
        dom_ids = kwargs.get('ids', None)
        dom_vec = kwargs.get('vec', None)
        if dom_ids is not None:
            assert isinstance(dom_ids, torch.Tensor)
            # assert dom_ids.type() is torch.int64, dom_ids
        if dom_vec is not None:
            assert isinstance(dom_vec, torch.Tensor)

        # feed domain to router
        concat, film = self.domain_router.forward(dom_ids, dom_vec)

        # concatenate domain channels (if configured)
        if self.concat is not None:
            x = self.concat(x, *concat)

        # force float32 with clamping control for gradient stability
        with self.safety.autocast_context(dtype=torch.float32):
            # encoders
            x1, x2, x3, x4, xb = self.body.encode(self.safety.clamp(x))
            xb = self.safety.clamp(xb)
            # FiLM at bottleneck if provided
            if self.film is not None:
                z = self.film.embed(*film)
                xb = self.film.film_bottleneck(xb, z)
                xb = self.safety.clamp(xb)
            # decoders
            xs = tuple(self.safety.clamp(xx) for xx in [x1, x2, x3, x4, xb])
            x = self.body.decode(xs)

        # return head outputs
        return self.heads.forward(x, self.heads.state.active, self.logit_adjust)

    def set_active_heads(self, active_heads: list[str] | None=None) -> None:
        '''Convenience method to set active heads in forward method.'''
        self.heads.state.active = active_heads

    def set_frozen_heads(self, frozen_heads: list[str] | None=None) -> None:
        '''Freeze gradient for selected heads.'''
        self.heads.state.frozen = frozen_heads
        self.heads.freeze(frozen_heads)

    def reset_heads(self):
        '''Reset heads.'''
        self.heads.state.active = None
        self.heads.state.frozen = None

    @property
    def encoder(self) -> list:
        '''Returns a list of encoders'''
        return [self.body.inc, *self.body.downs, self.body.bottleneck]

    @property
    def decoder(self) -> list:
        '''Returns a list of encoders'''
        return [*self.body.ups]

# internal pieces
class _HeadManager(torch.nn.Module):
    '''
    Docstring for HeadManager.
    '''

    def __init__(
            self,
            in_ch: int,
            heads: dict[str, int],
        ):
        '''
        Initialize the `HeadManager`.
        '''

        super().__init__()
        # output convolution block
        self.outc = torch.nn.ModuleDict({
            head_name: torch.nn.Conv2d(in_ch, num_classes, kernel_size=1)
            for head_name, num_classes in heads.items()
        })
        self.state = models.multihead.config.HeadsState()
        self.state.active = list(self.outc.keys())
        self.state.frozen = None

    def forward(
            self,
            x: torch.Tensor,
            active_heads: list[str] | None=None,
            logit_adjust: torch.nn.ParameterDict | None=None
        ) -> dict[str, torch.Tensor]:
        '''Forward active heads if provided.'''

        # if external active heads provided
        if active_heads is not None:
            self.state.active = active_heads
        # reset logic
        if self.state.active is None:
            self.state.active = list(self.outc.keys())

        # iterate through active heads
        output_logits: dict[str, torch.Tensor] = {}
        for head_name in self.state.active:
            conv = self.outc[head_name]
            logits = conv(x)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            # if external logit adjust is provided
            if logit_adjust is not None:
                head_a = logit_adjust.get(head_name, None)
                if head_a is not None:
                    logits = logits + head_a.to(logits.dtype).to(logits.device)
            output_logits[head_name] = logits
        return output_logits

    def freeze(self, frozen_heads: list[str] | None=None) -> None:
        '''Freeze selected heads if provided.'''
        if frozen_heads is None:
            return
        for h in frozen_heads:
            for p in self.outc[h].parameters():
                p.requires_grad = False

class _DomainRouter(torch.nn.Module):
    '''
    Docstring for DomainRouter
    '''
    def __init__(
            self,
            cfg: models.multihead.config.CondConfig
        ):
        super().__init__()
        self.cfg = cfg
        #
        self.concat_proj = torch.nn.Linear(
            in_features=cfg.domain_vec_dim,
            out_features=cfg.concat.out_dim
        ) if cfg.domain_vec_dim else None
        self.film_proj = torch.nn.Linear(
            in_features=cfg.domain_vec_dim,
            out_features=cfg.film.embed_dim
        ) if cfg.domain_vec_dim else None

    def forward(
            self,
            ids: torch.Tensor | None,
            vec: torch.Tensor | None
        ) -> tuple[tuple, tuple]:
        '''
        Docstring for forward

        :param self: Description
        :param ids: Description
        :type ids: torch.Tensor | None
        :param vec: Description
        :type vec: torch.Tensor | None
        '''

        # Decide and shape what goes to concat vs film
        concat_ids = ids if self.cfg.concat.use_ids else None
        concat_vec = None
        film_ids = ids if self.cfg.film.use_ids else None
        film_vec = None

        if self.cfg.concat.use_vec and vec is not None:
            concat_vec = self.concat_proj(vec) \
                if self.concat_proj is not None else vec

        if self.cfg.film.use_vec and vec is not None:
            film_vec = self.film_proj(vec) \
                if self.film_proj is not None else vec

        return (concat_ids, concat_vec), (film_ids, film_vec)

class _NumericSafety():
    '''Numeric safety utilities.'''
    def __init__(
            self,
            enable_clamp: bool,
            clamp_range: tuple[float, float]
        ):
        self.enable_clamp = enable_clamp
        self.clamp_range = clamp_range

    def autocast_context(
            self,
            enable: bool=True,
            dtype: torch.dtype=torch.float16
        ) -> torch.autocast:
        '''AMP context manager.'''
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.autocast(device_type, dtype, enable)


    def clamp(self, x: torch.Tensor) -> torch.Tensor:
        '''Gradient safetp clamp.'''
        if not self.enable_clamp:
            return x
        mmin, mmax = self.clamp_range
        return torch.clamp(x, min=mmin, max=mmax)
