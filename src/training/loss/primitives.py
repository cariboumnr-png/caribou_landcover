'''Loss calculation math blocks.'''

# third-party imports
import torch
import torch.nn
import torch.nn.functional
# local imports
import training.loss

class FocalLoss(training.loss.PrimitiveLoss):
    '''Multi-class focal loss with proper ignore_index handling.'''

    def __init__(
            self,
            alpha: list[float] | None,
            gamma: float,
            reduction: str,
            ignore_index: int
        ):
        '''
        Docstring for __init__

        :param self: Description
        :param alpha: Description
        :type alpha: list[float] | None
        :param gamma: Description
        :type gamma: float
        :param reduction: Description
        :type reduction: str
        :param ignore_index: Description
        :type ignore_index: int
        '''
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            *,
            mask: torch.Tensor | None=None
        ) -> torch.Tensor:
        '''Forward.'''

        # logits: [B,C,H,W]; targets: [B,H,W]
        _, c, _, _ = logits.shape
        device = logits.device

        # Build valid mask: not ignore_index
        valid = targets != self.ignore_index
        if mask is not None:
            assert mask.shape == targets.shape and mask.dtype == torch.bool, \
                f'{targets.shape}, {mask.shape}, {mask.dtype}'
            valid = valid & mask

        if valid.sum() == 0:
            return logits.new_zeros(())

        # Flatten
        logits = logits.permute(0, 2, 3, 1).reshape(-1, c) # [N, C]
        targets = targets.reshape(-1) # [N]
        valid = valid.reshape(-1) # [N]

        logits = logits[valid]                # [M, C]
        targets = targets[valid]              # [M]
        assert targets.max() < c, f'Invalid target index: {targets.max()} >= {c}'

        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        # clamp to avoid extreme grads
        log_probs = torch.clamp(log_probs, min=-30.0, max=30.0)

        probs = log_probs.exp()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [M]
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)          # [M]

        # clamp
        log_pt = torch.clamp(log_pt, min=-20.0)  # avoid -inf
        pt = torch.clamp(pt, min=1e-6)           # avoid 0

        # alpha weighting
        if self.alpha is None:
            alpha_t = 1.0 # no effect
        else:
            alpha_t = torch.tensor(self.alpha).to(device)[targets]  # per-class

        # loss compute
        loss = - alpha_t * (1 - pt).pow(self.gamma) * log_pt  # [M]

        # reduction nmethod
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

class DiceLoss(training.loss.PrimitiveLoss):
    '''Dice loss.'''

    def __init__(self, smooth: float, ignore_index: int):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            *,
            mask: torch.Tensor | None=None
        ) -> torch.Tensor:
        '''Forward.'''

        probs = torch.nn.functional.softmax(logits, dim=1)
        n, c, h, w = probs.shape
        targets_oh = torch.zeros(
            (n, c, h, w), device=logits.device, dtype=probs.dtype
        )
        valid_mask = torch.ones_like(
            targets, dtype=torch.float32, device=logits.device
        )

        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index).float()
            targets = torch.where(
                targets == self.ignore_index, torch.zeros_like(targets), targets
            )

        # external mask support
        if mask is not None:
            # accept bool or float; cast to float for multiplication
            valid_mask = valid_mask * (mask.float())
        # Early exit to avoid NaNs if nothing valid
        if valid_mask.sum().item() == 0:
            return logits.new_zeros(())

        targets_oh.scatter_(1, targets.unsqueeze(1), 1.0)
        probs = probs * valid_mask.unsqueeze(1)
        targets_oh = targets_oh * valid_mask.unsqueeze(1)

        dims = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dims)
        union = probs.sum(dims) + targets_oh.sum(dims)

        # Stabilize
        intersection = torch.clamp(intersection, min=0.0)
        union = torch.clamp(union, min=self.smooth)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice = torch.clamp(dice, min=0.0, max=1.0)

        return 1 - dice.mean()
