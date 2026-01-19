'''Unet backbone'''

# third-part imports
import torch
import torch.nn
# local imports
import models.backbones.base

class UNetBackbone(models.backbones.base.Backbone):
    '''
    Core body of a standard UNet architecture.

    This module implements:
    - An encoder (contracting) with multiple downsampling blocks.
    - A bottleneck for deep feature representation.
    - A decoder (expanding) with upsampling blocks & skip connections.
    - A final output layer for per-pixel predictions.

    The network preserves spatial details via skip connections while
    capturing global context through progressive downsampling.
    '''

    # core UNet body
    def __init__(self, in_ch: int, base_ch: int, **kwargs):
        '''
        Initialize the UNet body.

        Args:
            in_ch (int): Number of input channels.
            base_ch (int): Base number of feature channels.
                Deeper layers use multiples of this.
            heads (dict[str, int]): Dictionary mapping head names to
                the number of output classes. Each head produces a map
                via a 1×1 convolution.
            **kwargs: Additional options passed to convolution blocks.
                see `_DoubleConv`.

        Architecture:
            - Initial block: Converts input channels to `base_ch`.
            - Down blocks: Halve spatial resolution and double channels
                at each step.
            - Bottom block: Deepest representation with `16 × base_ch`
                channels.
            - Up blocks: Double spatial resolution and reduce channels,
                concatenating skips.
            - Output heads: Apply 1×1 convolutions to produce per-pixel
                class logits.
        '''

        super().__init__()
        self._out_channels = base_ch # conforming to base class
        # initial convolution block with no norm
        self.inc = _DoubleConvBlock(in_ch, base_ch, norm=None, **kwargs)
        # 4 downs (bottleneck at the end)
        self.downs = torch.nn.ModuleList([
            _DownsampleBlock(base_ch,   base_ch*2,  **kwargs),
            _DownsampleBlock(base_ch*2, base_ch*4,  **kwargs),
            _DownsampleBlock(base_ch*4, base_ch*8,  **kwargs),
            _DownsampleBlock(base_ch*8, base_ch*16, **kwargs),
        ])
        # bottleneck
        self.bottleneck = _DoubleConvBlock(base_ch*16, base_ch*16, **kwargs)
        # 4 ups
        self.ups = torch.nn.ModuleList([
            _UpsampleBlock(base_ch*16 + base_ch*8, base_ch*8,  **kwargs),
            _UpsampleBlock(base_ch*8  + base_ch*4, base_ch*4,  **kwargs),
            _UpsampleBlock(base_ch*4  + base_ch*2, base_ch*2,  **kwargs),
            _UpsampleBlock(base_ch*2  + base_ch,   base_ch,    **kwargs)
        ])

        # Kaiming weight initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def encode(self, x: torch.Tensor):
        '''Encoders'''

        x1 = self.inc(x)                # H     in_ch ->  b
        x2 = self.downs[0](x1)          # H/2   b     ->  2b
        x3 = self.downs[1](x2)          # H/4   2b    ->  4b
        x4 = self.downs[2](x3)          # H/8   4b    ->  8b
        x5 = self.downs[3](x4)          # H/16  8b    ->  16b
        xb = self.bottleneck(x5)        # H/16  16b   --  16b
        return x1, x2, x3, x4, xb

    def decode(self, xs: tuple[torch.Tensor, ...]):
        '''Decoders'''

        x1, x2, x3, x4, xb = xs
        x = self.ups[0](xb, x4)
        x = self.ups[1](x, x3)
        x = self.ups[2](x, x2)
        x = self.ups[3](x, x1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''End-to-end UNet pass without external conditioning.'''

        x1, x2, x3, x4, xb = self.encode(x)
        x = self.decode((x1, x2, x3, x4, xb))
        return x

    @property
    def out_channels(self) -> int:
        return self._out_channels

class _DoubleConvBlock(torch.nn.Module):
    '''Two convolution blocks.'''

    def __init__(self, in_ch: int, out_ch: int, **kwargs):
        '''doc'''
        super().__init__()

        # unpack keyword arguments
        norm = kwargs.get('norm', 'gn') # default group norm
        gn_groups = kwargs.get('gn_groups', 8) # if group norm is used
        p_drop = kwargs.get('p_drop', 0.05) # drop out rate
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            self._get_norm(norm, out_ch, gn_groups),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout2d(p_drop) if p_drop > 0 else torch.nn.Identity(),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            self._get_norm(norm, out_ch, gn_groups),
            torch.nn.ReLU(inplace=False),
        )

    def forward(self, x):
        '''Forward.'''
        return self.block(x)

    @staticmethod
    def _get_norm(kind: str | None, num_channels: int, gn_groups: int = 8):
        '''Define image normalization method.'''
        # no normalization
        if kind is None or kind.lower() == 'none':
            return torch.nn.Identity() # Indentity layer
        # availabel methods: batchNorm, GroupNorm, and LayerNorm
        kind = kind.lower()
        if kind == 'bn':
            return torch.nn.BatchNorm2d(num_channels)
        if kind == 'gn':
            g = min(gn_groups, num_channels)
            while num_channels % g != 0 and g > 1: # fall back to (1, ...)
                g -= 1
            return torch.nn.GroupNorm(g, num_channels, eps=1e-4, affine=True)
        if kind == 'ln':
            return torch.nn.GroupNorm(1, num_channels, affine=True)
        raise ValueError(f'Unknown norm type: {kind}')

class _DownsampleBlock(torch.nn.Module):
    '''Downsample block.'''

    def __init__(self, in_ch: int, out_ch: int, **kwargs):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            _DoubleConvBlock(in_ch, out_ch, **kwargs)
        )

    def forward(self, x):
        '''Forward.'''

        return self.block(x)

class _UpsampleBlock(torch.nn.Module):
    '''Upsample block.'''

    def __init__(self, in_ch: int, out_ch: int, **kwargs):
        super().__init__()
        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False
        )
        self.conv = _DoubleConvBlock(in_ch, out_ch, **kwargs)

    def forward(self, x, skip):
        '''Forward.'''

        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1) # concatenate skip connection here
        x = x.contiguous()
        return self.conv(x)
