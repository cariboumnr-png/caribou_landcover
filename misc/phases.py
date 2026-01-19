'''Phase generation from json config.'''

# third-party imports
import numpy
import torch

# local imports
import training.controller
import utils.funcs

def generate(
        headspecs_fpath: str,
        phases_fpath: str
    ) -> list[training.controller.Phase]:
    '''doc'''

    # read from configurations
    headspecs_json_list = utils.funcs.load_json(headspecs_fpath)
    phase_json_list = utils.funcs.load_json(phases_fpath)

    phase_list = []
    for phase in phase_json_list:
        new_phase = training.controller.Phase(
            original_json=phase,
            name=phase['name'],
            num_epochs=phase['num_epochs'],
            active_heads=phase['active_heads'],
            frozen_heads=phase['frozen_heads'],
            head_adjust={
                'head_w_adjust': phase['head_adjust']['head_w_adjust'],
                'alpha_adjust': _get_alpha_adjust(phase, headspecs_json_list)
            },
            lr_scale=phase['lr_scale']
        )
        phase_list.append(new_phase)

    return phase_list

def _get_alpha_adjust(
        phase: dict,
        headspecs: dict,
    ) -> dict[str, dict[int, torch.Tensor]]:
    '''Generate alpha adjust tensors for all active heads.'''

    adjust_alpha: dict[str, dict[int, torch.Tensor]] = {}
    # iterate through all active heads
    for head in phase['active_heads']:
        # if a head is provided with an alpha adjustment dict
        if head in phase['head_adjust']['alpha_adjust']:
            # find the corresponding headspec
            headspec = next(h for h in headspecs if h['name'] == head)
            # create default target ratios aligning with head num of classes
            target_ratios = [1.0] * headspec['num_classes']
            # go to the head and change target ratios
            for cls, v in phase['head_adjust']['alpha_adjust'][head].items():
                cls_idx = int(cls.split('class')[1]) - 1 # 1-based to 0-based
                target_ratios[cls_idx] = v
            # get alpha adjustment tensor
            adjust_alpha.update({
                head: _alpha_adjust_from_targets(
                        alpha=headspec['loss_cfg']['alpha'],
                        targets=target_ratios,
                        ramping=phase['head_adjust']['alpha_ramp'][head],
                        num_epochs=phase['num_epochs']
                    )
            })
    return adjust_alpha

def _alpha_adjust_from_targets(
        alpha: list[float],
        targets: list[float],
        num_epochs: int,
        ramping: dict[str, bool],
    ) -> dict[int, torch.Tensor]:
    '''Generate an alpha adjustment tensor from target ratios.'''

    # assertion guards
    assert numpy.isclose(sum(alpha), 1, rtol=1e-9)
    assert len(alpha) == len(targets)
    assert all(0 <= r <=1 for r in targets)

    # split target ratios
    reducers = [idx for idx, ratio in enumerate(targets) if ratio != 1]
    receivers = [idx for idx, ratio in enumerate(targets) if ratio == 1]

    # sum alpha at reduced locs
    unchanged = sum(alpha[idx] for idx in receivers)
    # sum alpha at deducted locs for each epoch
    deductions = []
    for ep in range(1, num_epochs + 1):
        ep_deduct = 0.0
        for idx in reducers:
            if ramping.get(f'class{idx + 1}', False): # whether to ramp class
                ep_deduct += (alpha[idx] - targets[idx]) * (1 - ep / num_epochs)
            else:
                ep_deduct += alpha[idx] - targets[idx]
        deductions.append(ep_deduct)

    # relocate deducted alphas to receivers equally
    adjustments: dict[int, torch.Tensor] = {}
    for ep in range(1, num_epochs + 1):
        adjust = [1.0] * len(alpha)
        deduction = deductions[ep - 1]
        for idx , ratio in enumerate(targets):
            if idx in receivers:
                adjust[idx] += deduction / unchanged
            else:
                if ramping.get(f'class{idx + 1}', False): # whether to ramp class
                    adjust[idx] = ep / num_epochs
                else:
                    adjust[idx] = ratio
        # # update return dict
        adjustments.update({ep: torch.tensor(adjust, dtype=torch.float32)})

    # return a list tensors
    return adjustments

if __name__ == '__main__':
    pp = generate(
        headspecs_fpath='./config/head_specs.json',
        phases_fpath='./config/train_phases.json'
    )
    for p in pp:
        if p.name != 'Phase2':
            continue
        aa = p.head_adjust['alpha_adjust']['layer1']
        for e, a in aa.items():
            ao = torch.Tensor([
                0.0940414468886662,
                0.008754173995271486,
                0.10241225756549191,
                0.26770745446916105,
                0.07461822823772474,
                0.4524664388436845
            ])
            print(e)
            print(a)
            print(a * ao, torch.sum(ao), torch.sum(a * ao))
            print('-------------------------')
