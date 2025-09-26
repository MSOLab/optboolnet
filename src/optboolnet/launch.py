from optboolnet.algorithm import BendersAttractorControl, BendersFixPointControl
from optboolnet.boolnet import CNFBooleanNetwork


def control_sync_attr_separation(
    bn,
    max_size: int,
    target: dict[str, int],
    exclude: list[str] = list(),
    max_attr_length: int = 15,
    **kwargs
):
    name = kwargs.get("name", "ControlSeparation")
    convert_bn = CNFBooleanNetwork.from_bnet(bn, target=target, exclude=exclude)
    s = BendersAttractorControl(name, convert_bn).get_control_strategies(
        max_control_size=max_size,
        max_length=max_attr_length,
        solve_separation=True,
        **kwargs
    )
    return s


def control_sync_attr_no_separation(
    bn,
    max_size: int,
    target: dict[str, int],
    exclude: list[str] = list(),
    max_attr_length: int = 15,
    **kwargs
):
    name = kwargs.get("name", "ControlNoSeparation")
    convert_bn = CNFBooleanNetwork.from_bnet(bn, target=target, exclude=exclude)
    s = BendersAttractorControl(name, convert_bn).get_control_strategies(
        max_control_size=max_size,
        max_length=max_attr_length,
        solve_separation=False,
        **kwargs
    )
    return s


def control_fixpoint(
    bn,
    max_size: int,
    target: dict[str, int],
    exclude: list[str] = list(),
    allow_empty_attractor: bool = True,
    **kwargs
):
    name = kwargs.get("name", "ControlFixpoint")
    convert_bn = CNFBooleanNetwork.from_bnet(bn, target=target, exclude=exclude)
    model = BendersFixPointControl(name, convert_bn)
    if allow_empty_attractor:
        model.allow_empty_attractor = True
    else:
        model.allow_empty_attractor = False
        model.use_high_point_relaxation = True
    s = model.get_control_strategies(max_control_size=max_size, max_length=1, **kwargs)
    return s
