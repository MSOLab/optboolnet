import os
import optboolnet as optbn
from optboolnet.config import ControlConfig
from typing import Dict, List

_INSTANCE_PATH = os.path.dirname(__file__)
_INSTANCE_GROUPS: Dict[str, List[str]] = {
    "small": ["S1", "S2", "S3", "S4"],
    "medium": ["M1", "M2", "M3"],
    "large": ["L1", "L2", "L3", "L4"],
}
_INSTANCE_LIST_FULL: List = [
    inst for inst_list in _INSTANCE_GROUPS.values() for inst in inst_list
]


def load_bn(fpath: str):
    return optbn.CNFBooleanNetwork(
        data=f"{fpath}/transition_formula.bnet",
        control_config=ControlConfig.from_json(f"{fpath}/control_setting.json"),
    )


def iter_bn(path_name: str):
    for fpath in os.listdir(path_name):
        yield fpath, optbn.CNFBooleanNetwork(
            data=f"{path_name}/{fpath}/transition_formula.bnet",
            control_config=ControlConfig.from_json(
                f"{path_name}/{fpath}/control_setting.json"
            ),
        )


def load_bn_in_repo(name: str):
    if name not in _INSTANCE_LIST_FULL:
        raise FileNotFoundError(
            f"Instance '{name}' is not in the repository. Try one of the following: {_INSTANCE_LIST_FULL}"
        )

    return optbn.CNFBooleanNetwork(
        data=f"{_INSTANCE_PATH}/{name}/transition_formula.bnet",
        control_config=ControlConfig.from_json(
            f"{_INSTANCE_PATH}/{name}/control_setting.json"
        ),
    )


def iter_bn_in_repo(group_name_list: List[str] = ["small", "medium", "large"]):
    for inst_group_name in group_name_list:
        for inst in _INSTANCE_GROUPS[inst_group_name]:
            yield inst, load_bn_in_repo(inst)
