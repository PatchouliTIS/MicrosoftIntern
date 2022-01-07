import os, sys, itertools
from dataclasses import dataclass
from functools import reduce
from dateutil import parser
import datetime

@dataclass
class Entity:
    """A data class representing an module
    """
    name: str
    version: str
    module_id: str
    family_id: str
    entity_type: str
    created_date: str
    status: str

def load_list(filepath):
    """Load module list from filepath and convert them to Entity objects

    Args:
        filepath (str): the path to module list file. 

    Returns:
        list: list of module entities
    """
    with open(filepath) as f:
        lines = f.readlines()
        return [Entity(*(l.strip().split(" * "))) for l in lines]

def select_active_modules(entities):
    """Filter the module list, and only keep active modules

    Args:
        entities (list): list of module entities

    Returns:
        list: list of filterred module entities
    """
    return [e for e in entities if e.entity_type == "Module" and e.status == "Active"]

def group_by_family(modules):
    """Group module list by family id

    Args:
        modules (list): list of module entities

    Returns:
        iterator: (key, group) pairs
    """
    return itertools.groupby(modules, key=lambda e: e.family_id)

def to_date(entity):
    """Convert moduel entity to date

    Args:
        entity (Entity): module entity

    Returns:
        datetime: the converted datetime object
    """
    return parser.parse(entity.created_date)
    
def choose_latest(groups):
    """Choose the latest module from module groups

    Args:
        groups (iterator): iterator returning (key, group) pairs

    Returns:
        list: list of module entities
    """
    return [reduce(lambda e1,e2: e1 if to_date(e1) > to_date(e2) else e2, list(g)) for k,g in groups]

def output_modules(modules, filepath):
    """Write module list to filepath

    Args:
        modules (list): list of module entities
        filepath (str): path to a file. the module list will be written to this file.
    """
    def to_line(entity):
        return " * ".join([entity.name, entity.version, entity.module_id, entity.created_date])

    lines = [to_line(m) for m in modules]
    with open(filepath, "w") as f:
        f.writelines("\n".join(lines))

def main(input_file, output_file):
    all_entities = load_list(input_file)
    all_modules = select_active_modules(all_entities)
    module_group_by_family = group_by_family(all_modules)
    modules = choose_latest(module_group_by_family)

    if len(modules) > 0:
        modules.sort(key=lambda x: datetime.datetime.now() - to_date(x))
        output_modules(modules, output_file)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("python filter_list.py <input file> <output file>")
    else:
        main(sys.argv[1], sys.argv[2])
