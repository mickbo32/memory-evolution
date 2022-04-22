"""Memory evolution.

This module provides useful gym environments for the study of memory functions,
with a particular focus on spatial memory, a very ancient form of memory well
conserved in vertebrates species.
With this module you can test your artificial agents in
several tasks and let spatial memory evolve in these artificial settings.

"""

__author__ = "Michele Baldo"
__copyright__ = "Copyright 2022, Michele Baldo"
__version__ = "0.0.0"

from . import agents
from . import envs
from . import evaluate
from . import geometry
from . import utils

from .evaluate import evaluate_agent

