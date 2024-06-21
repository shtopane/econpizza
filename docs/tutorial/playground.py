import sys
import os

# Add two directories up to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import econpizza as ep
example_hank2 = ep.examples.hank2
mod = ep.load(example_hank2)
_ = mod.solve_stst()