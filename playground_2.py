import pickle
import econpizza as ep
from jax import export

mod = ep.load(ep.examples.hank)
# Optionally, deserialize to verify
with open('model_stst.pkl', 'rb') as file:
    loaded_stst = pickle.load(file)

with open('model_distributions.pkl', 'rb') as file:
    loaded_distributions = pickle.load(file)

# print(loaded_stst)
# print(type(loaded_distributions))
# print(type(loaded_stst["B"]))

with open('model_func_eqns.pkl', 'rb') as file:
    func_eqns:bytearray = pickle.load(file)

rehydrated_exp: export.Exported = export.deserialize(func_eqns)
print(rehydrated_exp.in_avals)
# mod['stst'] = loaded_obj

# print(mod['stst'])