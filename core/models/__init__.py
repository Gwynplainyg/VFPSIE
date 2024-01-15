import torch
import sys
sys.path.append('.')
sys.path.append('core.models')



def fetch_model(config):
    model_name = config['model']
    if model_name=="VFPSIE":
        from core.models.VFPSIE import Model


    model=Model()
    return Model


