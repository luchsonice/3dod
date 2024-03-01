from cubercnn import data, util, vis
        

def priors_of_objects(cfg, dataset):
    priors = util.compute_priors(cfg, dataset)
    pass