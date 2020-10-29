import registry.registries as registry

def define_loss(name, args):
    return registry.CRITERIONS.get_instance(name, **args)

def define_loss_from_params(args):
    return registry.CRITERIONS.get_from_params(**args)