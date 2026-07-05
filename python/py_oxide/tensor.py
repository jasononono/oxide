from . import core, system


class Tensor:
    def __init__(self, ctensor):
        super().__setattr__("ctensor", ctensor)

    def __getitem__(self, key):
        return self.ctensor.get_element(key)

    def __setitem__(self, key, value):
        self.ctensor.set_element(key, value)

    def __getattr__(self, key):
        if key == "ndim":
            return self.ctensor.get_ndim()
        if key == "size":
            return self.ctensor.get_size()
        if key == "offset":
            return self.ctensor.get_offset()
        if key == "shape":
            return self.ctensor.get_shape().copy()
        if key == "strides":
            return self.ctensor.get_strides().copy()
        
        raise RuntimeError(f"Oxide: tensor attribute '{key}' does not exist")
        
    def __setattr__(self, key, value):
        raise RuntimeError("Oxide: standalone tensor attributes are immutable")
    
    def __str__(self):
        return self.ctensor.get_string()