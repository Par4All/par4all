import pyps


def validate_fusion(self,parallelize=False, flatten=True):
    if parallelize==True:
        self.privatize_module()
        self.coarse_grain_parallelization()

    if flatten==True:
        self.flatten_code(unroll=False)

    print "//" 
    print "// Code before fusion"
    print "//" 
    self.display()

    # make the fusion
    self.loop_fusion()

    print "//"
    print "// Code after fusion"
    print "//" 
    self.display()

pyps.module.validate_fusion=validate_fusion




def validate_fusion(self, concurrent=False, **props):
    for m in self:
        m.validate_fusion(**props)

pyps.modules.validate_fusion = validate_fusion


