import pyps


def validate_fusion(self,parallelize=False, flatten=True):
    if flatten==True:
        self.flatten_code(unroll=False)
    if parallelize==True:
        self.privatize_module()
        self.coarse_grain_parallelization()


    print "//" 
    print "// Code before fusion"
    print "//" 
    self.display()

    # make the fusion
    self.loop_fusion_with_regions()

    print "//"
    print "// Code after fusion"
    print "//" 
    self.display()

pyps.module.validate_fusion=validate_fusion




def validate_fusion(self, concurrent=False, **props):
    for m in self:
        m.validate_fusion(**props)

pyps.modules.validate_fusion = validate_fusion


