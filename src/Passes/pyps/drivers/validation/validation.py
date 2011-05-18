import pyps,os

class vworkspace(pyps.workspace): 
    ''' This workspace is intended to handle some special PIPS validation 
    suff'''
    
    def __init__(self, *sources, **kwargs):
        """init a workspace from (optionnal) sources for validation
           name will be gather from WSPACE and sources from FILE environment variables 
        """

        # this workspace is intended to be run with WSPACE and FILE 
        # environment variable defined
        wspace = os.getenv("WSPACE")
        if wspace == None :
            raise RuntimeError("WSPACE environment variable has to be defined")

        file = os.getenv('FILE')
        if file == None :
            raise RuntimeError("FILE environment variable has to be defined")

        super(vworkspace, self).__init__(file,
                                         *sources,
                                         name=wspace,
                                         deleteOnClose=True, 
                                         deleteOnCreate=True,
                                         **kwargs)

        