from pawsapp.tests import *

class TestToolsOutRegionsController(TestController):

    def test_index(self):
        response = self.app.get(url(controller='tools_out_regions', action='index'))
        assert 'OUT regions' in response
