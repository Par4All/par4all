from pawsapp.tests import *

class TestToolsInRegionsController(TestController):

    def test_index(self):
        response = self.app.get(url(controller='tools_in_regions', action='index'))
        assert 'IN regions' in response
