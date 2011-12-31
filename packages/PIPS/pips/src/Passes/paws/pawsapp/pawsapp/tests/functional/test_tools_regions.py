from pawsapp.tests import *

class TestRegionsController(TestController):

    def test_index(self):
        response = self.app.get(url(controller='tools_regions', action='index'))
        assert 'Array regions' in response
