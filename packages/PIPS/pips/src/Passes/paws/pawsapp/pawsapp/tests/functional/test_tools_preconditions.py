from pawsapp.tests import *

class TestPreconditionsController(TestController):

    def test_index(self):
        response = self.app.get(url(controller='tools_preconditions', action='index'))
        assert 'Preconditions over scalar integer variables' in response
