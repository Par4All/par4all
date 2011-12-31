from pawsapp.tests import *

class TestTutorialAcca2011Controller(TestController):

    def test_index(self):
        response = self.app.get(url(controller='tutorial_acca-2011', action='index'))
        assert 'Current demo step:' in response
