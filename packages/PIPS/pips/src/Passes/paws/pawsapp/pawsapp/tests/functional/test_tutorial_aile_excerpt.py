from pawsapp.tests import *

class TestTutorialAileExcerptController(TestController):

    def test_index(self):
        response = self.app.get(url(controller='tutorial_aile_excerpt', action='index'))
        assert 'Current demo step:' in response
