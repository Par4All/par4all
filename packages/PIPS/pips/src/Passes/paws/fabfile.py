# -*- coding: utf-8 -*-

from fabric.api import env, run, local

def setup():
    
    # Setup virtual env
    local('virtualenv --no-site-packages .')
    install_reqs(upgrade=False)
    freeze()
    develop()

# Install requirements
def install_reqs(upgrade=True):
    # Install requirements in virtualenv
    if upgrade:
        local('bin/pip install -U -r requirements.txt')
    else:
        local('bin/pip install -r requirements.txt')

# Freeze requirements
def freeze():
    """Freeze requirements
    """
    local('bin/pip freeze > installed.txt')

# Develop egg
def develop():
    """Develop egg
    """
    local('bin/python pawsapp/setup.py develop')

# Clean-up project directories
def clean():
    """Clean-up project directories
    """
    local('find . -name "*~" -exec rm {} \;')
