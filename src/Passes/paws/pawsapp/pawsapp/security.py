# -*- coding: utf-8 -*-

from pyramid.security import Allow, Everyone


# Users and groups

USERS  = {'paws': 'foot_print',
          }

GROUPS = {'paws': ['group:members'],
          }


# Basic permissions

class RootFactory(object):
    __acl__ = [ (Allow, 'group:members', 'view'),
                ]

    def __init__(self, request):
        pass


def groupfinder(userid, request):
    if userid in USERS:
        return GROUPS.get(userid, [])


