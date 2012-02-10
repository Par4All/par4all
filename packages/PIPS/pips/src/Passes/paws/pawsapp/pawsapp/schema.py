# -*- coding: utf-8 -*-

"""
Colander schema for advanced properties

"""

import colander


class DictBool(colander.MappingSchema):
    id      = colander.SchemaNode(colander.String())
    checked = colander.SchemaNode(colander.Boolean(), missing=False)
    val     = colander.SchemaNode(colander.Boolean(), missing=True)
    

class DictInt(colander.MappingSchema):
    id      = colander.SchemaNode(colander.String())
    checked = colander.SchemaNode(colander.Boolean(), missing=False)
    val     = colander.SchemaNode(colander.Int())

class DictStr(colander.MappingSchema):
    id      = colander.SchemaNode(colander.String())
    checked = colander.SchemaNode(colander.Boolean(), missing=False)
    val     = colander.SchemaNode(colander.String())

class Bool(colander.SequenceSchema):
    dict = DictBool()

class Int(colander.SequenceSchema):
    dict = DictInt()

class Str(colander.SequenceSchema):
    dict = DictStr()

class Properties(colander.MappingSchema):
    bool = Bool()
    int  = Int()
    str  = Str()

class Analyses(colander.SequenceSchema):
    dict = DictStr()

class Phases(colander.SequenceSchema):
    dict = DictBool()

class Params(colander.MappingSchema):    
    properties = Properties()
    analyses   = Analyses()
    phases     = Phases()
