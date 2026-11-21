; Copyright 2026 RepoMap MCP Contributors
; Derived from Aider (https://github.com/Aider-AI/aider)
; Copyright (c) Paul Gauthier, licensed under Apache License 2.0.
; Modified for repomap-mcp.


(struct_item
    name: (type_identifier) @name) @definition.struct

(enum_item
    name: (type_identifier) @name) @definition.enum

(union_item
    name: (type_identifier) @name) @definition.struct


(source_file
    (type_item
        name: (type_identifier) @name) @definition.type)

(mod_item
    body: (declaration_list
        (type_item
            name: (type_identifier) @name) @definition.type))


(declaration_list
    (function_item
        name: (identifier) @name) @definition.method)


(function_item
    name: (identifier) @name) @definition.function

(trait_item
    name: (type_identifier) @name) @definition.interface

(mod_item
    name: (identifier) @name) @definition.module


(macro_definition
    name: (identifier) @name) @definition.macro


(call_expression
    function: (identifier) @name) @reference.call

(call_expression
    function: (field_expression
        field: (field_identifier) @name)) @reference.call

(macro_invocation
    macro: (identifier) @name) @reference.call


(impl_item
    trait: (type_identifier) @name) @reference.implementation

(impl_item
    type: (type_identifier) @name
    !trait) @reference.implementation
