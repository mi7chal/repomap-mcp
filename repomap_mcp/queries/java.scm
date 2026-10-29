; Copyright 2026 RepoMap MCP Contributors
; Derived from Aider (https://github.com/Aider-AI/aider)
; Copyright (c) Paul Gauthier, licensed under Apache License 2.0.
; Modified for repomap-mcp.

(class_declaration name: (identifier) @name) @definition.class

(interface_declaration name: (identifier) @name) @definition.interface

(method_declaration name: (identifier) @name) @definition.method

(constructor_declaration name: (identifier) @name) @definition.method

(class_declaration
  superclass: (superclass (type_identifier) @name)) @reference.implementation

(class_declaration
  interfaces: (super_interfaces (type_list (type_identifier) @name))) @reference.implementation

(interface_declaration
  (extends_interfaces (type_list (type_identifier) @name))) @reference.implementation

(method_invocation name: (identifier) @name) @reference.call
