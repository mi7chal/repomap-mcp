; Copyright 2026 RepoMap MCP Contributors
; Derived from Aider (https://github.com/Aider-AI/aider)
; Copyright (c) Paul Gauthier, licensed under Apache License 2.0.
; Modified for repomap-mcp.

(class_declaration name: (name) @name) @definition.class

(interface_declaration name: (name) @name) @definition.interface

(method_declaration name: (name) @name) @definition.method

(function_definition name: (name) @name) @definition.function

(class_declaration
  (base_clause (name) @name)) @reference.implementation

(class_declaration
  (class_interface_clause (name) @name)) @reference.implementation

(interface_declaration
  (base_clause (name) @name)) @reference.implementation

(function_call_expression function: (name) @name) @reference.call
(member_call_expression name: (name) @name) @reference.call
