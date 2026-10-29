; Copyright 2026 RepoMap MCP Contributors
; Derived from Aider (https://github.com/Aider-AI/aider)
; Copyright (c) Paul Gauthier, licensed under Apache License 2.0.
; Modified for repomap-mcp.

(class_specifier name: (type_identifier) @name) @definition.class

(function_definition declarator: (function_declarator declarator: (identifier) @name)) @definition.function

(class_specifier
  (base_class_clause (type_identifier) @name)) @reference.implementation

(call_expression function: (identifier) @name) @reference.call
(call_expression function: (field_expression field: (field_identifier) @name)) @reference.call
