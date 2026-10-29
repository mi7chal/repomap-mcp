; Copyright 2026 RepoMap MCP Contributors
; Derived from Aider (https://github.com/Aider-AI/aider)
; Copyright (c) Paul Gauthier, licensed under Apache License 2.0.
; Modified for repomap-mcp.

(class name: [(constant) (scope_resolution name: (constant) @name)] @name) @definition.class

(module name: [(constant) (scope_resolution name: (constant) @name)] @name) @definition.module

(method name: (identifier) @name) @definition.method
(singleton_method name: (identifier) @name) @definition.method

(class
  superclass: (superclass (constant) @name)) @reference.implementation

(call method: (identifier) @name) @reference.call
