import Lake
open Lake DSL

package «lean-graph-extractor» where
  -- add package configuration options here

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.22.0"

lean_lib «LeanGraphExtractor» where
  -- add library configuration options here

@[default_target]
lean_exe «extract_constants» where
  root := `ExtractConstants
  -- Enables the use of the Lean interpreter by the executable (e.g.,
  -- `runFrontend`) at the expense of increased binary size on Linux.
  -- Remove this line if you do not need such functionality.
  supportInterpreter := true
