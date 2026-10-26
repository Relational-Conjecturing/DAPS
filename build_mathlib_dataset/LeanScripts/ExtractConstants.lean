/-
Extract constants (declarations) from Lean modules with complete metadata
Inspired by ntp-toolkit's declarations.lean and premises.lean

This script extracts:
- Full qualified names (e.g., Nat.Basic.add_comm)
- Declaration types (theorem, def, axiom, instance, inductive, structure, class)
- Type signatures with pp.fullNames enabled
- Line and column positions
- Dependencies (used constants)
- Whether it's a human theorem
-/

import Lean
import Mathlib.Lean.CoreM
import Mathlib.Control.Basic
import Mathlib.Lean.Expr.Basic
import Batteries.Lean.HashMap

open Lean Meta Core

/-! ## Pretty Printing Configuration -/

/-- Configure pretty printing to show full names and disable notations -/
def withFullNamesPP {m α} [Monad m] [MonadEnv m] [MonadWithOptions m] (x : m α) : m α := do
  withOptions (fun o => o
    |>.set `pp.fullNames true        -- Show fully qualified names
    |>.set `pp.funBinderTypes true   -- Show function binder types
    |>.set `pp.numericTypes true     -- Show numeric types (Nat not ℕ)
    |>.set `pp.coercions.types true  -- Show coercion types
    |>.set `pp.letVarTypes true      -- Show let variable types
  ) x

/-! ## Helper Functions -/

/-- Check if a name is an auxiliary lemma (should be unfolded) -/
def isAuxLemma : Name → Bool
| .num (.str _ "_auxLemma") _ => true
| _ => false

/-- Check if name should be blacklisted -/
def isBlackListedDecl (name : Name) : Bool :=
  name == ``sorryAx || name.isInternalDetail

/-- Check if a name is a theorem -/
def Name.isTheorem (name : Name) : CoreM Bool := do
  let .some ci := (← getEnv).find? name
    | return false
  match ci with
  | .thmInfo _ => return true
  | _ => return false

/-- Check if a name is a human theorem (has declaration range) -/
def isHumanTheorem (name : Name) : CoreM Bool := do
  let hasDeclRange := (← findDeclarationRanges? name).isSome
  let isTheorem ← Name.isTheorem name
  let notProjFn := !(← isProjectionFn name)
  return hasDeclRange && isTheorem && notProjFn

/-! ## Dependency Extraction -/

/-- Get all constants used in a declaration, with unfolding support -/
partial def getUsedConstantsWithUnfolding (c : ConstantInfo)
    (constantsMap : Std.HashMap Name ConstantInfo)
    (unfolding : Name → Bool := isAuxLemma) : NameSet × NameSet := Id.run do
  let mut direct : NameSet := ∅
  let mut unfolded : NameSet := ∅
  for n in c.getUsedConstantsAsSet do
    if unfolding n then
      if let some c := constantsMap.get? n then
        unfolded := unfolded ++ c.getUsedConstantsAsSet
    else
      direct := direct.insert n
  return (direct, unfolded)

/-- Traverse expression to collect constants in explicit arguments only -/
partial def getExplicitConstants : Expr → MetaM NameSet
| .app f x => do
  match (← try? (inferType f)) with
  | some (.forallE _ _ _ .default) => return (← getExplicitConstants f) ++ (← getExplicitConstants x)
  | _ => getExplicitConstants f
| .lam _ t b _ => do getExplicitConstants (b.instantiate1 (← mkFreshExprMVar t))
| .forallE _ t b _ => do getExplicitConstants (b.instantiate1 (← mkFreshExprMVar t))
| .letE n t v b _ => return (← getExplicitConstants v)
    ++ (← withLetDecl n t v fun fvar => getExplicitConstants (b.instantiate1 fvar))
| .const n _ => return NameSet.empty.insert n
| .mdata _ e => getExplicitConstants e
| _ => return NameSet.empty

/-- Get explicit constants from a ConstantInfo -/
def getConstantInfoExplicitConstants (c : ConstantInfo) : MetaM NameSet := do
  match c.value? with
  | some e => getExplicitConstants e
  | none => return ∅

/-! ## Main Extraction Logic -/

/-- Get the category string for a ConstantInfo -/
def getConstantCategory (env : Environment) (cinfo : ConstantInfo) : String :=
  match cinfo with
  | .axiomInfo _ => "axiom"
  | .thmInfo _ => "theorem"
  | .opaqueInfo _ => "opaque"
  | .defnInfo i =>
      if isInstanceCore env i.name then "instance"
      else "def"
  | .inductInfo i =>
      if isClass env i.name then "class"
      else if isStructure env i.name then "structure"
      else "inductive"
  | .ctorInfo _ => "constructor"
  | .recInfo _ => "recursor"
  | .quotInfo _ => "quot"

/-- Extract single constant information to JSON -/
def extractConstantInfo (name : Name) (cinfo : ConstantInfo)
    (constantsMap : Std.HashMap Name ConstantInfo) : MetaM Json := do
  let env ← getEnv
  let category := getConstantCategory env cinfo

  -- Get position information
  let range ← findDeclarationRanges? name
  let (line, column) := match range with
    | some r => (r.range.pos.line, r.range.pos.column)
    | none => (0, 0)

  -- Get type signature with full names
  let typeStr ← withFullNamesPP do
    let fmt ← ppExpr cinfo.type
    return fmt.pretty 100000

  -- Get docstring
  let doc? ← findDocString? env name

  -- Get dependencies
  let (direct, unfolded) := getUsedConstantsWithUnfolding cinfo constantsMap isAuxLemma
  let explicit ← getConstantInfoExplicitConstants cinfo

  let allDeps := Lean.RBTree.toList (direct ++ unfolded)
  let dependencies := List.map (fun depName => Json.mkObj [
    ("name", Json.str depName.toString),
    ("explicit", Json.bool (explicit.contains depName)),
    ("direct", Json.bool (direct.contains depName))
  ]) allDeps

  -- Check if it's a prop and human theorem
  let isProp ← Meta.isProp cinfo.type
  let isHumanThm ← isHumanTheorem name

  return Json.mkObj [
    ("name", Json.str name.toString),
    ("category", Json.str category),
    ("type", Json.str typeStr),
    ("line", Json.num line),
    ("column", Json.num column),
    ("isProp", Json.bool isProp),
    ("isHumanTheorem", Json.bool isHumanThm),
    ("doc", match doc? with | some doc => Json.str doc | none => Json.null),
    ("dependencies", Json.arr dependencies.toArray)
  ]

/-- Extract all constants from given modules -/
def extractModuleConstants (moduleNames : Array Name) : CoreM Unit := do
  let env ← getEnv
  let constantsMap := env.constants.map₁
  let const2ModIdx := env.const2ModIdx
  let moduleIdxs := moduleNames.filterMap env.getModuleIdx?

  -- Filter constants belonging to target modules
  let mut targetConstants : Array (Name × ConstantInfo) := #[]
  for (n, c) in constantsMap do
    if let some modIdx := const2ModIdx.get? n then
      if moduleIdxs.contains modIdx && !isBlackListedDecl n then
        targetConstants := targetConstants.push (n, c)

  -- Extract each constant
  for (name, cinfo) in targetConstants do
    let json ← MetaM.run' (extractConstantInfo name cinfo constantsMap)
    IO.println json.compress

/-! ## Main Entry Point -/

unsafe def main (args : List String) : IO UInt32 := do
  -- Parse module names from arguments
  let modules : Array Name := match args with
  | [] => #[`Mathlib]
  | args => args.toArray.map String.toName

  -- Import modules and extract
  unsafe enableInitializersExecution
  initSearchPath (← findSysroot)

  -- Use withImportModules to load the environment
  let imports : Array Import := modules.map fun m => { module := m : Import }
  withImportModules imports (opts := {}) (trustLevel := 1024) fun env => do
    let coreCtx : Core.Context := {
      fileName := "<ExtractConstants>",
      fileMap := default,
      maxHeartbeats := 0
    }
    let coreState : Core.State := { env := env }

    let (_, _) ← (extractModuleConstants modules).toIO coreCtx coreState
    return 0
