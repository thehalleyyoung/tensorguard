/-
Parity Runner: JSON-based executable for invoking Lean rules from Python.

Reads JSON from stdin: {"op": "matmul2", "args": [[3,4],[4,5]]}
Writes JSON to stdout: {"result": [3,5]} or {"error": "..."}

Supports a subset of operators for the subprocess approach (5-8 ops).
-/

import TensorGuard.Soundness
import TensorGuard.Extended
import TensorGuard.Parity
import Lean.Data.Json

open Lean (Json FromJson ToJson)
open TensorGuard

/-! ## JSON encoding/decoding -/

instance : ToJson (List Nat) where
  toJson ns := Json.arr (ns.toArray.map (fun n => Json.num (Int.ofNat n)))

instance : FromJson (List Nat) where
  fromJson? j := do
    let arr ← j.getArr?
    let nats ← arr.mapM (·.getNat?)
    return nats.toList

def shapeToList : Shape → List Nat
  | .nil => []
  | .cons n s => n :: shapeToList s

/-! ## Operator dispatch -/

def runMatmul2 (args : List (List Nat)) : Except String (List Nat) :=
  match args with
  | [s1, s2] =>
      match s1, s2 with
      | [m, k1], [k2, n] =>
          let shape1 := Shape.cons m (Shape.cons k1 Shape.nil)
          let shape2 := Shape.cons k2 (Shape.cons n Shape.nil)
          match matmul2 shape1 shape2 with
          | some result => Except.ok (shapeToList result)
          | none => Except.error "matmul2 failed: contraction dimension mismatch"
      | _, _ => Except.error "matmul2 requires rank-2 shapes"
  | _ => Except.error "matmul2 requires exactly 2 shape arguments"

def runBmm (args : List (List Nat)) : Except String (List Nat) :=
  match args with
  | [s1, s2] =>
      match s1, s2 with
      | [b1, m, k1], [b2, k2, n] =>
          let shape1 := Shape.cons b1 (Shape.cons m (Shape.cons k1 Shape.nil))
          let shape2 := Shape.cons b2 (Shape.cons k2 (Shape.cons n Shape.nil))
          match bmm shape1 shape2 with
          | some result => Except.ok (shapeToList result)
          | none => Except.error "bmm failed: batch or contraction dimension mismatch"
      | _, _ => Except.error "bmm requires rank-3 shapes"
  | _ => Except.error "bmm requires exactly 2 shape arguments"

def runTranspose2 (args : List (List Nat)) : Except String (List Nat) :=
  match args with
  | [s] =>
      match s with
      | [m, n] =>
          let shape := Shape.cons m (Shape.cons n Shape.nil)
          match transpose2 shape with
          | some result => Except.ok (shapeToList result)
          | none => Except.error "transpose2 failed"
      | _ => Except.error "transpose2 requires rank-2 shape"
  | _ => Except.error "transpose2 requires exactly 1 shape argument"

def runConv1dOut (args : List Nat) : Except String Nat :=
  match args with
  | [h_in, pad, dilation, k, stride] =>
      match conv1dOut h_in pad dilation k stride with
      | some out => Except.ok out
      | none => Except.error "conv1dOut failed"
  | _ => Except.error "conv1dOut requires 5 nat arguments"

def runConv2dOutH (args : List Nat) : Except String Nat :=
  match args with
  | [h_in, pad, dilation, k, stride] =>
      match conv2dOutH h_in pad dilation k stride with
      | some out => Except.ok out
      | none => Except.error "conv2dOutH failed"
  | _ => Except.error "conv2dOutH requires 5 nat arguments"

def runConv2dOutW (args : List Nat) : Except String Nat :=
  match args with
  | [w_in, pad, dilation, k, stride] =>
      match conv2dOutW w_in pad dilation k stride with
      | some out => Except.ok out
      | none => Except.error "conv2dOutW failed"
  | _ => Except.error "conv2dOutW requires 5 nat arguments"

def runMaxpool2dOutH (args : List Nat) : Except String Nat :=
  match args with
  | [h_in, pad, k, stride] =>
      match maxpool2dOutH h_in pad k stride with
      | some out => Except.ok out
      | none => Except.error "maxpool2dOutH failed"
  | _ => Except.error "maxpool2dOutH requires 4 nat arguments"

def runLinearShape (args : List (List Nat)) : Except String (List Nat) :=
  match args with
  | [shape, [in_feat, out_feat]] =>
      match linearShape shape in_feat out_feat with
      | some result => Except.ok result
      | none => Except.error "linearShape failed"
  | _ => Except.error "linearShape requires shape and [in_features, out_features]"

/-- Main dispatcher -/
def dispatchOp (op : String) (data : Json) : Except String Json := do
  match op with
  | "matmul2" =>
      let args ← FromJson.fromJson? (α := List (List Nat)) data
      let result ← runMatmul2 args
      return ToJson.toJson result
  | "bmm" =>
      let args ← FromJson.fromJson? (α := List (List Nat)) data
      let result ← runBmm args
      return ToJson.toJson result
  | "transpose2" =>
      let args ← FromJson.fromJson? (α := List (List Nat)) data
      let result ← runTranspose2 args
      return ToJson.toJson result
  | "conv1dOut" =>
      let args ← FromJson.fromJson? (α := List Nat) data
      let result ← runConv1dOut args
      return Json.num result
  | "conv2dOutH" =>
      let args ← FromJson.fromJson? (α := List Nat) data
      let result ← runConv2dOutH args
      return Json.num result
  | "conv2dOutW" =>
      let args ← FromJson.fromJson? (α := List Nat) data
      let result ← runConv2dOutW args
      return Json.num result
  | "maxpool2dOutH" =>
      let args ← FromJson.fromJson? (α := List Nat) data
      let result ← runMaxpool2dOutH args
      return Json.num result
  | "linearShape" =>
      let args ← FromJson.fromJson? (α := List (List Nat)) data
      let result ← runLinearShape args
      return ToJson.toJson result
  | _ => Except.error s!"Unknown operator: {op}"

/-! ## Main I/O loop -/

def main (args : List String) : IO Unit := do
  -- Read all input
  let mut lines := []
  repeat
    let line ← (← IO.getStdin).getLine
    if line.isEmpty then break
    lines := line :: lines
  let input := String.join lines.reverse
  
  match Json.parse input with
  | Except.error e =>
      IO.println (Json.mkObj [("error", Json.str s!"JSON parse error: {e}")])
  | Except.ok json =>
      match json.getObjVal? "op", json.getObjVal? "args" with
      | Except.ok op_json, Except.ok args_json =>
          match op_json.getStr? with
          | Except.ok op =>
              match dispatchOp op args_json with
              | Except.ok result =>
                  IO.println (Json.mkObj [("result", result)])
              | Except.error e =>
                  IO.println (Json.mkObj [("error", Json.str e)])
          | Except.error e =>
              IO.println (Json.mkObj [("error", Json.str s!"op field must be string: {e}")])
      | _, _ =>
          IO.println (Json.mkObj [("error", Json.str "Expected JSON with 'op' and 'args' fields")])
