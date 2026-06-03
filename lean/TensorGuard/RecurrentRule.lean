/-
TensorGuard recurrent hidden-state contracts for `nn.RNN` / `nn.GRU` /
`nn.LSTM` (Step 237).

This file mechanizes the shape-only contract implemented by
`src/model_checker.py::_recurrent_output_shape` and
`::_recurrent_state_shape` for already-admitted recurrent calls:

  * sequence outputs preserve the input sequence/batch layout and replace the
    feature axis with `num_directions * H_out`;
  * `h_n` has shape `(num_directions * num_layers, H_out)` for unbatched input
    and `(num_directions * num_layers, batch, H_out)` for batched input;
  * `c_n` exists only for LSTM and uses `hidden_size`, even when a projection
    changes `H_out`;
  * `batch_first` changes only which input axis is copied into state tensors;
  * bidirectionality doubles the output feature axis and the state depth, not
    the per-state hidden feature.

The model is deliberately Nat/list based and assumes the Python front-end has
already rejected bad ranks, bad input_size, and invalid LSTM projection values.
Pure Lean 4 core (no mathlib).
-/

namespace TensorGuard
namespace RecurrentRule

inductive CellKind
  | rnn
  | gru
  | lstm
  deriving DecidableEq, Repr

inductive StateKind
  | hidden
  | cell
  deriving DecidableEq, Repr

structure RecurrentSpec where
  cell : CellKind
  inputSize : Nat
  hiddenSize : Nat
  numLayers : Nat
  batchFirst : Bool
  bidirectional : Bool
  projSize : Nat
  deriving DecidableEq, Repr

def directions (bidirectional : Bool) : Nat :=
  if bidirectional then 2 else 1

def depth (spec : RecurrentSpec) : Nat :=
  directions spec.bidirectional * spec.numLayers

def hiddenOut (spec : RecurrentSpec) : Nat :=
  match spec.cell with
  | CellKind.lstm => if 0 < spec.projSize then spec.projSize else spec.hiddenSize
  | _ => spec.hiddenSize

def outputFeature (spec : RecurrentSpec) : Nat :=
  directions spec.bidirectional * hiddenOut spec

def stateFeature (spec : RecurrentSpec) : StateKind → Option Nat
  | StateKind.hidden => some (hiddenOut spec)
  | StateKind.cell =>
      match spec.cell with
      | CellKind.lstm => some spec.hiddenSize
      | _ => none

def outputShape (spec : RecurrentSpec) : List Nat → Option (List Nat)
  | [seq, input] =>
      if input == spec.inputSize then
        some [seq, outputFeature spec]
      else
        none
  | [d0, d1, input] =>
      if input == spec.inputSize then
        some [d0, d1, outputFeature spec]
      else
        none
  | _ => none

def stateShape
    (spec : RecurrentSpec) (state : StateKind) : List Nat → Option (List Nat)
  | [_seq, input] =>
      if input == spec.inputSize then
        match stateFeature spec state with
        | some feat => some [depth spec, feat]
        | none => none
      else
        none
  | [d0, d1, input] =>
      if input == spec.inputSize then
        match stateFeature spec state with
        | some feat =>
            let batch := if spec.batchFirst then d0 else d1
            some [depth spec, batch, feat]
        | none => none
      else
        none
  | _ => none

/- ========================================================================== -/
/- Generic transformation laws                                                -/
/- ========================================================================== -/

theorem batch_first_output_preserves_layout
    (cell : CellKind) (input hidden layers proj batch seq : Nat)
    (bidir : Bool) :
    outputShape
      {
        cell := cell,
        inputSize := input,
        hiddenSize := hidden,
        numLayers := layers,
        batchFirst := true,
        bidirectional := bidir,
        projSize := proj
      }
      [batch, seq, input]
      =
      some [
        batch,
        seq,
        outputFeature {
          cell := cell,
          inputSize := input,
          hiddenSize := hidden,
          numLayers := layers,
          batchFirst := true,
          bidirectional := bidir,
          projSize := proj
        }
      ] := by
  simp [outputShape]

theorem time_major_output_preserves_layout
    (cell : CellKind) (input hidden layers proj seq batch : Nat)
    (bidir : Bool) :
    outputShape
      {
        cell := cell,
        inputSize := input,
        hiddenSize := hidden,
        numLayers := layers,
        batchFirst := false,
        bidirectional := bidir,
        projSize := proj
      }
      [seq, batch, input]
      =
      some [
        seq,
        batch,
        outputFeature {
          cell := cell,
          inputSize := input,
          hiddenSize := hidden,
          numLayers := layers,
          batchFirst := false,
          bidirectional := bidir,
          projSize := proj
        }
      ] := by
  simp [outputShape]

theorem batch_first_state_selects_dim0
    (cell : CellKind) (input hidden layers proj batch seq : Nat) :
    stateShape
      {
        cell := cell,
        inputSize := input,
        hiddenSize := hidden,
        numLayers := layers,
        batchFirst := true,
        bidirectional := false,
        projSize := proj
      }
      StateKind.hidden
      [batch, seq, input]
      =
      some [
        layers,
        batch,
        hiddenOut {
          cell := cell,
          inputSize := input,
          hiddenSize := hidden,
          numLayers := layers,
          batchFirst := true,
          bidirectional := false,
          projSize := proj
        }
      ] := by
  cases cell <;> simp [stateShape, stateFeature, depth, directions, hiddenOut]

theorem time_major_state_selects_dim1
    (cell : CellKind) (input hidden layers proj seq batch : Nat) :
    stateShape
      {
        cell := cell,
        inputSize := input,
        hiddenSize := hidden,
        numLayers := layers,
        batchFirst := false,
        bidirectional := false,
        projSize := proj
      }
      StateKind.hidden
      [seq, batch, input]
      =
      some [
        layers,
        batch,
        hiddenOut {
          cell := cell,
          inputSize := input,
          hiddenSize := hidden,
          numLayers := layers,
          batchFirst := false,
          bidirectional := false,
          projSize := proj
        }
      ] := by
  cases cell <;> simp [stateShape, stateFeature, depth, directions, hiddenOut]

theorem bidirectional_output_feature_doubles
    (cell : CellKind) (input hidden layers proj seq batch : Nat) :
    outputShape
      {
        cell := cell,
        inputSize := input,
        hiddenSize := hidden,
        numLayers := layers,
        batchFirst := false,
        bidirectional := true,
        projSize := proj
      }
      [seq, batch, input]
      =
      some [
        seq,
        batch,
        2 * hiddenOut {
          cell := cell,
          inputSize := input,
          hiddenSize := hidden,
          numLayers := layers,
          batchFirst := false,
          bidirectional := true,
          projSize := proj
        }
      ] := by
  cases cell <;> simp [outputShape, outputFeature, directions, hiddenOut]

theorem bidirectional_state_depth_doubles
    (cell : CellKind) (input hidden layers proj batch seq : Nat) :
    stateShape
      {
        cell := cell,
        inputSize := input,
        hiddenSize := hidden,
        numLayers := layers,
        batchFirst := true,
        bidirectional := true,
        projSize := proj
      }
      StateKind.hidden
      [batch, seq, input]
      =
      some [
        2 * layers,
        batch,
        hiddenOut {
          cell := cell,
          inputSize := input,
          hiddenSize := hidden,
          numLayers := layers,
          batchFirst := true,
          bidirectional := true,
          projSize := proj
        }
      ] := by
  cases cell <;> simp [stateShape, stateFeature, depth, directions, hiddenOut]

theorem lstm_cell_state_uses_hidden_size_under_projection
    (input hidden layers proj batch seq : Nat) :
    stateShape
      {
        cell := CellKind.lstm,
        inputSize := input,
        hiddenSize := hidden,
        numLayers := layers,
        batchFirst := true,
        bidirectional := true,
        projSize := proj
      }
      StateKind.cell
      [batch, seq, input]
      =
      some [2 * layers, batch, hidden] := by
  simp [stateShape, stateFeature, depth, directions]

theorem gru_cell_state_rejected
    (input hidden layers proj batch seq : Nat) :
    stateShape
      {
        cell := CellKind.gru,
        inputSize := input,
        hiddenSize := hidden,
        numLayers := layers,
        batchFirst := true,
        bidirectional := true,
        projSize := proj
      }
      StateKind.cell
      [batch, seq, input]
      =
      none := by
  simp [stateShape, stateFeature]

theorem rnn_cell_state_rejected
    (input hidden layers proj batch seq : Nat) :
    stateShape
      {
        cell := CellKind.rnn,
        inputSize := input,
        hiddenSize := hidden,
        numLayers := layers,
        batchFirst := true,
        bidirectional := true,
        projSize := proj
      }
      StateKind.cell
      [batch, seq, input]
      =
      none := by
  simp [stateShape, stateFeature]

/- ========================================================================== -/
/- Executable theorem-shaped examples                                         -/
/- ========================================================================== -/

def projectedBiLSTM : RecurrentSpec :=
  {
    cell := CellKind.lstm,
    inputSize := 11,
    hiddenSize := 9,
    numLayers := 2,
    batchFirst := true,
    bidirectional := true,
    projSize := 5
  }

theorem projected_bilstm_output_shape :
    outputShape projectedBiLSTM [3, 5, 11] = some [3, 5, 10] := by
  decide

theorem projected_bilstm_h_state_shape :
    stateShape projectedBiLSTM StateKind.hidden [3, 5, 11] =
      some [4, 3, 5] := by
  decide

theorem projected_bilstm_c_state_shape :
    stateShape projectedBiLSTM StateKind.cell [3, 5, 11] =
      some [4, 3, 9] := by
  decide

def timeMajorBiGRU : RecurrentSpec :=
  {
    cell := CellKind.gru,
    inputSize := 8,
    hiddenSize := 6,
    numLayers := 3,
    batchFirst := false,
    bidirectional := true,
    projSize := 0
  }

theorem time_major_bigru_output_shape :
    outputShape timeMajorBiGRU [7, 4, 8] = some [7, 4, 12] := by
  decide

theorem time_major_bigru_h_state_shape :
    stateShape timeMajorBiGRU StateKind.hidden [7, 4, 8] =
      some [6, 4, 6] := by
  decide

def unbatchedRNN : RecurrentSpec :=
  {
    cell := CellKind.rnn,
    inputSize := 6,
    hiddenSize := 5,
    numLayers := 2,
    batchFirst := false,
    bidirectional := true,
    projSize := 0
  }

theorem unbatched_rnn_output_shape :
    outputShape unbatchedRNN [9, 6] = some [9, 10] := by
  decide

theorem unbatched_rnn_h_state_shape :
    stateShape unbatchedRNN StateKind.hidden [9, 6] = some [4, 5] := by
  decide

theorem wrong_input_size_rejected :
    outputShape projectedBiLSTM [3, 5, 12] = none := by
  decide

theorem bad_rank_rejected :
    outputShape projectedBiLSTM [3, 5, 11, 1] = none := by
  decide

end RecurrentRule
end TensorGuard
