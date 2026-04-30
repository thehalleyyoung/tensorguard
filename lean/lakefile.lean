import Lake
open Lake DSL

package tensorguard where

lean_lib TensorGuard where
  roots := #[`TensorGuard]

lean_exe parity_runner where
  root := `ParityRunner
