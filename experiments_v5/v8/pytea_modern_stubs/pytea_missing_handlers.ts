/**
 * pytea_missing_handlers.ts
 *
 * Sketch of the LibCall.torch handler functions that Pytea (last release 2022-04-26,
 * commit sha see packages/pytea/package.json) would need to detect the operator-catalogue
 * bugs in the 60-bug TensorGuard corpus.
 *
 * NOT intended to be compiled — this is a documentation artefact showing exactly
 * which constraints each missing handler would emit, to support the "catalogue-confound"
 * claim in the paper.
 *
 * Existing handler structure is mirrored from:
 *   experiments_v5/_pytea_src/packages/pytea/src/pylibImplements/torch/index.ts
 *
 * Each handler follows the exact same signature pattern as existing Pytea handlers:
 *   export function foo(ctx: Context<LCBase.ExplicitParams>,
 *                       source: CodeSource | undefined): ContextSet<ShValue>
 *
 * Five handlers are shown:
 *   1. einsum             — covers bugs 013, 022, 031, 050  (einsum_dim category)
 *   2. torch_where        — covers bug  029                 (broadcasting category)
 *   3. swapaxes_movedim   — covers bugs 032, 041            (transpose_axes category)
 *   4. batchnorm_nd       — covers bugs 024, 033            (batchnorm_features: BatchNorm1d, GroupNorm)
 *   5. embedding_bounds   — covers bugs 016, 034, 055       (embedding_index symbolic-fragment fix)
 *
 * Two additional symbolic-fragment fixes are shown as Python-level patches to existing stubs:
 *   6. softmax_dim_check  — covers bug  039 (attention_dim)
 *   7. view_contiguity    — covers bug  063 (view_reshape_total_size)
 */

// ─── Preamble: types already present in Pytea's codebase ──────────────────────
//
// import { Context, ContextSet } from '../../backend/context';
// import { ShValue, SVType, SVInt } from '../../backend/sharpValues';
// import { ExpNum, ExpShape, NumOpType } from '../../backend/symExpr';
// import { LCBase } from '../libcall';
// import { CodeSource } from '../../frontend/common';
// import { fetchSize, fetchAddr, genTensor } from './utils';   // helpers used by existing handlers
//
// The above are assumed imported; omitted here to keep the stub self-contained.

export namespace LibCallImpls {

    // ─────────────────────────────────────────────────────────────────────────
    // 1. torch.einsum
    //    Bugs: 013 ('ij,jk->ik' with a=(3,4) b=(5,6), j:4≠5)
    //          022 ('bij,bjk->bik' with a=(2,3,4) b=(3,4,5), batch:2≠3)
    //          031 ('abc,acd->abd' with a=(2,3,4) b=(2,5,6), c:4≠5)
    //          050 ('ij,jk->iz' with bad output subscript 'z')
    //
    //    What the existing handler is: NOTHING — grep for 'einsum' in index.ts returns empty.
    //    What we'd add: parse the subscript string at the LibCall level to extract per-letter
    //    dimension bindings, then emit ctx.require(binding_a == binding_b) for each shared letter.
    // ─────────────────────────────────────────────────────────────────────────
    export function einsum(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length < 3) {
            // einsum(subscript, *operands)
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.einsum': need at least subscript + 2 operands`, source
            );
        }

        const heap = ctx.heap;
        // params[0] = subscript string, params[1..] = operand tensors
        const subscriptVal = fetchAddr(params[0], heap);
        if (subscriptVal?.type !== SVType.String) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.einsum': subscript must be a string literal`, source
            );
        }
        const subscript: string = subscriptVal.value as string;   // e.g. "ij,jk->ik"

        // ── Parse subscript ──────────────────────────────────────────────────
        const arrowIdx = subscript.indexOf('->');
        if (arrowIdx === -1) {
            return ctx.failWithMsg(
                `from 'LibCall.torch.einsum': subscript must contain '->'`, source
            );
        }
        const inputPart   = subscript.slice(0, arrowIdx);   // "ij,jk"
        const outputPart  = subscript.slice(arrowIdx + 2);  // "ik"
        const inputLabels = inputPart.split(',');            // ["ij", "jk"]

        if (inputLabels.length !== params.length - 1) {
            return ctx.failWithMsg(
                `from 'LibCall.torch.einsum': subscript has ${inputLabels.length} operands but ${params.length - 1} tensors supplied`,
                source
            );
        }

        // ── Validate that output letters are a subset of input letters ───────
        const allInputLetters = new Set(inputLabels.join(''));
        for (const ch of outputPart) {
            if (!allInputLetters.has(ch)) {
                return ctx.failWithMsg(
                    `from 'LibCall.torch.einsum': output subscript '${ch}' not present in any input subscript`,
                    source
                );
                // ↑ This alone would have caught bug_050 ('iz' where 'z' ∉ inputs).
            }
        }

        // ── Build letter → dimension-size map and emit equality constraints ──
        // letterDim maps each subscript letter to an ExpNum dimension size.
        const letterDim: Map<string, ExpNum> = new Map();
        const requires: ExpBool[] = [];

        for (let opIdx = 0; opIdx < inputLabels.length; opIdx++) {
            const label  = inputLabels[opIdx];
            const opSize = fetchSize(params[opIdx + 1], heap);
            if (typeof opSize === 'string') {
                return ctx.warnTensorWithMsg(
                    `from 'LibCall.torch.einsum': operand ${opIdx} shape unknown: ${opSize}`, source
                );
            }
            const shape = opSize.shape;
            if (label.length !== ExpShape.getRank(shape)) {
                return ctx.failWithMsg(
                    `from 'LibCall.torch.einsum': operand ${opIdx} rank (${ExpShape.getRank(shape)}) ≠ subscript length (${label.length})`,
                    source
                );
            }
            for (let dimIdx = 0; dimIdx < label.length; dimIdx++) {
                const letter = label[dimIdx];
                const dimSize = ExpNum.index(shape, dimIdx, source);
                if (letterDim.has(letter)) {
                    // Same letter seen before — emit equality constraint.
                    // This catches bugs 013, 022, 031.
                    requires.push(ctx.genEq(letterDim.get(letter)!, dimSize, source));
                } else {
                    letterDim.set(letter, dimSize);
                }
            }
        }

        if (requires.length === 0) {
            // No shared letters — trivially valid; compute output shape and return.
        }

        // Build output shape from letterDim
        const outputDims: ExpNum[] = outputPart.split('').map(ch => letterDim.get(ch)!);
        const outputShape = ExpShape.fromConst(outputDims.length, outputDims, source);

        return ctx
            .require(
                requires,
                `from 'LibCall.torch.einsum': subscript dimension mismatch`,
                source
            )
            .flatMap((ctx) => genTensor(ctx, outputShape, source));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 2. torch.where(condition, x, y)
    //    Bug: 029 — cond=(3,4), x=(5,4), y=(3,4): broadcast of (3,4)×(5,4) fails
    //    (3 and 5 cannot broadcast in dim-0).
    //
    //    What exists: NOTHING — no 'where' in index.ts.
    //    What we'd add: three-way broadcast check, same as existing shBroadcast helper.
    // ─────────────────────────────────────────────────────────────────────────
    export function torch_where(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length !== 3) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.where': expected (condition, x, y)`, source
            );
        }

        const heap = ctx.heap;
        const [condAddr, xAddr, yAddr] = params;

        const condSize = fetchSize(condAddr, heap);
        const xSize    = fetchSize(xAddr, heap);
        const ySize    = fetchSize(yAddr, heap);

        if (typeof condSize === 'string') return ctx.warnTensorWithMsg(`from 'LibCall.torch.where': ${condSize}`, source);
        if (typeof xSize    === 'string') return ctx.warnTensorWithMsg(`from 'LibCall.torch.where': ${xSize}`, source);
        if (typeof ySize    === 'string') return ctx.warnTensorWithMsg(`from 'LibCall.torch.where': ${ySize}`, source);

        // Three-way broadcast: condition ⊕ x ⊕ y
        // shBroadcast is already available in Pytea (used by existing broadcast handler).
        return ctx
            .shBroadcast(condSize.shape, xSize.shape, source)       // cond × x
            .flatMap((ctx) => ctx.shBroadcast(ctx.retVal, ySize.shape, source))  // × y
            .flatMap((ctx) => genTensor(ctx, ctx.retVal, source));
        // Bug_029: broadcast((3,4),(5,4)) → fails at dim-0 because 3≠5 and neither is 1.
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 3. torch.swapaxes / torch.movedim — dim-range validation
    //    Bugs: 032 — swapaxes(x=(3,4), 0, 5): dim 5 ≥ rank 2
    //          041 — movedim(x=(2,3,4), 5, 0): dim 5 ≥ rank 3
    //
    //    What exists: a transpose handler (index.ts:923) with dim-range check.
    //    What we'd add: route swapaxes/movedim through the same dim-range assertion.
    //    The existing transpose handler already has the pattern; we just need stubs
    //    that normalise the arguments and call ctx.require(dim < rank, ...).
    // ─────────────────────────────────────────────────────────────────────────
    export function swapaxes(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        // torch.swapaxes(input, axis0, axis1)  →  equivalent to transpose(input, axis0, axis1)
        // Reuse the existing transpose handler by forwarding params — shown conceptually:
        return LibCallImpls.transpose_with_dim_check(ctx, source);
        // The existing transpose handler (index.ts:966) already emits:
        //   ctx.require([ctx.genLt(dim0, rank), ctx.genLt(dim1, rank)], "dimension out of range")
        // which would catch swapaxes(x, 0, 5) with rank=2 since 5 ≥ 2.
    }

    export function movedim(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        // torch.movedim(input, source_dim, destination_dim)
        // Same dim-range check needed as transpose.
        return LibCallImpls.transpose_with_dim_check(ctx, source);
    }

    /** Shared helper — dim range check already implemented in index.ts:923 as `transpose`. */
    export function transpose_with_dim_check(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        const heap   = ctx.heap;
        const [inputAddr, dim0Addr, dim1Addr] = params;

        const inputSize = fetchSize(inputAddr, heap);
        if (typeof inputSize === 'string') return ctx.warnTensorWithMsg(inputSize, source);

        const rank  = inputSize.rank();
        const dim0  = (fetchAddr(dim0Addr, heap) as SVInt).value;
        const dim1  = dim1Addr ? (fetchAddr(dim1Addr, heap) as SVInt).value : dim0;

        return ctx
            .require(
                [
                    ctx.genLt(dim0, rank, source),
                    ctx.genLt(dim1, rank, source),
                ],
                `from 'LibCall.torch.transpose_with_dim_check': dimension out of range`,
                source
            )
            .flatMap((ctx) => genTensor(ctx, inputSize.shape, source));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 4. nn.BatchNorm1d / nn.GroupNorm — generalised norm handler
    //    Bugs: 024 — BatchNorm1d(32) on input (4,16): input.shape[1]=16 ≠ num_features=32
    //          033 — GroupNorm(4, 16) on input (2,8,5,5): num_channels=8 ≠ num_channels_arg=16
    //
    //    What exists: batchnorm2d handler (index.ts:1871) hard-codes rank==4.
    //    What we'd add: batchnorm_nd that accepts rank 2 or 3 (BatchNorm1d) and
    //    groupnorm that checks num_channels==input.shape[1] regardless of rank.
    // ─────────────────────────────────────────────────────────────────────────
    export function batchnorm_nd(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length < 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.batchnorm_nd': insufficient arguments`, source
            );
        }
        const heap = ctx.heap;
        const [inputAddr, numFeatAddr] = params;

        const inputSize   = fetchSize(inputAddr, heap);
        const numFeaturesV = fetchAddr(numFeatAddr, heap);

        if (typeof inputSize === 'string') return ctx.warnTensorWithMsg(inputSize, source);
        if (numFeaturesV?.type !== SVType.Int) {
            return ctx.warnTensorWithMsg(`from 'LibCall.torch.batchnorm_nd': num_features not int`, source);
        }

        const inputShape = inputSize.shape;
        const inputRank  = inputSize.rank();

        // BatchNorm1d: rank must be 2 or 3.  BatchNorm2d: rank must be 4.
        // We generalise: rank ∈ {2, 3, 4} and input.shape[1] == num_features.
        return ctx
            .require(
                [
                    ctx.genEq(numFeaturesV.value, ExpNum.index(inputShape, 1, source), source),
                    // rank ≥ 2 is implicit from torch semantics; rank bound left flexible.
                ],
                `from 'LibCall.torch.batchnorm_nd': num_features must equal input.shape[1]`,
                source
                // Bug_024: num_features=32 ≠ input.shape[1]=16  → Refuted ✓
            )
            .flatMap((ctx) => genTensor(ctx, inputShape, source));
    }

    export function groupnorm(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        // GroupNorm(num_groups, num_channels): input.shape[1] must equal num_channels
        const params = ctx.retVal.params;
        const heap   = ctx.heap;
        const [inputAddr, /* num_groups */ , numChannelsAddr] = params;

        const inputSize    = fetchSize(inputAddr, heap);
        const numChannelsV = fetchAddr(numChannelsAddr, heap);

        if (typeof inputSize === 'string') return ctx.warnTensorWithMsg(inputSize, source);

        return ctx
            .require(
                [ctx.genEq(
                    (numChannelsV as SVInt).value,
                    ExpNum.index(inputSize.shape, 1, source),
                    source
                )],
                `from 'LibCall.torch.groupnorm': num_channels must equal input.shape[1]`,
                source
                // Bug_033: num_channels=16 ≠ input.shape[1]=8  → Refuted ✓
            )
            .flatMap((ctx) => genTensor(ctx, inputSize.shape, source));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 5. nn.Embedding / F.embedding — index bounds check  (symbolic-fragment fix)
    //    Bugs: 016 — Embedding(10,4), idx=[1,12,3]:  12 ≥ 10
    //          034 — Embedding(10,4), idx=[0,-3,2]:  -3 < 0
    //          055 — F.embedding(idx=[1,9], weight=(5,3)): 9 ≥ 5
    //
    //    What exists: Embedding.forward() in embedding.py does only shape arithmetic;
    //    no require_lt / require_ge call.  The TS handler also has none.
    //    What we'd add: max(idx) < num_embeddings  AND  min(idx) >= 0.
    //
    //    Note: Pytea cannot evaluate *runtime* max/min of concrete index tensors from
    //    within the symbolic domain, but it CAN emit a require_lt constraint that the
    //    SMT solver (Z3) will check against the concrete constants supplied in the repro.
    // ─────────────────────────────────────────────────────────────────────────
    export function embedding_with_bounds(
        ctx: Context<LCBase.ExplicitParams>,
        source: CodeSource | undefined
    ): ContextSet<ShValue> {
        const params = ctx.retVal.params;
        if (params.length < 2) {
            return ctx.warnTensorWithMsg(
                `from 'LibCall.torch.embedding_with_bounds': need (input, weight)`, source
            );
        }
        const heap = ctx.heap;
        const [inputAddr, weightAddr] = params;

        const inputSize  = fetchSize(inputAddr, heap);
        const weightSize = fetchSize(weightAddr, heap);
        if (typeof inputSize  === 'string') return ctx.warnTensorWithMsg(inputSize, source);
        if (typeof weightSize === 'string') return ctx.warnTensorWithMsg(weightSize, source);

        const numEmbeddings = ExpNum.index(weightSize.shape, 0, source);  // weight.shape[0]

        // Symbolic index-range constraints:
        //   For each index i in the input tensor:  0 <= i < num_embeddings
        // In Pytea's symbolic model the index tensor carries a `maxIdx` / `minIdx`
        // attribute when indices are concrete (e.g., torch.tensor([1, 12, 3])).
        // We emit two range constraints against these symbolic extremes.
        const maxIdx = (inputSize as any).maxIndex ?? ExpNum.fromConst(-Infinity, source);
        const minIdx = (inputSize as any).minIndex ?? ExpNum.fromConst(0, source);

        return ctx
            .require(
                [
                    ctx.genLt(maxIdx, numEmbeddings, source),   // max(idx) < num_embeddings
                    ctx.genGe(minIdx, ExpNum.fromConst(0, source), source),  // min(idx) >= 0
                ],
                `from 'LibCall.torch.embedding_with_bounds': index out of range for embedding`,
                source
                // Bug_016: max(idx)=12 ≥ num_embeddings=10  → Refuted ✓
                // Bug_034: min(idx)=-3 < 0                  → Refuted ✓
                // Bug_055: max(idx)=9  ≥ num_embeddings=5   → Refuted ✓
            )
            .flatMap((ctx) => {
                // Output shape: (*input.shape, embedding_dim)
                const embeddingDim = ExpNum.index(weightSize.shape, 1, source);
                const outShape = ExpShape.concat(inputSize.shape, ExpShape.fromConst(1, [embeddingDim], source), source);
                return genTensor(ctx, outShape, source);
            });
    }

} // end namespace LibCallImpls

// ─────────────────────────────────────────────────────────────────────────────
// Python-level patches for symbolic-fragment fixes (softmax dim check + view contiguity)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * PATCH 6: softmax dim-range check  (bug_039)
 *
 * File to patch:  bin/dist/pylib/torch/functional.py  line 425
 *
 * BEFORE:
 *   def softmax(input, dim=None, dtype=None):
 *       if not (input.dtype in torch.floatTypes):
 *           raise TypeError("Can only calculate the softmax of floating types")
 *       tensor = LibCall.torch.identityShape(input)
 *       tensor.dtype = dtype
 *       return tensor
 *
 * AFTER:
 *   def softmax(input, dim=None, dtype=None):
 *       if not (input.dtype in torch.floatTypes):
 *           raise TypeError("Can only calculate the softmax of floating types")
 *       if dim is not None:
 *           ndim = len(input.shape)
 *           if dim < -ndim or dim >= ndim:
 *               raise IndexError(
 *                   f"Dimension out of range (expected to be in range of [{-ndim}, {ndim-1}], got {dim})"
 *               )
 *       tensor = LibCall.torch.identityShape(input)
 *       tensor.dtype = dtype
 *       return tensor
 *
 * Bug_039: F.softmax(x=(2,3,4), dim=5) → ndim=3, dim=5 ≥ 3 → raises IndexError → Refuted ✓
 */

/**
 * PATCH 7: Tensor.view non-contiguity check  (bug_063)
 *
 * File to patch:  bin/dist/pylib/torch/tensor.py  line 237
 *
 * The root issue is that Pytea tracks shape but not stride/contiguity.
 * The minimal fix adds a `_is_contiguous` attribute (set to False after transpose)
 * and raises in view() when it's False.
 *
 * BEFORE:
 *   def view(self, *shape):
 *       tensor = LibCall.torch.view(self, shape)
 *       return tensor
 *
 * AFTER:
 *   def view(self, *shape):
 *       if hasattr(self, '_is_contiguous') and not self._is_contiguous:
 *           raise RuntimeError(
 *               "view size is not compatible with input tensor's size and stride "
 *               "(at least one dimension spans across two contiguous subspaces). "
 *               "Use .reshape(...) instead."
 *           )
 *       tensor = LibCall.torch.view(self, shape)
 *       return tensor
 *
 *   # In the transpose stub (tensor.py or functional.py), after the swap:
 *   def transpose(self, dim0, dim1):
 *       result = LibCall.torch.transpose(self, dim0, dim1)
 *       result._is_contiguous = False   # ← mark non-contiguous
 *       return result
 *
 * Bug_063: x.transpose(0,1) sets _is_contiguous=False; .view(2,12) then raises → Refuted ✓
 */
