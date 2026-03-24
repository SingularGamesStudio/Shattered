using System;
using GPU.Delaunay;
using GPU.Neighbors;
using UnityEngine;
using UnityEngine.Rendering;

namespace GPU.Solver {
    internal sealed class LayerSolveRuntime {
        private readonly XPBISolver solver;
        private readonly ComputeShader shader;
        private readonly ComputeShader solverDebugShader;

        internal LayerSolveRuntime(XPBISolver solver) {
            this.solver = solver;
            shader = solver.LayerSolveShader;
            solverDebugShader = solver.SolverDebugShader;
        }

        internal XPBISolver Solver => solver;

        private ComputeBuffer currentVolumeBits;
        private ComputeBuffer currentTotalMassBits;
        private ComputeBuffer fixedChildPosBits;
        private ComputeBuffer fixedChildCount;
        private ComputeBuffer L;
        private ComputeBuffer F0;
        private ComputeBuffer lambda;
        private ComputeBuffer durabilityLambda;
        private ComputeBuffer collisionLambda;
        private ComputeBuffer savedVelPrefix;
        private ComputeBuffer velDeltaBits;
        private ComputeBuffer velPrev;
        private ComputeBuffer lambdaPrev;
        private ComputeBuffer jrVelDeltaBits;
        private ComputeBuffer jrLambdaDelta;
        private ComputeBuffer coarseFixed;
        private ComputeBuffer restrictedDeltaVBits;
        private ComputeBuffer restrictedDeltaVCount;
        private ComputeBuffer restrictedDeltaVAvg;

        internal int kRelaxColored;
        internal int kRelaxColoredPersistentCoarse;
        internal int kJRSavePrevAndClear;
        internal int kJRComputeDeltas;
        internal int kJRApply;
        internal int kProlongate;
        internal int kCommitDeformation;
        internal int kSmoothProlongatedFineVel;
        internal int kCopyVelToPrev;
        internal int kApplyXsph;
        internal int kApplyPositionCorrection;

        private ComputeBuffer pos => solver.pos;
        private ComputeBuffer vel => solver.vel;
        private ComputeBuffer materialIds => solver.materialIds;
        private ComputeBuffer invMass => solver.invMass;
        private ComputeBuffer restVolume => solver.restVolume;
        private ComputeBuffer parentIndex => solver.parentIndex;
        private ComputeBuffer parentIndices => solver.parentIndices;
        private ComputeBuffer parentWeights => solver.parentWeights;
        private ComputeBuffer F => solver.F;
        private ComputeBuffer Fp => solver.Fp;
        private ComputeBuffer fineContacts => solver.collisionEvent.CollisionEventsBuffer;
        private ComputeBuffer fineContactCount => solver.collisionEvent.CollisionEventCountBuffer;
        private ComputeBuffer coarseContacts => solver.collisionEvent.CoarseContactsBuffer;
        private ComputeBuffer coarseContactCount => solver.collisionEvent.CoarseContactCountBuffer;
        private ComputeBuffer boundaryEdgeV0Gi => solver.collisionEvent.BoundaryEdgeV0Buffer;
        private ComputeBuffer boundaryEdgeV1Gi => solver.collisionEvent.BoundaryEdgeV1Buffer;
        private ComputeBuffer boundaryVertexGi => solver.collisionEvent.BoundaryVertexGiBuffer;
        private ComputeBuffer xferColCount => solver.collisionEvent.XferColCountBuffer;
        private ComputeBuffer xferColNXBits => solver.collisionEvent.XferColNXBitsBuffer;
        private ComputeBuffer xferColNYBits => solver.collisionEvent.XferColNYBitsBuffer;
        private ComputeBuffer xferColPenBits => solver.collisionEvent.XferColPenBitsBuffer;
        private ComputeBuffer xferColSBits => solver.collisionEvent.XferColSBitsBuffer;
        private ComputeBuffer xferColTBits => solver.collisionEvent.XferColTBitsBuffer;
        private ComputeBuffer xferColQAGi => solver.collisionEvent.XferColQAGiBuffer;
        private ComputeBuffer xferColQBGi => solver.collisionEvent.XferColQBGiBuffer;
        private ComputeBuffer xferColOAGi => solver.collisionEvent.XferColOAGiBuffer;
        private ComputeBuffer xferColOBGi => solver.collisionEvent.XferColOBGiBuffer;
        private ComputeBuffer defaultDtOwnerByLocal => solver.layerMappingCache.DefaultDtOwnerByLocal;
        private ComputeBuffer defaultDtCollisionOwnerByLocal => solver.layerMappingCache.DefaultDtCollisionOwnerByLocal;

        private void BindDtGlobalMappingParams(CommandBuffer cb, int kernel, bool useDtGlobalNodeMap, int dtLocalBase, ComputeBuffer dtGlobalNodeMap, ComputeBuffer dtGlobalToLayerLocalMap) {
            solver.layerMappingCache.BindDtGlobalMappingParams(cb, shader, kernel, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
        }

        internal ComputeBuffer CurrentVolumeBits => currentVolumeBits;
        internal ComputeBuffer CurrentTotalMassBits => currentTotalMassBits;
        internal ComputeBuffer FixedChildPosBits => fixedChildPosBits;
        internal ComputeBuffer FixedChildCount => fixedChildCount;
        internal ComputeBuffer CorrectionL => L;
        internal ComputeBuffer CachedF0 => F0;
        internal ComputeBuffer Lambda => lambda;
        internal ComputeBuffer DurabilityLambda => durabilityLambda;
        internal ComputeBuffer CollisionLambda => collisionLambda;
        internal ComputeBuffer SavedVelPrefix => savedVelPrefix;
        internal ComputeBuffer VelDeltaBits => velDeltaBits;
        internal ComputeBuffer VelPrev => velPrev;
        internal ComputeBuffer LambdaPrev => lambdaPrev;
        internal ComputeBuffer JRVelDeltaBits => jrVelDeltaBits;
        internal ComputeBuffer JRLambdaDelta => jrLambdaDelta;
        internal ComputeBuffer CoarseFixed => coarseFixed;
        internal ComputeBuffer RestrictedDeltaVBits => restrictedDeltaVBits;
        internal ComputeBuffer RestrictedDeltaVCount => restrictedDeltaVCount;
        internal ComputeBuffer RestrictedDeltaVAvg => restrictedDeltaVAvg;

        public struct DtMappingContext {
            public readonly bool UseDtGlobalNodeMap;
            public readonly int DtLocalBase;
            public readonly ComputeBuffer DtGlobalNodeMap;
            public readonly ComputeBuffer DtGlobalToLayerLocalMap;
            public readonly ComputeBuffer DtOwnerByLocal;
            public readonly ComputeBuffer DtCollisionOwnerByLocal;

            public DtMappingContext(
                bool useDtGlobalNodeMap,
                int dtLocalBase,
                ComputeBuffer dtGlobalNodeMap,
                ComputeBuffer dtGlobalToLayerLocalMap,
                ComputeBuffer dtOwnerByLocal,
                ComputeBuffer dtCollisionOwnerByLocal
            ) {
                UseDtGlobalNodeMap = useDtGlobalNodeMap;
                DtLocalBase = dtLocalBase;
                DtGlobalNodeMap = dtGlobalNodeMap;
                DtGlobalToLayerLocalMap = dtGlobalToLayerLocalMap;
                DtOwnerByLocal = dtOwnerByLocal;
                DtCollisionOwnerByLocal = dtCollisionOwnerByLocal;
            }
        }

        internal readonly struct RelaxBufferContext {
            public readonly INeighborSearch NeighborSearch;
            public readonly int BaseIndex;
            public readonly int ActiveCount;
            public readonly int FineCount;
            public readonly int TickIndex;
            public readonly float LayerKernelH;
            public readonly DtMappingContext Mapping;

            public RelaxBufferContext(
                INeighborSearch neighborSearch,
                int baseIndex,
                int activeCount,
                int fineCount,
                int tickIndex,
                float layerKernelH,
                DtMappingContext mapping
            ) {
                NeighborSearch = neighborSearch;
                BaseIndex = baseIndex;
                ActiveCount = activeCount;
                FineCount = fineCount;
                TickIndex = tickIndex;
                LayerKernelH = layerKernelH;
                Mapping = mapping;
            }
        }

        internal int KRelaxColored => kRelaxColored;
        internal int KRelaxColoredPersistentCoarse => kRelaxColoredPersistentCoarse;
        internal int KJRSavePrevAndClear => kJRSavePrevAndClear;
        internal int KJRComputeDeltas => kJRComputeDeltas;
        internal int KJRApply => kJRApply;
        internal int KProlongate => kProlongate;
        internal int KCommitDeformation => kCommitDeformation;
        internal int KSmoothProlongatedFineVel => kSmoothProlongatedFineVel;
        internal int KCopyVelToPrev => kCopyVelToPrev;
        internal int KApplyXsph => kApplyXsph;
        internal int KApplyPositionCorrection => kApplyPositionCorrection;

        internal void AllocateRuntimeBuffers(int newCapacity) {
            int collisionEventCapacity = Math.Max(4096, newCapacity * 32);
            int coarseContactCapacity = newCapacity * Const.NeighborCount * Const.CollisionTransferManifoldSlots;
            int collisionLambdaCapacity = collisionEventCapacity + coarseContactCapacity;

            currentVolumeBits = new ComputeBuffer(newCapacity, sizeof(uint), ComputeBufferType.Structured);
            currentTotalMassBits = new ComputeBuffer(newCapacity, sizeof(uint), ComputeBufferType.Structured);
            fixedChildPosBits = new ComputeBuffer(newCapacity * 2, sizeof(uint), ComputeBufferType.Structured);
            fixedChildCount = new ComputeBuffer(newCapacity, sizeof(uint), ComputeBufferType.Structured);
            L = new ComputeBuffer(newCapacity, sizeof(float) * 4, ComputeBufferType.Structured);
            F0 = new ComputeBuffer(newCapacity, sizeof(float) * 4, ComputeBufferType.Structured);
            lambda = new ComputeBuffer(newCapacity, sizeof(float), ComputeBufferType.Structured);
            durabilityLambda = new ComputeBuffer(newCapacity * Const.NeighborCount, sizeof(float), ComputeBufferType.Structured);
            collisionLambda = new ComputeBuffer(collisionLambdaCapacity, sizeof(float), ComputeBufferType.Structured);
            savedVelPrefix = new ComputeBuffer(newCapacity, sizeof(float) * 2, ComputeBufferType.Structured);
            velDeltaBits = new ComputeBuffer(newCapacity * 2, sizeof(uint), ComputeBufferType.Structured);
            velPrev = new ComputeBuffer(newCapacity, sizeof(float) * 2, ComputeBufferType.Structured);
            lambdaPrev = new ComputeBuffer(newCapacity, sizeof(float), ComputeBufferType.Structured);
            jrVelDeltaBits = new ComputeBuffer(newCapacity * 2, sizeof(uint), ComputeBufferType.Structured);
            jrLambdaDelta = new ComputeBuffer(newCapacity, sizeof(float), ComputeBufferType.Structured);
            coarseFixed = new ComputeBuffer(newCapacity, sizeof(uint), ComputeBufferType.Structured);
            restrictedDeltaVBits = new ComputeBuffer(newCapacity * 2, sizeof(uint), ComputeBufferType.Structured);
            restrictedDeltaVCount = new ComputeBuffer(newCapacity, sizeof(uint), ComputeBufferType.Structured);
            restrictedDeltaVAvg = new ComputeBuffer(newCapacity, sizeof(float) * 2, ComputeBufferType.Structured);
        }

        internal void ReleaseRuntimeBuffers() {
            currentVolumeBits?.Dispose(); currentVolumeBits = null;
            currentTotalMassBits?.Dispose(); currentTotalMassBits = null;
            fixedChildPosBits?.Dispose(); fixedChildPosBits = null;
            fixedChildCount?.Dispose(); fixedChildCount = null;
            L?.Dispose(); L = null;
            F0?.Dispose(); F0 = null;
            lambda?.Dispose(); lambda = null;
            durabilityLambda?.Dispose(); durabilityLambda = null;
            collisionLambda?.Dispose(); collisionLambda = null;
            savedVelPrefix?.Dispose(); savedVelPrefix = null;
            velDeltaBits?.Dispose(); velDeltaBits = null;
            velPrev?.Dispose(); velPrev = null;
            lambdaPrev?.Dispose(); lambdaPrev = null;
            jrVelDeltaBits?.Dispose(); jrVelDeltaBits = null;
            jrLambdaDelta?.Dispose(); jrLambdaDelta = null;
            coarseFixed?.Dispose(); coarseFixed = null;
            restrictedDeltaVBits?.Dispose(); restrictedDeltaVBits = null;
            restrictedDeltaVCount?.Dispose(); restrictedDeltaVCount = null;
            restrictedDeltaVAvg?.Dispose(); restrictedDeltaVAvg = null;
        }

        internal void CacheRuntimeKernels() {
            kRelaxColored = shader.FindKernel("RelaxColored");
            kRelaxColoredPersistentCoarse = shader.FindKernel("RelaxColoredPersistentCoarse");
            kJRSavePrevAndClear = shader.FindKernel("JR_SavePrevAndClear");
            kJRComputeDeltas = shader.FindKernel("JR_ComputeDeltas");
            kJRApply = shader.FindKernel("JR_Apply");
            kProlongate = shader.FindKernel("Prolongate");
            kCommitDeformation = shader.FindKernel("CommitDeformation");
            kSmoothProlongatedFineVel = shader.FindKernel("SmoothProlongatedFineVel");
            kCopyVelToPrev = shader.FindKernel("CopyVelToPrev");
            kApplyXsph = shader.FindKernel("ApplyXsph");
            kApplyPositionCorrection = shader.FindKernel("ApplyPositionCorrection");
        }

        internal void BindLayerColoringBuffers(CommandBuffer cb, NeighborColoring layerColoring) {
            cb.SetComputeBufferParam(shader, kRelaxColored, "_ColorOrder", layerColoring.OrderBuffer);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_ColorStarts", layerColoring.StartsBuffer);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_ColorCounts", layerColoring.CountsBuffer);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_ColorOrder", layerColoring.OrderBuffer);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_ColorStarts", layerColoring.StartsBuffer);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_ColorCounts", layerColoring.CountsBuffer);
        }

        internal void SetConvergenceDebugDisabled(CommandBuffer cb) {
            cb.SetComputeIntParam(shader, "_ConvergenceDebugEnable", 0);
        }

        internal void SetCollisionEnable(CommandBuffer cb, bool enabled) {
            cb.SetComputeIntParam(shader, "_CollisionEnable", enabled ? 1 : 0);
        }

        internal void SetUseTransferredCollisionsParam(CommandBuffer cb, bool enabled) {
            cb.SetComputeIntParam(shader, "_UseTransferredCollisions", enabled ? 1 : 0);
        }

        internal void SetPersistentRelaxParams(CommandBuffer cb, int persistentIterations, int baseDebugIter) {
            cb.SetComputeIntParam(shader, "_PersistentIters", persistentIterations);
            cb.SetComputeIntParam(shader, "_PersistentBaseDebugIter", baseDebugIter);
        }

        internal void SetConvergenceDebugIter(CommandBuffer cb, int iter) {
            cb.SetComputeIntParam(shader, "_ConvergenceDebugIter", iter);
        }

        internal void SetColorIndex(CommandBuffer cb, int color) {
            cb.SetComputeIntParam(shader, "_ColorIndex", color);
        }

        internal void SetJRParams(CommandBuffer cb) {
            cb.SetComputeFloatParam(shader, "_JROmegaV", Const.JROmegaV);
            cb.SetComputeFloatParam(shader, "_JROmegaL", Const.JROmegaL);
        }

        internal void PrepareSolveBuffers(CommandBuffer cb, in RelaxBufferContext context) {
            INeighborSearch neighborSearch = context.NeighborSearch;
            int baseIndex = context.BaseIndex;
            int activeCount = context.ActiveCount;
            int fineCount = context.FineCount;
            float layerKernelH = context.LayerKernelH;
            bool useDtGlobalNodeMap = context.Mapping.UseDtGlobalNodeMap;
            int dtLocalBase = context.Mapping.DtLocalBase;
            ComputeBuffer dtGlobalNodeMap = context.Mapping.DtGlobalNodeMap;
            ComputeBuffer dtGlobalToLayerLocalMap = context.Mapping.DtGlobalToLayerLocalMap;
            ComputeBuffer dtOwnerByLocal = context.Mapping.DtOwnerByLocal;
            ComputeBuffer dtCollisionOwnerByLocal = context.Mapping.DtCollisionOwnerByLocal;
            var matLib = MaterialLibrary.Instance;
            var physicalParams = matLib != null ? matLib.PhysicalParamsBuffer : null;
            int physicalParamCount = (matLib != null && physicalParams != null) ? matLib.MaterialCount : 0;

            cb.SetComputeIntParam(shader, "_Base", baseIndex);
            cb.SetComputeIntParam(shader, "_ActiveCount", activeCount);
            cb.SetComputeIntParam(shader, "_FineCount", fineCount);
            cb.SetComputeIntParam(shader, "_DtNeighborCount", neighborSearch.NeighborCount);
            cb.SetComputeIntParam(shader, "_PhysicalParamCount", physicalParamCount);
            cb.SetComputeFloatParam(shader, "_LayerKernelH", layerKernelH);
            cb.SetComputeIntParam(shader, "_UseDtOwnerFilter", dtOwnerByLocal != null ? 1 : 0);
            cb.SetComputeIntParam(shader, "_CollisionEventCapacity", solver.collisionEvent.CollisionEventsBuffer != null ? solver.collisionEvent.CollisionEventsBuffer.count : 0);
            cb.SetComputeIntParam(shader, "_CoarseContactCapacity", solver.collisionEvent.CoarseContactsBuffer != null ? solver.collisionEvent.CoarseContactsBuffer.count : 0);

            ComputeBuffer convergenceDebugBinding = solver.solverDebug.ConvergenceDebug ?? solver.solverDebug.ProlongationConstraintDebug ?? solver.solverDebug.ConvergenceDebugFallback;
            cb.SetComputeBufferParam(shader, kRelaxColored, "_ConvergenceDebug", convergenceDebugBinding);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_ConvergenceDebug", convergenceDebugBinding);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_ConvergenceDebug", convergenceDebugBinding);
            cb.SetComputeBufferParam(solverDebugShader, solver.solverDebug.ClearConvergenceDebugStatsKernel, "_ConvergenceDebug", convergenceDebugBinding);

            cb.SetComputeBufferParam(shader, kRelaxColored, "_Pos", pos);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_Vel", vel);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_MaterialIds", materialIds);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_InvMass", invMass);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_RestVolume", restVolume);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_F0", F0);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_L", L);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_CurrentVolumeBits", currentVolumeBits);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_CurrentTotalMassBits", currentTotalMassBits);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_FixedChildPosBits", fixedChildPosBits);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_FixedChildCount", fixedChildCount);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_Lambda", lambda);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_DurabilityLambda", durabilityLambda);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_CollisionLambda", collisionLambda);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_FineContacts", fineContacts);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_FineContactCountBuffer", fineContactCount);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_CoarseContacts", coarseContacts);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_CoarseContactCountBuffer", coarseContactCount);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_BoundaryEdgeV0Gi", boundaryEdgeV0Gi);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_BoundaryEdgeV1Gi", boundaryEdgeV1Gi);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_BoundaryVertexGi", boundaryVertexGi);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_XferColCount", xferColCount);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_XferColNXBits", xferColNXBits);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_XferColNYBits", xferColNYBits);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_XferColPenBits", xferColPenBits);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_XferColSBits", xferColSBits);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_XferColTBits", xferColTBits);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_XferColQAGi", xferColQAGi);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_XferColQBGi", xferColQBGi);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_XferColOAGi", xferColOAGi);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_XferColOBGi", xferColOBGi);

            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_Pos", pos);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_Vel", vel);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_MaterialIds", materialIds);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_InvMass", invMass);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_RestVolume", restVolume);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_F0", F0);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_L", L);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_CurrentVolumeBits", currentVolumeBits);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_CurrentTotalMassBits", currentTotalMassBits);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_FixedChildPosBits", fixedChildPosBits);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_FixedChildCount", fixedChildCount);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_Lambda", lambda);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_DurabilityLambda", durabilityLambda);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_CollisionLambda", collisionLambda);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_FineContacts", fineContacts);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_FineContactCountBuffer", fineContactCount);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_CoarseContacts", coarseContacts);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_CoarseContactCountBuffer", coarseContactCount);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_BoundaryEdgeV0Gi", boundaryEdgeV0Gi);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_BoundaryEdgeV1Gi", boundaryEdgeV1Gi);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_BoundaryVertexGi", boundaryVertexGi);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_XferColCount", xferColCount);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_XferColNXBits", xferColNXBits);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_XferColNYBits", xferColNYBits);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_XferColPenBits", xferColPenBits);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_XferColSBits", xferColSBits);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_XferColTBits", xferColTBits);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_XferColQAGi", xferColQAGi);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_XferColQBGi", xferColQBGi);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_XferColOAGi", xferColOAGi);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_XferColOBGi", xferColOBGi);

            cb.SetComputeBufferParam(shader, kProlongate, "_InvMass", invMass);
            cb.SetComputeBufferParam(shader, kProlongate, "_Vel", vel);
            cb.SetComputeBufferParam(shader, kProlongate, "_Pos", pos);
            cb.SetComputeBufferParam(shader, kProlongate, "_MaterialIds", materialIds);
            cb.SetComputeBufferParam(shader, kProlongate, "_RestVolume", restVolume);
            cb.SetComputeBufferParam(shader, kProlongate, "_CurrentVolumeBits", currentVolumeBits);
            cb.SetComputeBufferParam(shader, kProlongate, "_FixedChildPosBits", fixedChildPosBits);
            cb.SetComputeBufferParam(shader, kProlongate, "_FixedChildCount", fixedChildCount);
            cb.SetComputeBufferParam(shader, kProlongate, "_ParentIndex", parentIndex);
            cb.SetComputeBufferParam(shader, kProlongate, "_ParentIndices", parentIndices);
            cb.SetComputeBufferParam(shader, kProlongate, "_ParentWeights", parentWeights);
            cb.SetComputeBufferParam(shader, kProlongate, "_SavedVelPrefix", savedVelPrefix);
            cb.SetComputeBufferParam(shader, kProlongate, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            cb.SetComputeBufferParam(shader, kProlongate, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            cb.SetComputeBufferParam(shader, kProlongate, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            cb.SetComputeBufferParam(shader, kProlongate, "_DtCollisionOwnerByLocal", dtCollisionOwnerByLocal ?? dtOwnerByLocal ?? defaultDtCollisionOwnerByLocal);

            cb.SetComputeBufferParam(shader, kCommitDeformation, "_Pos", pos);
            cb.SetComputeBufferParam(shader, kCommitDeformation, "_Vel", vel);
            cb.SetComputeBufferParam(shader, kCommitDeformation, "_MaterialIds", materialIds);
            cb.SetComputeBufferParam(shader, kCommitDeformation, "_InvMass", invMass);
            cb.SetComputeBufferParam(shader, kCommitDeformation, "_F0", F0);
            cb.SetComputeBufferParam(shader, kCommitDeformation, "_L", L);
            cb.SetComputeBufferParam(shader, kCommitDeformation, "_CurrentVolumeBits", currentVolumeBits);
            cb.SetComputeBufferParam(shader, kCommitDeformation, "_FixedChildPosBits", fixedChildPosBits);
            cb.SetComputeBufferParam(shader, kCommitDeformation, "_FixedChildCount", fixedChildCount);
            cb.SetComputeBufferParam(shader, kCommitDeformation, "_F", F);
            cb.SetComputeBufferParam(shader, kCommitDeformation, "_Fp", Fp);

            cb.SetComputeBufferParam(shader, kRelaxColored, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            cb.SetComputeBufferParam(shader, kRelaxColored, "_DtCollisionOwnerByLocal", dtCollisionOwnerByLocal ?? dtOwnerByLocal ?? defaultDtCollisionOwnerByLocal);
            BindDtGlobalMappingParams(cb, kRelaxColored, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_DtCollisionOwnerByLocal", dtCollisionOwnerByLocal ?? dtOwnerByLocal ?? defaultDtCollisionOwnerByLocal);
            BindDtGlobalMappingParams(cb, kRelaxColoredPersistentCoarse, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            cb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_Vel", vel);
            cb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_InvMass", invMass);
            cb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_CoarseFixed", coarseFixed);
            cb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_Lambda", lambda);
            cb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_VelPrev", velPrev);
            cb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_LambdaPrev", lambdaPrev);
            cb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_JRVelDeltaBits", jrVelDeltaBits);
            cb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_JRLambdaDelta", jrLambdaDelta);

            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_Pos", pos);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_MaterialIds", materialIds);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_InvMass", invMass);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_RestVolume", restVolume);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_F0", F0);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_L", L);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_CurrentVolumeBits", currentVolumeBits);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_CurrentTotalMassBits", currentTotalMassBits);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_FixedChildPosBits", fixedChildPosBits);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_FixedChildCount", fixedChildCount);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_VelPrev", velPrev);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_LambdaPrev", lambdaPrev);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_DurabilityLambda", durabilityLambda);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_CollisionLambda", collisionLambda);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_FineContacts", fineContacts);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_FineContactCountBuffer", fineContactCount);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_CoarseContacts", coarseContacts);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_CoarseContactCountBuffer", coarseContactCount);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_BoundaryEdgeV0Gi", boundaryEdgeV0Gi);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_BoundaryEdgeV1Gi", boundaryEdgeV1Gi);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_BoundaryVertexGi", boundaryVertexGi);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_JRVelDeltaBits", jrVelDeltaBits);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_JRLambdaDelta", jrLambdaDelta);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_DtCollisionOwnerByLocal", dtCollisionOwnerByLocal ?? dtOwnerByLocal ?? defaultDtCollisionOwnerByLocal);
            cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_CoarseFixed", coarseFixed);

            cb.SetComputeBufferParam(shader, kJRApply, "_Vel", vel);
            cb.SetComputeBufferParam(shader, kJRApply, "_Lambda", lambda);
            cb.SetComputeBufferParam(shader, kJRApply, "_Pos", pos);
            cb.SetComputeBufferParam(shader, kJRApply, "_MaterialIds", materialIds);
            cb.SetComputeBufferParam(shader, kJRApply, "_InvMass", invMass);
            cb.SetComputeBufferParam(shader, kJRApply, "_RestVolume", restVolume);
            cb.SetComputeBufferParam(shader, kJRApply, "_CurrentVolumeBits", currentVolumeBits);
            cb.SetComputeBufferParam(shader, kJRApply, "_FixedChildPosBits", fixedChildPosBits);
            cb.SetComputeBufferParam(shader, kJRApply, "_FixedChildCount", fixedChildCount);
            cb.SetComputeBufferParam(shader, kJRApply, "_CoarseFixed", coarseFixed);
            cb.SetComputeBufferParam(shader, kJRApply, "_JRVelDeltaBits", jrVelDeltaBits);
            cb.SetComputeBufferParam(shader, kJRApply, "_JRLambdaDelta", jrLambdaDelta);

            BindDtGlobalMappingParams(cb, kJRSavePrevAndClear, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(cb, kJRComputeDeltas, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(cb, kJRApply, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            cb.SetComputeBufferParam(shader, kCommitDeformation, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            cb.SetComputeBufferParam(shader, kCommitDeformation, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            cb.SetComputeBufferParam(shader, kCommitDeformation, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            cb.SetComputeBufferParam(shader, kCommitDeformation, "_DtCollisionOwnerByLocal", dtCollisionOwnerByLocal ?? dtOwnerByLocal ?? defaultDtCollisionOwnerByLocal);
            BindDtGlobalMappingParams(cb, kCommitDeformation, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            cb.SetComputeBufferParam(shader, kRelaxColored, "_CoarseFixed", coarseFixed);
            cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_CoarseFixed", coarseFixed);
            cb.SetComputeBufferParam(shader, kCommitDeformation, "_CoarseFixed", coarseFixed);

            cb.SetComputeBufferParam(shader, kSmoothProlongatedFineVel, "_Vel", vel);
            cb.SetComputeBufferParam(shader, kSmoothProlongatedFineVel, "_Pos", pos);
            cb.SetComputeBufferParam(shader, kSmoothProlongatedFineVel, "_InvMass", invMass);
            cb.SetComputeBufferParam(shader, kSmoothProlongatedFineVel, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            cb.SetComputeBufferParam(shader, kSmoothProlongatedFineVel, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            cb.SetComputeBufferParam(shader, kSmoothProlongatedFineVel, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            cb.SetComputeBufferParam(shader, kSmoothProlongatedFineVel, "_DtCollisionOwnerByLocal", dtCollisionOwnerByLocal ?? dtOwnerByLocal ?? defaultDtCollisionOwnerByLocal);
            BindDtGlobalMappingParams(cb, kSmoothProlongatedFineVel, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            cb.SetComputeBufferParam(shader, kCopyVelToPrev, "_Vel", vel);
            cb.SetComputeBufferParam(shader, kCopyVelToPrev, "_VelPrev", velPrev);
            cb.SetComputeBufferParam(shader, kCopyVelToPrev, "_InvMass", invMass);
            cb.SetComputeBufferParam(shader, kCopyVelToPrev, "_CoarseFixed", coarseFixed);
            BindDtGlobalMappingParams(cb, kCopyVelToPrev, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            cb.SetComputeBufferParam(shader, kApplyXsph, "_Vel", vel);
            cb.SetComputeBufferParam(shader, kApplyXsph, "_VelPrev", velPrev);
            cb.SetComputeBufferParam(shader, kApplyXsph, "_Pos", pos);
            cb.SetComputeBufferParam(shader, kApplyXsph, "_InvMass", invMass);
            cb.SetComputeBufferParam(shader, kApplyXsph, "_RestVolume", restVolume);
            cb.SetComputeBufferParam(shader, kApplyXsph, "_CurrentVolumeBits", currentVolumeBits);
            cb.SetComputeBufferParam(shader, kApplyXsph, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            cb.SetComputeBufferParam(shader, kApplyXsph, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            cb.SetComputeBufferParam(shader, kApplyXsph, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            cb.SetComputeBufferParam(shader, kApplyXsph, "_DtCollisionOwnerByLocal", dtCollisionOwnerByLocal ?? dtOwnerByLocal ?? defaultDtCollisionOwnerByLocal);
            cb.SetComputeBufferParam(shader, kApplyXsph, "_CoarseFixed", coarseFixed);
            BindDtGlobalMappingParams(cb, kApplyXsph, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            cb.SetComputeBufferParam(shader, kApplyPositionCorrection, "_Pos", pos);
            cb.SetComputeBufferParam(shader, kApplyPositionCorrection, "_Vel", vel);
            cb.SetComputeBufferParam(shader, kApplyPositionCorrection, "_InvMass", invMass);
            cb.SetComputeBufferParam(shader, kApplyPositionCorrection, "_RestVolume", restVolume);
            cb.SetComputeBufferParam(shader, kApplyPositionCorrection, "_CurrentTotalMassBits", currentTotalMassBits);
            cb.SetComputeBufferParam(shader, kApplyPositionCorrection, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            cb.SetComputeBufferParam(shader, kApplyPositionCorrection, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            cb.SetComputeBufferParam(shader, kApplyPositionCorrection, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            cb.SetComputeBufferParam(shader, kApplyPositionCorrection, "_DtCollisionOwnerByLocal", dtCollisionOwnerByLocal ?? dtOwnerByLocal ?? defaultDtCollisionOwnerByLocal);
            cb.SetComputeBufferParam(shader, kApplyPositionCorrection, "_CoarseFixed", coarseFixed);
            BindDtGlobalMappingParams(cb, kApplyPositionCorrection, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            BindDtGlobalMappingParams(cb, kProlongate, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            if (physicalParams != null) {
                cb.SetComputeBufferParam(shader, kRelaxColored, "_PhysicalParams", physicalParams);
                cb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_PhysicalParams", physicalParams);
                cb.SetComputeBufferParam(shader, kJRComputeDeltas, "_PhysicalParams", physicalParams);
                cb.SetComputeBufferParam(shader, kJRApply, "_PhysicalParams", physicalParams);
                cb.SetComputeBufferParam(shader, kProlongate, "_PhysicalParams", physicalParams);
                cb.SetComputeBufferParam(shader, kCommitDeformation, "_PhysicalParams", physicalParams);
            }
        }
    }
}