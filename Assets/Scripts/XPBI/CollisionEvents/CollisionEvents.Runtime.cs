using System;
using Unity.Mathematics;
using UnityEngine;
using GPU.Neighbors;
using GPU.Delaunay;
using UnityEngine.Rendering;

namespace GPU.Solver {
    internal sealed partial class CollisionEvents {
        private const int SdfResolution = 64;
        private const int DefaultBinResolution = 32;
        private const int DefaultMaxEdgesPerBin = 96;
        private const int DefaultMaxVertsPerBin = 96;

        internal ComputeBuffer collisionEvents;
        private ComputeBuffer collisionEventCount;
        private ComputeBuffer fineNodeManifoldCount;
        private ComputeBuffer fineNodeManifoldOverflow;
        private ComputeBuffer fineNodeManifold;
        private ComputeBuffer fineNodeContactLambda;

        private ComputeBuffer coarseContacts;
        private ComputeBuffer coarseContactCount;
        private ComputeBuffer coarseNodeContactOverflow;
        private ComputeBuffer coarseNodeContactLambda;

        private ComputeBuffer colAnchorGi;
        private ComputeBuffer colNodeGi0;
        private ComputeBuffer colNodeGi1;
        private ComputeBuffer colNodeGi2;
        private ComputeBuffer colNodeGi3;
        private ComputeBuffer colBeta0;
        private ComputeBuffer colBeta1;
        private ComputeBuffer colBeta2;
        private ComputeBuffer colBeta3;
        private ComputeBuffer colNX;
        private ComputeBuffer colNY;
        private ComputeBuffer colPen;
        private ComputeBuffer colScale;
        private ComputeBuffer colOwnerA;
        private ComputeBuffer colOwnerB;
        private ComputeBuffer nodeCollisionRefCount;
        private ComputeBuffer nodeCollisionRefWrite;
        private ComputeBuffer nodeCollisionRefStart;
        private ComputeBuffer nodeCollisionRefs;

        private ComputeBuffer xferColCount;
        private ComputeBuffer xferColNXBits;
        private ComputeBuffer xferColNYBits;
        private ComputeBuffer xferColPenBits;
        private ComputeBuffer xferColSBits;
        private ComputeBuffer xferColTBits;
        private ComputeBuffer xferColQAGi;
        private ComputeBuffer xferColQBGi;
        private ComputeBuffer xferColOAGi;
        private ComputeBuffer xferColOBGi;

        private ComputeBuffer boundaryEdgeCount;

        private ComputeBuffer ownerGridOrigin;
        private ComputeBuffer ownerGridDim;
        private ComputeBuffer ownerGridTexel;
        private ComputeBuffer ownerGridBase;

        private ComputeBuffer ownerBinOrigin;
        private ComputeBuffer ownerBinDim;
        private ComputeBuffer ownerBinBase;

        private ComputeBuffer ownerPairs;

        private ComputeBuffer ownerBoundaryEdgeCounts;
        private ComputeBuffer ownerBoundaryOverflow;
        private ComputeBuffer ownerBoundaryEdgeRefs;

        private ComputeBuffer boundaryEdgeOwner;
        private ComputeBuffer boundaryEdgeV0Gi;
        private ComputeBuffer boundaryEdgeV1Gi;
        private ComputeBuffer boundaryOutwardIsRight;
        private ComputeBuffer boundaryEdgeP0;
        private ComputeBuffer boundaryEdgeP1;
        private ComputeBuffer boundaryEdgeNOut;
        private ComputeBuffer boundaryEdgeNIn;
        private ComputeBuffer boundaryEdgePseudoN0;
        private ComputeBuffer boundaryEdgePseudoN1;

        private ComputeBuffer boundaryVertexOwner;
        private ComputeBuffer boundaryVertexGi;
        private ComputeBuffer boundaryVertexP;
        private ComputeBuffer boundaryVertexPseudoN;

        private ComputeBuffer edgeBinCounts;
        private ComputeBuffer edgeBinOverflow;
        private ComputeBuffer edgeBinRefs;

        private ComputeBuffer vertBinCounts;
        private ComputeBuffer vertBinOverflow;
        private ComputeBuffer vertBinRefs;

        private ComputeBuffer sdfPhi;
        private ComputeBuffer sdfGrad;
        private ComputeBuffer sdfFeatType;
        private ComputeBuffer sdfFeatId;

        private int boundaryEdgeCapacity;
        private int ownerCapacity;
        private int pairCapacity;
        private int totalGridCellCapacity;
        private int totalBinCapacity;
        private int maxBoundaryEdgesPerOwner;
        private int maxEdgesPerBin;
        private int maxVertsPerBin;
        private int queryPairCount;

        private float2[] ownerGridOriginCpu = Array.Empty<float2>();
        private uint2[] ownerGridDimCpu = Array.Empty<uint2>();
        private float[] ownerGridTexelCpu = Array.Empty<float>();
        private uint[] ownerGridBaseCpu = Array.Empty<uint>();

        private float2[] ownerBinOriginCpu = Array.Empty<float2>();
        private uint2[] ownerBinDimCpu = Array.Empty<uint2>();
        private uint[] ownerBinBaseCpu = Array.Empty<uint>();

        private uint2[] ownerPairsCpu = Array.Empty<uint2>();
        private readonly uint[] oneUint = new uint[1];
        private int[] identityGlobalNodeMapCpu = Array.Empty<int>();
        private ComputeBuffer identityGlobalNodeMap;

        internal int kClearState;
        internal int kBuildBoundaryFeatures;
        internal int kBinBoundaryEdges;
        internal int kBinBoundaryVertices;
        internal int kBuildOwnerFeatureField;
        internal int kQueryNodeSurfaceContacts;
        internal int kQueryEdgeEdgeNodeContacts;
        internal int kClearFineNodeManifolds;
        internal int kScatterFineNodeContactsToManifolds;
        internal int kClearCoarseNodeContacts;
        internal int kPropagateFineContactsToCoarse;

        internal ComputeBuffer CollisionEventsBuffer => collisionEvents;
        internal ComputeBuffer CollisionEventCountBuffer => collisionEventCount;
        internal ComputeBuffer FineNodeContactCountPerNodeBuffer => fineNodeManifoldCount;
        internal ComputeBuffer FineNodeContactsPerNodeBuffer => fineNodeManifold;
        internal ComputeBuffer FineNodeContactLambdaBuffer => fineNodeContactLambda;
        internal ComputeBuffer CoarseContactsBuffer => coarseContacts;
        internal ComputeBuffer CoarseContactCountBuffer => coarseContactCount;
        internal ComputeBuffer CoarseNodeContactCountPerNodeBuffer => coarseContactCount;
        internal ComputeBuffer CoarseNodeContactsPerNodeBuffer => coarseContacts;
        internal ComputeBuffer CoarseNodeContactLambdaBuffer => coarseNodeContactLambda;
        internal ComputeBuffer ColAnchorGiBuffer => colAnchorGi;
        internal ComputeBuffer ColNodeGi0Buffer => colNodeGi0;
        internal ComputeBuffer ColNodeGi1Buffer => colNodeGi1;
        internal ComputeBuffer ColNodeGi2Buffer => colNodeGi2;
        internal ComputeBuffer ColNodeGi3Buffer => colNodeGi3;
        internal ComputeBuffer ColBeta0Buffer => colBeta0;
        internal ComputeBuffer ColBeta1Buffer => colBeta1;
        internal ComputeBuffer ColBeta2Buffer => colBeta2;
        internal ComputeBuffer ColBeta3Buffer => colBeta3;
        internal ComputeBuffer ColNXBuffer => colNX;
        internal ComputeBuffer ColNYBuffer => colNY;
        internal ComputeBuffer ColPenBuffer => colPen;
        internal ComputeBuffer ColScaleBuffer => colScale;
        internal ComputeBuffer ColOwnerABuffer => colOwnerA;
        internal ComputeBuffer ColOwnerBBuffer => colOwnerB;
        internal ComputeBuffer NodeCollisionRefCountBuffer => nodeCollisionRefCount;
        internal ComputeBuffer NodeCollisionRefWriteBuffer => nodeCollisionRefWrite;
        internal ComputeBuffer NodeCollisionRefStartBuffer => nodeCollisionRefStart;
        internal ComputeBuffer NodeCollisionRefsBuffer => nodeCollisionRefs;
        internal ComputeBuffer BoundaryChunkCountBuffer => boundaryEdgeCount;
        internal ComputeBuffer BoundaryChunksBuffer => boundaryEdgeV0Gi;
        internal ComputeBuffer BoundaryEdgeOwnerBuffer => boundaryEdgeOwner;
        internal ComputeBuffer BoundaryEdgeV0Buffer => boundaryEdgeV0Gi;
        internal ComputeBuffer BoundaryEdgeV1Buffer => boundaryEdgeV1Gi;
        internal ComputeBuffer BoundaryVertexGiBuffer => boundaryVertexGi;
        internal ComputeBuffer BoundaryEdgeNormalBuffer => boundaryEdgeNOut;
        internal ComputeBuffer BoundaryEdgeP0Buffer => boundaryEdgeP0;
        internal ComputeBuffer BoundaryEdgeP1Buffer => boundaryEdgeP1;

        internal ComputeBuffer XferColCountBuffer => xferColCount;
        internal ComputeBuffer XferColNXBitsBuffer => xferColNXBits;
        internal ComputeBuffer XferColNYBitsBuffer => xferColNYBits;
        internal ComputeBuffer XferColPenBitsBuffer => xferColPenBits;
        internal ComputeBuffer XferColSBitsBuffer => xferColSBits;
        internal ComputeBuffer XferColTBitsBuffer => xferColTBits;
        internal ComputeBuffer XferColQAGiBuffer => xferColQAGi;
        internal ComputeBuffer XferColQBGiBuffer => xferColQBGi;
        internal ComputeBuffer XferColOAGiBuffer => xferColOAGi;
        internal ComputeBuffer XferColOBGiBuffer => xferColOBGi;

        internal int ClearCollisionEventCountKernel => -1;
        internal int ClearTransferredCollisionKernel => kClearCoarseNodeContacts;
        internal int RestrictCollisionEventsToActivePairsKernel => -1;

        internal int BoundaryChunkSortCapacity => 1;
        internal int BoundaryChunkCapacity => boundaryEdgeCapacity;
        internal int OwnerPairCapacity => pairCapacity;
        internal int TotalBinCapacity => totalBinCapacity;
        internal int SdfGridDimX => SdfResolution;
        internal int SdfGridDimY => SdfResolution;
        internal int MaxBoundaryEdgesPerOwner => maxBoundaryEdgesPerOwner;
        internal int QueryPairCount => queryPairCount;
        internal int LbvhLeafOffset => 0;
        internal int LbvhNodeCapacity => 0;

        private static void ReleaseBuffer(ref ComputeBuffer buffer) {
            buffer?.Dispose();
            buffer = null;
        }

        private ComputeBuffer EnsureIdentityGlobalNodeMap(int count) {
            int required = math.max(1, count);
            if (identityGlobalNodeMap == null || !identityGlobalNodeMap.IsValid() || identityGlobalNodeMap.count != required) {
                ReleaseBuffer(ref identityGlobalNodeMap);
                identityGlobalNodeMap = new ComputeBuffer(required, sizeof(int), ComputeBufferType.Structured);
            }

            if (identityGlobalNodeMapCpu.Length < required)
                identityGlobalNodeMapCpu = new int[required];

            for (int i = 0; i < required; i++)
                identityGlobalNodeMapCpu[i] = i;

            identityGlobalNodeMap.SetData(identityGlobalNodeMapCpu, 0, 0, required);
            return identityGlobalNodeMap;
        }

        private void BindDtGlobalMappingParams(CommandBuffer cb, int kernel, bool useDtGlobalNodeMap, int dtLocalBase, ComputeBuffer dtGlobalNodeMap, ComputeBuffer dtGlobalToLayerLocalMap) {
            solver.layerMappingCache.BindDtGlobalMappingParams(cb, shader, kernel, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
        }

        private static int NextPow2(int v) {
            v = math.max(1, v);
            int p = 1;
            while (p < v)
                p <<= 1;
            return p;
        }

        private void EnsureOwnerCapacity(int requiredOwnerCount) {
            int required = math.max(1, requiredOwnerCount);
            if (required <= ownerCapacity && ownerGridOrigin != null && ownerGridOrigin.IsValid())
                return;

            ownerCapacity = NextPow2(required);
            int binsPerOwner = DefaultBinResolution * DefaultBinResolution;
            int gridCellsPerOwner = SdfResolution * SdfResolution;

            maxBoundaryEdgesPerOwner = math.max(64, (boundaryEdgeCapacity + ownerCapacity - 1) / ownerCapacity * 2);
            maxEdgesPerBin = DefaultMaxEdgesPerBin;
            maxVertsPerBin = DefaultMaxVertsPerBin;
            totalGridCellCapacity = ownerCapacity * gridCellsPerOwner;
            totalBinCapacity = ownerCapacity * binsPerOwner;

            ReleaseBuffer(ref ownerGridOrigin);
            ownerGridOrigin = new ComputeBuffer(ownerCapacity, sizeof(float) * 2, ComputeBufferType.Structured);
            ReleaseBuffer(ref ownerGridDim);
            ownerGridDim = new ComputeBuffer(ownerCapacity, sizeof(uint) * 2, ComputeBufferType.Structured);
            ReleaseBuffer(ref ownerGridTexel);
            ownerGridTexel = new ComputeBuffer(ownerCapacity, sizeof(float), ComputeBufferType.Structured);
            ReleaseBuffer(ref ownerGridBase);
            ownerGridBase = new ComputeBuffer(ownerCapacity, sizeof(uint), ComputeBufferType.Structured);

            ReleaseBuffer(ref ownerBinOrigin);
            ownerBinOrigin = new ComputeBuffer(ownerCapacity, sizeof(float) * 2, ComputeBufferType.Structured);
            ReleaseBuffer(ref ownerBinDim);
            ownerBinDim = new ComputeBuffer(ownerCapacity, sizeof(uint) * 2, ComputeBufferType.Structured);
            ReleaseBuffer(ref ownerBinBase);
            ownerBinBase = new ComputeBuffer(ownerCapacity, sizeof(uint), ComputeBufferType.Structured);

            ReleaseBuffer(ref ownerBoundaryEdgeCounts);
            ownerBoundaryEdgeCounts = new ComputeBuffer(ownerCapacity, sizeof(uint), ComputeBufferType.Structured);
            ReleaseBuffer(ref ownerBoundaryOverflow);
            ownerBoundaryOverflow = new ComputeBuffer(ownerCapacity, sizeof(uint), ComputeBufferType.Structured);
            ReleaseBuffer(ref ownerBoundaryEdgeRefs);
            ownerBoundaryEdgeRefs = new ComputeBuffer(ownerCapacity * maxBoundaryEdgesPerOwner, sizeof(uint), ComputeBufferType.Structured);

            ReleaseBuffer(ref edgeBinCounts);
            edgeBinCounts = new ComputeBuffer(totalBinCapacity, sizeof(uint), ComputeBufferType.Structured);
            ReleaseBuffer(ref edgeBinOverflow);
            edgeBinOverflow = new ComputeBuffer(totalBinCapacity, sizeof(uint), ComputeBufferType.Structured);
            ReleaseBuffer(ref edgeBinRefs);
            edgeBinRefs = new ComputeBuffer(totalBinCapacity * maxEdgesPerBin, sizeof(uint), ComputeBufferType.Structured);

            ReleaseBuffer(ref vertBinCounts);
            vertBinCounts = new ComputeBuffer(totalBinCapacity, sizeof(uint), ComputeBufferType.Structured);
            ReleaseBuffer(ref vertBinOverflow);
            vertBinOverflow = new ComputeBuffer(totalBinCapacity, sizeof(uint), ComputeBufferType.Structured);
            ReleaseBuffer(ref vertBinRefs);
            vertBinRefs = new ComputeBuffer(totalBinCapacity * maxVertsPerBin, sizeof(uint), ComputeBufferType.Structured);

            ReleaseBuffer(ref sdfPhi);
            sdfPhi = new ComputeBuffer(totalGridCellCapacity, sizeof(float), ComputeBufferType.Structured);
            ReleaseBuffer(ref sdfGrad);
            sdfGrad = new ComputeBuffer(totalGridCellCapacity, sizeof(float) * 2, ComputeBufferType.Structured);
            ReleaseBuffer(ref sdfFeatType);
            sdfFeatType = new ComputeBuffer(totalGridCellCapacity, sizeof(uint), ComputeBufferType.Structured);
            ReleaseBuffer(ref sdfFeatId);
            sdfFeatId = new ComputeBuffer(totalGridCellCapacity, sizeof(uint), ComputeBufferType.Structured);

            if (ownerGridOriginCpu.Length < ownerCapacity)
                ownerGridOriginCpu = new float2[ownerCapacity];
            if (ownerGridDimCpu.Length < ownerCapacity)
                ownerGridDimCpu = new uint2[ownerCapacity];
            if (ownerGridTexelCpu.Length < ownerCapacity)
                ownerGridTexelCpu = new float[ownerCapacity];
            if (ownerGridBaseCpu.Length < ownerCapacity)
                ownerGridBaseCpu = new uint[ownerCapacity];

            if (ownerBinOriginCpu.Length < ownerCapacity)
                ownerBinOriginCpu = new float2[ownerCapacity];
            if (ownerBinDimCpu.Length < ownerCapacity)
                ownerBinDimCpu = new uint2[ownerCapacity];
            if (ownerBinBaseCpu.Length < ownerCapacity)
                ownerBinBaseCpu = new uint[ownerCapacity];
        }

        internal void AllocateRuntimeBuffers(int newCapacity) {
            int collisionCapacity = math.max(4096, newCapacity * 32);
            int transferCapacity = newCapacity * Const.NeighborCount * Const.CollisionTransferManifoldSlots;
            int fineNodeManifoldCapacity = newCapacity * 2;
            int coarseNodeManifoldCapacity = newCapacity * 4;
            int stencilCapacity = collisionCapacity + transferCapacity;

            boundaryEdgeCapacity = math.max(2048, newCapacity * 12);
            pairCapacity = math.max(256, newCapacity * 2);

            // FineNodeContact in XPBI.CollisionEventsNew.* is 52 bytes.
            collisionEvents = new ComputeBuffer(collisionCapacity, sizeof(uint) * 6 + sizeof(float) * 4 + sizeof(float) * 2 + sizeof(float), ComputeBufferType.Structured);
            collisionEventCount = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Structured);

            fineNodeManifoldCount = new ComputeBuffer(newCapacity, sizeof(uint), ComputeBufferType.Structured);
            fineNodeManifoldOverflow = new ComputeBuffer(newCapacity, sizeof(uint), ComputeBufferType.Structured);
            fineNodeManifold = new ComputeBuffer(fineNodeManifoldCapacity, sizeof(uint) * 6 + sizeof(float) * 4 + sizeof(float) * 2 + sizeof(float), ComputeBufferType.Structured);
            fineNodeContactLambda = new ComputeBuffer(fineNodeManifoldCapacity, sizeof(float), ComputeBufferType.Structured);

            // XPBI_NodeContact in XPBI.CollisionEventsNew.Transfer.hlsl is 52 bytes.
            coarseContacts = new ComputeBuffer(coarseNodeManifoldCapacity, sizeof(uint) * 6 + sizeof(float) * 4 + sizeof(float) * 2 + sizeof(float), ComputeBufferType.Structured);
            coarseContactCount = new ComputeBuffer(newCapacity, sizeof(uint), ComputeBufferType.Structured);
            coarseNodeContactOverflow = new ComputeBuffer(newCapacity, sizeof(uint), ComputeBufferType.Structured);
            coarseNodeContactLambda = new ComputeBuffer(coarseNodeManifoldCapacity, sizeof(float), ComputeBufferType.Structured);

            colAnchorGi = new ComputeBuffer(stencilCapacity, sizeof(uint), ComputeBufferType.Structured);
            colNodeGi0 = new ComputeBuffer(stencilCapacity, sizeof(uint), ComputeBufferType.Structured);
            colNodeGi1 = new ComputeBuffer(stencilCapacity, sizeof(uint), ComputeBufferType.Structured);
            colNodeGi2 = new ComputeBuffer(stencilCapacity, sizeof(uint), ComputeBufferType.Structured);
            colNodeGi3 = new ComputeBuffer(stencilCapacity, sizeof(uint), ComputeBufferType.Structured);
            colBeta0 = new ComputeBuffer(stencilCapacity, sizeof(float), ComputeBufferType.Structured);
            colBeta1 = new ComputeBuffer(stencilCapacity, sizeof(float), ComputeBufferType.Structured);
            colBeta2 = new ComputeBuffer(stencilCapacity, sizeof(float), ComputeBufferType.Structured);
            colBeta3 = new ComputeBuffer(stencilCapacity, sizeof(float), ComputeBufferType.Structured);
            colNX = new ComputeBuffer(stencilCapacity, sizeof(float), ComputeBufferType.Structured);
            colNY = new ComputeBuffer(stencilCapacity, sizeof(float), ComputeBufferType.Structured);
            colPen = new ComputeBuffer(stencilCapacity, sizeof(float), ComputeBufferType.Structured);
            colScale = new ComputeBuffer(stencilCapacity, sizeof(float), ComputeBufferType.Structured);
            colOwnerA = new ComputeBuffer(stencilCapacity, sizeof(uint), ComputeBufferType.Structured);
            colOwnerB = new ComputeBuffer(stencilCapacity, sizeof(uint), ComputeBufferType.Structured);

            nodeCollisionRefCount = new ComputeBuffer(newCapacity, sizeof(uint), ComputeBufferType.Structured);
            nodeCollisionRefWrite = new ComputeBuffer(newCapacity, sizeof(uint), ComputeBufferType.Structured);
            nodeCollisionRefStart = new ComputeBuffer(newCapacity + 1, sizeof(uint), ComputeBufferType.Structured);
            nodeCollisionRefs = new ComputeBuffer(stencilCapacity, sizeof(uint), ComputeBufferType.Structured);

            xferColCount = new ComputeBuffer(transferCapacity, sizeof(uint), ComputeBufferType.Structured);
            xferColNXBits = new ComputeBuffer(transferCapacity, sizeof(uint), ComputeBufferType.Structured);
            xferColNYBits = new ComputeBuffer(transferCapacity, sizeof(uint), ComputeBufferType.Structured);
            xferColPenBits = new ComputeBuffer(transferCapacity, sizeof(uint), ComputeBufferType.Structured);
            xferColSBits = new ComputeBuffer(transferCapacity, sizeof(uint), ComputeBufferType.Structured);
            xferColTBits = new ComputeBuffer(transferCapacity, sizeof(uint), ComputeBufferType.Structured);
            xferColQAGi = new ComputeBuffer(transferCapacity, sizeof(uint), ComputeBufferType.Structured);
            xferColQBGi = new ComputeBuffer(transferCapacity, sizeof(uint), ComputeBufferType.Structured);
            xferColOAGi = new ComputeBuffer(transferCapacity, sizeof(uint), ComputeBufferType.Structured);
            xferColOBGi = new ComputeBuffer(transferCapacity, sizeof(uint), ComputeBufferType.Structured);

            boundaryEdgeCount = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Structured);

            boundaryEdgeOwner = new ComputeBuffer(boundaryEdgeCapacity, sizeof(uint), ComputeBufferType.Structured);
            boundaryEdgeV0Gi = new ComputeBuffer(boundaryEdgeCapacity, sizeof(uint), ComputeBufferType.Structured);
            boundaryEdgeV1Gi = new ComputeBuffer(boundaryEdgeCapacity, sizeof(uint), ComputeBufferType.Structured);
            boundaryOutwardIsRight = new ComputeBuffer(boundaryEdgeCapacity, sizeof(uint), ComputeBufferType.Structured);
            boundaryEdgeP0 = new ComputeBuffer(boundaryEdgeCapacity, sizeof(float) * 2, ComputeBufferType.Structured);
            boundaryEdgeP1 = new ComputeBuffer(boundaryEdgeCapacity, sizeof(float) * 2, ComputeBufferType.Structured);
            boundaryEdgeNOut = new ComputeBuffer(boundaryEdgeCapacity, sizeof(float) * 2, ComputeBufferType.Structured);
            boundaryEdgeNIn = new ComputeBuffer(boundaryEdgeCapacity, sizeof(float) * 2, ComputeBufferType.Structured);
            boundaryEdgePseudoN0 = new ComputeBuffer(boundaryEdgeCapacity, sizeof(float) * 2, ComputeBufferType.Structured);
            boundaryEdgePseudoN1 = new ComputeBuffer(boundaryEdgeCapacity, sizeof(float) * 2, ComputeBufferType.Structured);

            boundaryVertexOwner = new ComputeBuffer(boundaryEdgeCapacity, sizeof(uint), ComputeBufferType.Structured);
            boundaryVertexGi = new ComputeBuffer(boundaryEdgeCapacity, sizeof(uint), ComputeBufferType.Structured);
            boundaryVertexP = new ComputeBuffer(boundaryEdgeCapacity, sizeof(float) * 2, ComputeBufferType.Structured);
            boundaryVertexPseudoN = new ComputeBuffer(boundaryEdgeCapacity, sizeof(float) * 2, ComputeBufferType.Structured);

            ownerPairs = new ComputeBuffer(pairCapacity, sizeof(uint) * 2, ComputeBufferType.Structured);
            if (ownerPairsCpu.Length < pairCapacity)
                ownerPairsCpu = new uint2[pairCapacity];

            ownerCapacity = 0;
            totalGridCellCapacity = 0;
            totalBinCapacity = 0;
            maxBoundaryEdgesPerOwner = 0;
            maxEdgesPerBin = 0;
            maxVertsPerBin = 0;
            queryPairCount = 0;

            EnsureOwnerCapacity(1);
        }

        internal void ReleaseRuntimeBuffers() {
            ReleaseBuffer(ref collisionEvents);
            ReleaseBuffer(ref collisionEventCount);
            ReleaseBuffer(ref fineNodeManifoldCount);
            ReleaseBuffer(ref fineNodeManifoldOverflow);
            ReleaseBuffer(ref fineNodeManifold);
            ReleaseBuffer(ref fineNodeContactLambda);
            ReleaseBuffer(ref coarseContacts);
            ReleaseBuffer(ref coarseContactCount);
            ReleaseBuffer(ref coarseNodeContactOverflow);
            ReleaseBuffer(ref coarseNodeContactLambda);

            ReleaseBuffer(ref colAnchorGi);
            ReleaseBuffer(ref colNodeGi0);
            ReleaseBuffer(ref colNodeGi1);
            ReleaseBuffer(ref colNodeGi2);
            ReleaseBuffer(ref colNodeGi3);
            ReleaseBuffer(ref colBeta0);
            ReleaseBuffer(ref colBeta1);
            ReleaseBuffer(ref colBeta2);
            ReleaseBuffer(ref colBeta3);
            ReleaseBuffer(ref colNX);
            ReleaseBuffer(ref colNY);
            ReleaseBuffer(ref colPen);
            ReleaseBuffer(ref colScale);
            ReleaseBuffer(ref colOwnerA);
            ReleaseBuffer(ref colOwnerB);
            ReleaseBuffer(ref nodeCollisionRefCount);
            ReleaseBuffer(ref nodeCollisionRefWrite);
            ReleaseBuffer(ref nodeCollisionRefStart);
            ReleaseBuffer(ref nodeCollisionRefs);

            ReleaseBuffer(ref xferColCount);
            ReleaseBuffer(ref xferColNXBits);
            ReleaseBuffer(ref xferColNYBits);
            ReleaseBuffer(ref xferColPenBits);
            ReleaseBuffer(ref xferColSBits);
            ReleaseBuffer(ref xferColTBits);
            ReleaseBuffer(ref xferColQAGi);
            ReleaseBuffer(ref xferColQBGi);
            ReleaseBuffer(ref xferColOAGi);
            ReleaseBuffer(ref xferColOBGi);

            ReleaseBuffer(ref boundaryEdgeCount);

            ReleaseBuffer(ref ownerGridOrigin);
            ReleaseBuffer(ref ownerGridDim);
            ReleaseBuffer(ref ownerGridTexel);
            ReleaseBuffer(ref ownerGridBase);
            ReleaseBuffer(ref ownerBinOrigin);
            ReleaseBuffer(ref ownerBinDim);
            ReleaseBuffer(ref ownerBinBase);
            ReleaseBuffer(ref ownerPairs);
            ReleaseBuffer(ref ownerBoundaryEdgeCounts);
            ReleaseBuffer(ref ownerBoundaryOverflow);
            ReleaseBuffer(ref ownerBoundaryEdgeRefs);
            ReleaseBuffer(ref boundaryEdgeOwner);
            ReleaseBuffer(ref boundaryEdgeV0Gi);
            ReleaseBuffer(ref boundaryEdgeV1Gi);
            ReleaseBuffer(ref boundaryOutwardIsRight);
            ReleaseBuffer(ref boundaryEdgeP0);
            ReleaseBuffer(ref boundaryEdgeP1);
            ReleaseBuffer(ref boundaryEdgeNOut);
            ReleaseBuffer(ref boundaryEdgeNIn);
            ReleaseBuffer(ref boundaryEdgePseudoN0);
            ReleaseBuffer(ref boundaryEdgePseudoN1);
            ReleaseBuffer(ref boundaryVertexOwner);
            ReleaseBuffer(ref boundaryVertexGi);
            ReleaseBuffer(ref boundaryVertexP);
            ReleaseBuffer(ref boundaryVertexPseudoN);
            ReleaseBuffer(ref edgeBinCounts);
            ReleaseBuffer(ref edgeBinOverflow);
            ReleaseBuffer(ref edgeBinRefs);
            ReleaseBuffer(ref vertBinCounts);
            ReleaseBuffer(ref vertBinOverflow);
            ReleaseBuffer(ref vertBinRefs);
            ReleaseBuffer(ref sdfPhi);
            ReleaseBuffer(ref sdfGrad);
            ReleaseBuffer(ref sdfFeatType);
            ReleaseBuffer(ref sdfFeatId);
            ReleaseBuffer(ref identityGlobalNodeMap);
            identityGlobalNodeMapCpu = Array.Empty<int>();

            boundaryEdgeCapacity = 0;
            ownerCapacity = 0;
            pairCapacity = 0;
            totalGridCellCapacity = 0;
            totalBinCapacity = 0;
            maxBoundaryEdgesPerOwner = 0;
            maxEdgesPerBin = 0;
            maxVertsPerBin = 0;
            queryPairCount = 0;

            ownerGridOriginCpu = Array.Empty<float2>();
            ownerGridDimCpu = Array.Empty<uint2>();
            ownerGridTexelCpu = Array.Empty<float>();
            ownerGridBaseCpu = Array.Empty<uint>();
            ownerBinOriginCpu = Array.Empty<float2>();
            ownerBinDimCpu = Array.Empty<uint2>();
            ownerBinBaseCpu = Array.Empty<uint>();
            ownerPairsCpu = Array.Empty<uint2>();
        }

        internal void CacheRuntimeKernels() {
            kClearState = shader.FindKernel("ClearState");
            kBuildBoundaryFeatures = shader.FindKernel("BuildBoundaryFeatures");
            kBinBoundaryEdges = shader.FindKernel("BinBoundaryEdges");
            kBinBoundaryVertices = shader.FindKernel("BinBoundaryVertices");
            kBuildOwnerFeatureField = shader.FindKernel("BuildOwnerFeatureField");
            kQueryNodeSurfaceContacts = shader.FindKernel("QueryNodeSurfaceContacts");
            kQueryEdgeEdgeNodeContacts = shader.FindKernel("QueryEdgeEdgeNodeContacts");
            kClearFineNodeManifolds = shader.FindKernel("ClearFineNodeManifolds");
            kScatterFineNodeContactsToManifolds = shader.FindKernel("ScatterFineNodeContactsToManifolds");
            kClearCoarseNodeContacts = shader.FindKernel("ClearCoarseNodeContacts");
            kPropagateFineContactsToCoarse = shader.FindKernel("PropagateFineContactsToCoarse");
        }

        private void BindStencilKernelBuffers(CommandBuffer cb, int kernel) {
            cb.SetComputeBufferParam(shader, kernel, "_Contacts", collisionEvents);
            cb.SetComputeBufferParam(shader, kernel, "_ContactCount", collisionEventCount);
            cb.SetComputeBufferParam(shader, kernel, "_CoarseContacts", coarseContacts);
            cb.SetComputeBufferParam(shader, kernel, "_CoarseContactCountBuffer", coarseContactCount);

            cb.SetComputeBufferParam(shader, kernel, "_ColAnchorGi", colAnchorGi);
            cb.SetComputeBufferParam(shader, kernel, "_ColNodeGi0", colNodeGi0);
            cb.SetComputeBufferParam(shader, kernel, "_ColNodeGi1", colNodeGi1);
            cb.SetComputeBufferParam(shader, kernel, "_ColNodeGi2", colNodeGi2);
            cb.SetComputeBufferParam(shader, kernel, "_ColNodeGi3", colNodeGi3);
            cb.SetComputeBufferParam(shader, kernel, "_ColBeta0", colBeta0);
            cb.SetComputeBufferParam(shader, kernel, "_ColBeta1", colBeta1);
            cb.SetComputeBufferParam(shader, kernel, "_ColBeta2", colBeta2);
            cb.SetComputeBufferParam(shader, kernel, "_ColBeta3", colBeta3);
            cb.SetComputeBufferParam(shader, kernel, "_ColNX", colNX);
            cb.SetComputeBufferParam(shader, kernel, "_ColNY", colNY);
            cb.SetComputeBufferParam(shader, kernel, "_ColPen", colPen);
            cb.SetComputeBufferParam(shader, kernel, "_ColScale", colScale);
            cb.SetComputeBufferParam(shader, kernel, "_ColOwnerA", colOwnerA);
            cb.SetComputeBufferParam(shader, kernel, "_ColOwnerB", colOwnerB);

            cb.SetComputeBufferParam(shader, kernel, "_NodeCollisionRefCount", nodeCollisionRefCount);
            cb.SetComputeBufferParam(shader, kernel, "_NodeCollisionRefWrite", nodeCollisionRefWrite);
            cb.SetComputeBufferParam(shader, kernel, "_NodeCollisionRefStart", nodeCollisionRefStart);
            cb.SetComputeBufferParam(shader, kernel, "_NodeCollisionRefs", nodeCollisionRefs);
        }

        private void BindTransferKernelBuffers(CommandBuffer cb, int kernel) {
            cb.SetComputeBufferParam(shader, kernel, "_FineNodeContacts", collisionEvents);
            cb.SetComputeBufferParam(shader, kernel, "_FineNodeContactCount", collisionEventCount);
            cb.SetComputeBufferParam(shader, kernel, "_LayerParentIndices", solver.parentIndices);
            cb.SetComputeBufferParam(shader, kernel, "_LayerParentWeights", solver.parentWeights);
            cb.SetComputeBufferParam(shader, kernel, "_CoarseNodeContacts", coarseContacts);
            cb.SetComputeBufferParam(shader, kernel, "_CoarseNodeContactCount", coarseContactCount);
            cb.SetComputeBufferParam(shader, kernel, "_CoarseNodeContactOverflow", coarseNodeContactOverflow);
            cb.SetComputeBufferParam(shader, kernel, "_CoarseNodeContactLambda", coarseNodeContactLambda);
        }

        private int BuildOwnerPairs(int ownerCount) {
            int pairCount = 0;
            for (uint a = 0; a < ownerCount; a++) {
                for (uint b = a + 1u; b < ownerCount; b++) {
                    if (pairCount >= ownerPairsCpu.Length)
                        return pairCount;
                    ownerPairsCpu[pairCount++] = new uint2(a, b);
                }
            }
            return pairCount;
        }

        private void PrepareOwnerGridData(int ownerCount, float2 boundsMin, float2 boundsMax, float layerKernelH) {
            float margin = math.max(Const.CollisionSdfBandHalfWidthScale * layerKernelH, 1e-3f);

            float2 min = boundsMin;
            float2 max = boundsMax;
            if (math.any(max <= min)) {
                min = new float2(-1f, -1f);
                max = new float2(1f, 1f);
            }

            min -= margin;
            max += margin;

            float2 size = max - min;
            float maxExtent = math.max(size.x, size.y);
            float texel = math.max(maxExtent / SdfResolution, 1e-4f);

            uint2 gridDim = new uint2(SdfResolution, SdfResolution);
            uint2 binDim = new uint2(DefaultBinResolution, DefaultBinResolution);
            uint gridCellsPerOwner = (uint)(SdfResolution * SdfResolution);
            uint binsPerOwner = (uint)(DefaultBinResolution * DefaultBinResolution);

            for (int owner = 0; owner < ownerCount; owner++) {
                ownerGridOriginCpu[owner] = min;
                ownerGridDimCpu[owner] = gridDim;
                ownerGridTexelCpu[owner] = texel;
                ownerGridBaseCpu[owner] = (uint)owner * gridCellsPerOwner;

                ownerBinOriginCpu[owner] = min;
                ownerBinDimCpu[owner] = binDim;
                ownerBinBaseCpu[owner] = (uint)owner * binsPerOwner;
            }
        }

        private void BindKernelBuffers(
            CommandBuffer cb,
            int kernel,
            DT layer0Dt,
            int dtReadSlot,
            ComputeBuffer dtOwnerByLocal,
            ComputeBuffer dtGlobalNodeMap,
            ComputeBuffer dtGlobalToLayerLocalMap,
            bool useDtGlobalNodeMap
        ) {
            cb.SetComputeBufferParam(shader, kernel, "_Pos", solver.pos);
            cb.SetComputeBufferParam(shader, kernel, "_Vel", solver.vel);

            cb.SetComputeBufferParam(shader, kernel, "_DtCollisionOwnerByLocal", dtOwnerByLocal ?? solver.layerMappingCache.DefaultDtCollisionOwnerByLocal);
            cb.SetComputeBufferParam(shader, kernel, "_DtHalfEdges", layer0Dt.GetHalfEdgesBuffer(dtReadSlot));
            cb.SetComputeBufferParam(shader, kernel, "_DtBoundaryEdgeFlags", layer0Dt.BoundaryEdgeFlagsBuffer);
            cb.SetComputeBufferParam(shader, kernel, "_DtGlobalVertexByLocal", dtGlobalNodeMap ?? solver.layerMappingCache.DefaultDtGlobalNodeMap);
            cb.SetComputeBufferParam(shader, kernel, "_DtTriInternal", layer0Dt.TriInternalBuffer);
            cb.SetComputeBufferParam(shader, kernel, "_DtBoundaryNormals", layer0Dt.BoundaryNormalsBuffer);

            BindDtGlobalMappingParams(cb, kernel, useDtGlobalNodeMap, 0, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            cb.SetComputeBufferParam(shader, kernel, "_OwnerGridOrigin", ownerGridOrigin);
            cb.SetComputeBufferParam(shader, kernel, "_OwnerGridDim", ownerGridDim);
            cb.SetComputeBufferParam(shader, kernel, "_OwnerGridTexel", ownerGridTexel);
            cb.SetComputeBufferParam(shader, kernel, "_OwnerGridBase", ownerGridBase);

            cb.SetComputeBufferParam(shader, kernel, "_OwnerBinOrigin", ownerBinOrigin);
            cb.SetComputeBufferParam(shader, kernel, "_OwnerBinDim", ownerBinDim);
            cb.SetComputeBufferParam(shader, kernel, "_OwnerBinBase", ownerBinBase);

            cb.SetComputeBufferParam(shader, kernel, "_OwnerPairs", ownerPairs);

            cb.SetComputeBufferParam(shader, kernel, "_OwnerBoundaryEdgeCounts", ownerBoundaryEdgeCounts);
            cb.SetComputeBufferParam(shader, kernel, "_OwnerBoundaryOverflow", ownerBoundaryOverflow);
            cb.SetComputeBufferParam(shader, kernel, "_OwnerBoundaryEdgeRefs", ownerBoundaryEdgeRefs);

            cb.SetComputeBufferParam(shader, kernel, "_BoundaryEdgeOwner", boundaryEdgeOwner);
            cb.SetComputeBufferParam(shader, kernel, "_BoundaryEdgeV0Gi", boundaryEdgeV0Gi);
            cb.SetComputeBufferParam(shader, kernel, "_BoundaryEdgeV1Gi", boundaryEdgeV1Gi);
            cb.SetComputeBufferParam(shader, kernel, "_BoundaryOutwardIsRight", boundaryOutwardIsRight);
            cb.SetComputeBufferParam(shader, kernel, "_BoundaryEdgeP0", boundaryEdgeP0);
            cb.SetComputeBufferParam(shader, kernel, "_BoundaryEdgeP1", boundaryEdgeP1);
            cb.SetComputeBufferParam(shader, kernel, "_BoundaryEdgeNOut", boundaryEdgeNOut);
            cb.SetComputeBufferParam(shader, kernel, "_BoundaryEdgeNIn", boundaryEdgeNIn);
            cb.SetComputeBufferParam(shader, kernel, "_BoundaryEdgePseudoN0", boundaryEdgePseudoN0);
            cb.SetComputeBufferParam(shader, kernel, "_BoundaryEdgePseudoN1", boundaryEdgePseudoN1);

            cb.SetComputeBufferParam(shader, kernel, "_BoundaryVertexOwner", boundaryVertexOwner);
            cb.SetComputeBufferParam(shader, kernel, "_BoundaryVertexGi", boundaryVertexGi);
            cb.SetComputeBufferParam(shader, kernel, "_BoundaryVertexP", boundaryVertexP);
            cb.SetComputeBufferParam(shader, kernel, "_BoundaryVertexPseudoN", boundaryVertexPseudoN);

            cb.SetComputeBufferParam(shader, kernel, "_EdgeBinCounts", edgeBinCounts);
            cb.SetComputeBufferParam(shader, kernel, "_EdgeBinOverflow", edgeBinOverflow);
            cb.SetComputeBufferParam(shader, kernel, "_EdgeBinRefs", edgeBinRefs);

            cb.SetComputeBufferParam(shader, kernel, "_VertBinCounts", vertBinCounts);
            cb.SetComputeBufferParam(shader, kernel, "_VertBinOverflow", vertBinOverflow);
            cb.SetComputeBufferParam(shader, kernel, "_VertBinRefs", vertBinRefs);

            cb.SetComputeBufferParam(shader, kernel, "_SdfPhi", sdfPhi);
            cb.SetComputeBufferParam(shader, kernel, "_SdfGrad", sdfGrad);
            cb.SetComputeBufferParam(shader, kernel, "_SdfFeatType", sdfFeatType);
            cb.SetComputeBufferParam(shader, kernel, "_SdfFeatId", sdfFeatId);

            cb.SetComputeBufferParam(shader, kernel, "_FineNodeContacts", collisionEvents);
            cb.SetComputeBufferParam(shader, kernel, "_FineNodeContactCount", collisionEventCount);
            cb.SetComputeBufferParam(shader, kernel, "_FineNodeManifoldCount", fineNodeManifoldCount);
            cb.SetComputeBufferParam(shader, kernel, "_FineNodeManifoldOverflow", fineNodeManifoldOverflow);
            cb.SetComputeBufferParam(shader, kernel, "_FineNodeManifold", fineNodeManifold);
        }

        private void PrepareLayer0BuildBuffers(
            CommandBuffer cb,
            INeighborSearch layer0NeighborSearch,
            DT layer0Dt,
            int dtReadSlot,
            int layer0ActiveCount,
            float layer0KernelH,
            float2 layer0BoundsMin,
            float2 layer0BoundsMax,
            bool useDtGlobalNodeMap,
            ComputeBuffer dtGlobalNodeMap,
            ComputeBuffer dtGlobalToLayerLocalMap,
            ComputeBuffer dtOwnerByLocal,
            bool resetForPrepass
        ) {
            if (layer0Dt == null)
                return;

            int collisionOwnerCount = math.max(1, solver.solveRanges.Count);
            EnsureOwnerCapacity(collisionOwnerCount);

            PrepareOwnerGridData(collisionOwnerCount, layer0BoundsMin, layer0BoundsMax, layer0KernelH);
            queryPairCount = BuildOwnerPairs(collisionOwnerCount);

            if (resetForPrepass) {
                oneUint[0] = (uint)layer0Dt.HalfEdgeCount;
                cb.SetBufferData(boundaryEdgeCount, oneUint);
            }

            if (collisionOwnerCount > 0) {
                cb.SetBufferData(ownerGridOrigin, ownerGridOriginCpu, 0, 0, collisionOwnerCount);
                cb.SetBufferData(ownerGridDim, ownerGridDimCpu, 0, 0, collisionOwnerCount);
                cb.SetBufferData(ownerGridTexel, ownerGridTexelCpu, 0, 0, collisionOwnerCount);
                cb.SetBufferData(ownerGridBase, ownerGridBaseCpu, 0, 0, collisionOwnerCount);

                cb.SetBufferData(ownerBinOrigin, ownerBinOriginCpu, 0, 0, collisionOwnerCount);
                cb.SetBufferData(ownerBinDim, ownerBinDimCpu, 0, 0, collisionOwnerCount);
                cb.SetBufferData(ownerBinBase, ownerBinBaseCpu, 0, 0, collisionOwnerCount);
            }

            if (queryPairCount > 0)
                cb.SetBufferData(ownerPairs, ownerPairsCpu, 0, 0, queryPairCount);

            cb.SetComputeIntParam(shader, "_DtHalfEdgeCount", layer0Dt.HalfEdgeCount);
            cb.SetComputeIntParam(shader, "_DtTriCount", layer0Dt.TriCount);
            cb.SetComputeIntParam(shader, "_DtLocalVertexCount", layer0ActiveCount);
            cb.SetComputeIntParam(shader, "_OwnerCount", collisionOwnerCount);
            cb.SetComputeIntParam(shader, "_MaxBoundaryEdgesPerOwner", maxBoundaryEdgesPerOwner);
            cb.SetComputeIntParam(shader, "_MaxEdgesPerBin", maxEdgesPerBin);
            cb.SetComputeIntParam(shader, "_MaxVertsPerBin", maxVertsPerBin);
            cb.SetComputeIntParam(shader, "_MaxFineNodeContacts", collisionEvents != null ? collisionEvents.count : 0);
            cb.SetComputeIntParam(shader, "_MaxGridDimX", SdfResolution);
            cb.SetComputeIntParam(shader, "_MaxGridDimY", SdfResolution);
            cb.SetComputeIntParam(shader, "_QueryPairCount", queryPairCount);
            cb.SetComputeIntParam(shader, "_QuerySwap", 0);
            cb.SetComputeIntParam(shader, "_FineContactCount", collisionEvents != null ? collisionEvents.count : 0);
            cb.SetComputeIntParam(shader, "_ActiveCount", layer0ActiveCount);
            cb.SetComputeIntParam(shader, "_CoarseParentsPerNode", 2);
            cb.SetComputeIntParam(shader, "_CoarseNodeContactStride", 4);

            cb.SetComputeFloatParam(shader, "_LayerKernelH", layer0KernelH);
            cb.SetComputeFloatParam(shader, "_CollisionSupportScale", Const.CollisionSupportScale);
            cb.SetComputeFloatParam(shader, "_SdfBandWorld", math.max(Const.CollisionSdfBandHalfWidthScale * layer0KernelH, 1e-4f));
            cb.SetComputeFloatParam(shader, "_SdfFar", 1e6f);
            cb.SetComputeFloatParam(shader, "_SdfEps", 1e-8f);
            cb.SetComputeFloatParam(shader, "_OwnerBinSizeScale", 2f);

            int[] kernels = {
                kClearState,
                kBuildBoundaryFeatures,
                kBinBoundaryEdges,
                kBinBoundaryVertices,
                kBuildOwnerFeatureField,
                kQueryNodeSurfaceContacts,
                kQueryEdgeEdgeNodeContacts,
                kClearFineNodeManifolds,
                kScatterFineNodeContactsToManifolds,
            };

            ComputeBuffer globalVertexByLocal = dtGlobalNodeMap;
            if (globalVertexByLocal == null || !globalVertexByLocal.IsValid())
                globalVertexByLocal = EnsureIdentityGlobalNodeMap(layer0ActiveCount);
            for (int i = 0; i < kernels.Length; i++)
                BindKernelBuffers(cb, kernels[i], layer0Dt, dtReadSlot, dtOwnerByLocal, globalVertexByLocal, dtGlobalToLayerLocalMap, useDtGlobalNodeMap);

            BindTransferKernelBuffers(cb, kClearCoarseNodeContacts);
            BindTransferKernelBuffers(cb, kPropagateFineContactsToCoarse);

            _ = layer0NeighborSearch;
            _ = dtGlobalToLayerLocalMap;
        }
    }
}
