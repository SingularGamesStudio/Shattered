using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace GPU.Delaunay {
    /// <summary>
    /// Manages compute-based graph coloring for Delaunay triangulation vertices.
    /// Uses a parallel greedy coloring algorithm with 16 colors and 2-hop conflict detection.
    /// </summary>
    public sealed class DTColoring : IDisposable {
        // Compute shader and kernel indices
        readonly ComputeShader shader;
        readonly bool ownsShaderInstance;

        readonly int kInitTriGrid;
        readonly int kClearWork;
        readonly int kBuildWorkListFromDirty;
        readonly int kBuildWorkListFromConflicts;
        readonly int kBuildWorkArgs;
        readonly int kDetectConflictsWork;
        readonly int kChooseWork;
        readonly int kApplyWork;
        readonly int kClearMeta;
        readonly int kBuildCounts;
        readonly int kBuildStarts;
        readonly int kScatterOrder;
        readonly int kBuildRelaxArgs;
        readonly int kCountFinalConflicts;

        // Core buffers
        ComputeBuffer color;          // Current color per vertex (-1 or 0..15)
        ComputeBuffer proposed;       // Proposed new color during iteration
        ComputeBuffer prio;            // Priority per vertex

        // Work‑list construction
        ComputeBuffer markEpoch;       // Epoch stamp per vertex for duplicate prevention
        ComputeBuffer workCount;       // [0] = number of items in work list
        ComputeBuffer workList;        // Vertex indices (size = activeCount)
        ComputeBuffer workArgs;        // Indirect dispatch arguments (uint3)

        // Per‑color metadata for scattering / ordering
        ComputeBuffer counts;          // Number of vertices per color (size 16)
        ComputeBuffer starts;           // Start index in orderOut per color (size 16)
        ComputeBuffer write;            // Running write pointer during scatter (size 16)
        ComputeBuffer orderOut;         // Final vertex indices sorted by color

        // Indirect arguments for relaxation passes (one uint3 per color)
        ComputeBuffer relaxArgs;

        // Debug counter (total conflicts)
        ComputeBuffer debug;

        // Configuration state
        int activeCount;
        int dtNeighborCount;
        uint epoch;
        uint seed;

        // Cached dispatch group sizes
        const int ThreadGroupSize = 256;
        int activeGroups;   // (activeCount + ThreadGroupSize - 1) / ThreadGroupSize

        // Public properties for external access (read‑only buffers)
        public ComputeBuffer DebugBuffer => debug;
        public ComputeBuffer ColorBuffer => color;
        public ComputeBuffer OrderBuffer => orderOut;
        public ComputeBuffer RelaxArgsBuffer => relaxArgs;
        public ComputeBuffer StartsBuffer => starts;
        public ComputeBuffer CountsBuffer => counts;

        /// <summary>Creates a new instance, cloning the given compute shader.</summary>
        public DTColoring(ComputeShader shader) {
            if (!shader) throw new ArgumentNullException(nameof(shader));
            // Clone to allow per‑instance parameter modifications without affecting others.
            this.shader = UnityEngine.Object.Instantiate(shader);
            ownsShaderInstance = true;

            // Resolve kernel indices once.
            kInitTriGrid = this.shader.FindKernel("ColoringInitTriGrid");
            kClearWork = this.shader.FindKernel("ColoringClearWork");
            kBuildWorkListFromDirty = this.shader.FindKernel("ColoringBuildWorkListFromDirty");
            kBuildWorkListFromConflicts = this.shader.FindKernel("ColoringBuildWorkListFromConflicts");
            kBuildWorkArgs = this.shader.FindKernel("ColoringBuildWorkArgs");
            kDetectConflictsWork = this.shader.FindKernel("ColoringDetectConflictsWork");
            kChooseWork = this.shader.FindKernel("ColoringChooseWork");
            kApplyWork = this.shader.FindKernel("ColoringApplyWork");
            kClearMeta = this.shader.FindKernel("ColoringClearMeta");
            kBuildCounts = this.shader.FindKernel("ColoringBuildCounts");
            kBuildStarts = this.shader.FindKernel("ColoringBuildStarts");
            kScatterOrder = this.shader.FindKernel("ColoringScatterOrder");
            kBuildRelaxArgs = this.shader.FindKernel("ColoringBuildRelaxArgs");
            kCountFinalConflicts = this.shader.FindKernel("ColoringCountFinalConflicts");
        }

        /// <summary>
        /// Initialises or resizes the internal buffers.
        /// Must be called before any enqueue methods.
        /// </summary>
        /// <param name="activeCount">Number of active vertices.</param>
        /// <param name="dtNeighborCount">Maximum neighbour count per vertex (padded).</param>
        /// <param name="seed">Random seed for priority generation.</param>
        public void Init(int activeCount, int dtNeighborCount, uint seed) {
            if (activeCount <= 0) throw new ArgumentOutOfRangeException(nameof(activeCount));
            if (dtNeighborCount <= 0) throw new ArgumentOutOfRangeException(nameof(dtNeighborCount));

            DisposeBuffers();

            this.activeCount = activeCount;
            this.dtNeighborCount = dtNeighborCount;
            this.seed = seed;
            epoch = 1u;
            activeGroups = (activeCount + ThreadGroupSize - 1) / ThreadGroupSize;

            debug = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Structured);

            color = new ComputeBuffer(activeCount, sizeof(int), ComputeBufferType.Structured);
            proposed = new ComputeBuffer(activeCount, sizeof(int), ComputeBufferType.Structured);
            prio = new ComputeBuffer(activeCount, sizeof(uint), ComputeBufferType.Structured);

            markEpoch = new ComputeBuffer(activeCount, sizeof(uint), ComputeBufferType.Structured);
            // Initialise to zero (all epochs different from first epoch = 1)
            markEpoch.SetData(new uint[activeCount]);

            workCount = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Structured);
            workList = new ComputeBuffer(activeCount, sizeof(uint), ComputeBufferType.Structured);
            workArgs = new ComputeBuffer(3, sizeof(uint), ComputeBufferType.IndirectArguments);

            counts = new ComputeBuffer(16, sizeof(uint), ComputeBufferType.Structured);
            starts = new ComputeBuffer(16, sizeof(uint), ComputeBufferType.Structured);
            write = new ComputeBuffer(16, sizeof(uint), ComputeBufferType.Structured);
            orderOut = new ComputeBuffer(activeCount, sizeof(uint), ComputeBufferType.Structured);

            relaxArgs = new ComputeBuffer(16 * 3, sizeof(uint), ComputeBufferType.IndirectArguments);
        }

        /// <summary>
        /// Enqueues commands to initialise colours from a triangular grid pattern.
        /// Should be called once after vertices are placed.
        /// </summary>
        /// <param name="cb">Command buffer to append to.</param>
        /// <param name="positions">Buffer of vertex positions (float2).</param>
        /// <param name="dt">Delaunay triangulation data (needed for neighbour info).</param>
        /// <param name="layerCellSize">Grid cell size for the triangular pattern.</param>
        public void EnqueueInitTriGrid(CommandBuffer cb, ComputeBuffer positions, DT dt, float layerCellSize) {
            if (cb == null) throw new ArgumentNullException(nameof(cb));
            if (positions == null) throw new ArgumentNullException(nameof(positions));
            if (color == null) throw new InvalidOperationException("DTColoring.Init must be called first.");

            SetCommonParams(cb, positions, layerCellSize);

            cb.SetComputeBufferParam(shader, kInitTriGrid, "_ColoringColor", color);
            cb.SetComputeBufferParam(shader, kInitTriGrid, "_ColoringProposed", proposed);
            cb.SetComputeBufferParam(shader, kInitTriGrid, "_ColoringPrio", prio);

            cb.DispatchCompute(shader, kInitTriGrid, activeGroups, 1, 1);

            EnqueueRebuildOrderAndArgs(cb);
        }

        /// <summary>
        /// Enqueues one or more coloring iterations after the Delaunay structure has been maintained.
        /// Uses dirty flags from the DT to focus work, then performs multiple conflict‑detection passes.
        /// </summary>
        /// <param name="cb">Command buffer to append to.</param>
        /// <param name="positions">Vertex positions (float2).</param>
        /// <param name="dt">Delaunay triangulation (neighbors, counts, dirty flags).</param>
        /// <param name="layerCellSize">Grid cell size (unused in iterations but passed for consistency).</param>
        /// <param name="iterations">Number of coloring iterations to perform.</param>
        public void EnqueueUpdateAfterMaintain(
            CommandBuffer cb,
            ComputeBuffer positions,
            DT dt,
            float layerCellSize,
            int iterations
        ) {
            if (cb == null) throw new ArgumentNullException(nameof(cb));
            if (positions == null) throw new ArgumentNullException(nameof(positions));
            if (dt == null) throw new ArgumentNullException(nameof(dt));
            if (color == null) throw new InvalidOperationException("DTColoring.Init must be called first.");
            if (iterations <= 0) throw new ArgumentOutOfRangeException(nameof(iterations));

            SetCommonParams(cb, positions, layerCellSize);

            // Bind neighbour buffers (needed by several kernels)
            cb.SetComputeBufferParam(shader, kBuildWorkListFromDirty, "_ColoringDtNeighbors", dt.NeighborsBuffer);
            cb.SetComputeBufferParam(shader, kBuildWorkListFromDirty, "_ColoringDtNeighborCounts", dt.NeighborCountsBuffer);
            cb.SetComputeBufferParam(shader, kBuildWorkListFromDirty, "_ColoringDtDirtyFlags", dt.DirtyVertexFlagsBuffer);

            cb.SetComputeBufferParam(shader, kBuildWorkListFromConflicts, "_ColoringDtNeighbors", dt.NeighborsBuffer);
            cb.SetComputeBufferParam(shader, kBuildWorkListFromConflicts, "_ColoringDtNeighborCounts", dt.NeighborCountsBuffer);

            cb.SetComputeBufferParam(shader, kChooseWork, "_ColoringDtNeighbors", dt.NeighborsBuffer);
            cb.SetComputeBufferParam(shader, kChooseWork, "_ColoringDtNeighborCounts", dt.NeighborCountsBuffer);
            cb.SetComputeBufferParam(shader, kApplyWork, "_ColoringDtNeighbors", dt.NeighborsBuffer);
            cb.SetComputeBufferParam(shader, kApplyWork, "_ColoringDtNeighborCounts", dt.NeighborCountsBuffer);

            // Bind coloring buffers for all kernels that need them.
            BindColoringBuffers(cb, kBuildWorkListFromDirty);
            BindColoringBuffers(cb, kBuildWorkListFromConflicts);
            BindColoringBuffers(cb, kChooseWork);
            BindColoringBuffers(cb, kApplyWork);

            // Set up work‑list control kernels.
            cb.SetComputeBufferParam(shader, kClearWork, "_ColoringWorkCount", workCount);
            cb.SetComputeBufferParam(shader, kClearWork, "_ColoringDebug", debug);
            cb.SetComputeBufferParam(shader, kBuildWorkArgs, "_ColoringWorkCount", workCount);
            cb.SetComputeBufferParam(shader, kBuildWorkArgs, "_ColoringWorkArgs", workArgs);

            // First pass: build work list from dirty vertices (wake‑up region)
            epoch++;
            cb.SetComputeIntParam(shader, "_ColoringEpoch", unchecked((int)epoch));
            cb.DispatchCompute(shader, kClearWork, 1, 1, 1);
            cb.DispatchCompute(shader, kBuildWorkListFromDirty, activeGroups, 1, 1);
            cb.DispatchCompute(shader, kBuildWorkArgs, 1, 1, 1);

            for (int i = 0; i < iterations; i++) {
                // Rebuild work list from conflicts (global) using new epoch
                epoch++;
                cb.SetComputeIntParam(shader, "_ColoringEpoch", unchecked((int)epoch));

                cb.DispatchCompute(shader, kClearWork, 1, 1, 1);
                cb.DispatchCompute(shader, kBuildWorkListFromConflicts, activeGroups, 1, 1);
                cb.DispatchCompute(shader, kBuildWorkArgs, 1, 1, 1);

                // Choose new colours and apply them (indirect dispatch)
                cb.DispatchCompute(shader, kChooseWork, workArgs, 0);
                cb.DispatchCompute(shader, kApplyWork, workArgs, 0);
            }

            // Count remaining conflicts for debugging / convergence check
            cb.DispatchCompute(shader, kClearWork, 1, 1, 1);

            cb.SetComputeBufferParam(shader, kCountFinalConflicts, "_ColoringColor", color);
            cb.SetComputeBufferParam(shader, kCountFinalConflicts, "_ColoringPrio", prio);
            cb.SetComputeBufferParam(shader, kCountFinalConflicts, "_ColoringDtNeighbors", dt.NeighborsBuffer);
            cb.SetComputeBufferParam(shader, kCountFinalConflicts, "_ColoringDtNeighborCounts", dt.NeighborCountsBuffer);
            cb.SetComputeIntParam(shader, "_ColoringActiveCount", activeCount);
            cb.SetComputeIntParam(shader, "_ColoringDtNeighborCount", dtNeighborCount);
            cb.SetComputeBufferParam(shader, kCountFinalConflicts, "_ColoringDebug", debug);
            cb.DispatchCompute(shader, kCountFinalConflicts, activeGroups, 1, 1);

            // Rebuild the ordered vertex list and indirect arguments for relaxation
            EnqueueRebuildOrderAndArgs(cb);
        }

        /// <summary>
        /// Rebuilds the vertex order buffer (sorted by colour) and the indirect dispatch arguments
        /// for relaxation passes. Called automatically after initialisation and after updates.
        /// </summary>
        public void EnqueueRebuildOrderAndArgs(CommandBuffer cb) {
            if (cb == null) throw new ArgumentNullException(nameof(cb));
            if (color == null) throw new InvalidOperationException("DTColoring.Init must be called first.");

            // Clear per‑colour metadata
            cb.SetComputeBufferParam(shader, kClearMeta, "_ColoringCounts", counts);
            cb.SetComputeBufferParam(shader, kClearMeta, "_ColoringStarts", starts);
            cb.SetComputeBufferParam(shader, kClearMeta, "_ColoringWrite", write);
            cb.DispatchCompute(shader, kClearMeta, 1, 1, 1);

            // Count vertices per colour
            cb.SetComputeBufferParam(shader, kBuildCounts, "_ColoringColor", color);
            cb.SetComputeBufferParam(shader, kBuildCounts, "_ColoringCounts", counts);
            cb.SetComputeIntParam(shader, "_ColoringActiveCount", activeCount);
            cb.SetComputeIntParam(shader, "_ColoringMaxColors", 16);
            cb.DispatchCompute(shader, kBuildCounts, activeGroups, 1, 1);

            // Prefix sum to compute start indices
            cb.SetComputeBufferParam(shader, kBuildStarts, "_ColoringCounts", counts);
            cb.SetComputeBufferParam(shader, kBuildStarts, "_ColoringStarts", starts);
            cb.SetComputeBufferParam(shader, kBuildStarts, "_ColoringWrite", write);
            cb.SetComputeIntParam(shader, "_ColoringMaxColors", 16);
            cb.DispatchCompute(shader, kBuildStarts, 1, 1, 1);

            // Scatter vertices into orderOut
            cb.SetComputeBufferParam(shader, kScatterOrder, "_ColoringColor", color);
            cb.SetComputeBufferParam(shader, kScatterOrder, "_ColoringWrite", write);
            cb.SetComputeBufferParam(shader, kScatterOrder, "_ColoringOrderOut", orderOut);
            cb.SetComputeIntParam(shader, "_ColoringActiveCount", activeCount);
            cb.SetComputeIntParam(shader, "_ColoringMaxColors", 16);
            cb.DispatchCompute(shader, kScatterOrder, activeGroups, 1, 1);

            // Build indirect arguments for relaxation (one set per colour)
            cb.SetComputeBufferParam(shader, kBuildRelaxArgs, "_ColoringCounts", counts);
            cb.SetComputeBufferParam(shader, kBuildRelaxArgs, "_RelaxArgs", relaxArgs);
            cb.SetComputeIntParam(shader, "_ColoringMaxColors", 16);
            cb.DispatchCompute(shader, kBuildRelaxArgs, 1, 1, 1);
        }

        // Sets parameters common to most kernels (positions, counts, seed, etc.)
        void SetCommonParams(CommandBuffer cb, ComputeBuffer positions, float layerCellSize) {
            // Use SetComputeIntParams where possible to reduce API calls (but only for contiguous arrays)
            // Here we only have a few ints, so separate calls are fine.
            cb.SetComputeIntParam(shader, "_ColoringActiveCount", activeCount);
            cb.SetComputeIntParam(shader, "_ColoringDtNeighborCount", dtNeighborCount);
            cb.SetComputeIntParam(shader, "_ColoringMaxColors", 16);
            cb.SetComputeIntParam(shader, "_ColoringSeed", unchecked((int)seed));
            cb.SetComputeIntParam(shader, "_ColoringEpoch", unchecked((int)epoch));
            cb.SetComputeFloatParam(shader, "_ColoringLayerCellSize", layerCellSize);

            cb.SetComputeBufferParam(shader, kInitTriGrid, "_Positions", positions);
        }

        // Binds the core coloring buffers to a specific kernel.
        void BindColoringBuffers(CommandBuffer cb, int kernel) {
            cb.SetComputeBufferParam(shader, kernel, "_ColoringColor", color);
            cb.SetComputeBufferParam(shader, kernel, "_ColoringProposed", proposed);
            cb.SetComputeBufferParam(shader, kernel, "_ColoringPrio", prio);
            cb.SetComputeBufferParam(shader, kernel, "_ColoringMarkEpoch", markEpoch);
            cb.SetComputeBufferParam(shader, kernel, "_ColoringWorkCount", workCount);
            cb.SetComputeBufferParam(shader, kernel, "_ColoringWorkList", workList);
            cb.SetComputeBufferParam(shader, kernel, "_ColoringDebug", debug);
        }

        /// <summary>
        /// Asynchronously reads the conflict count from the debug buffer.
        /// </summary>
        public void ReadConflictCountAsync(Action<uint> callback) {
            if (debug == null) {
                callback?.Invoke(0);
                return;
            }
            AsyncGPUReadback.Request(debug, (request) => {
                if (request.hasError) {
                    Debug.LogError("Async readback of conflict count failed.");
                    callback?.Invoke(0);
                } else {
                    uint value = request.GetData<uint>()[0];
                    callback?.Invoke(value);
                }
            });
        }

        /// <summary>
        /// Synchronously reads the last conflict count from the debug buffer.
        /// </summary>
        public uint GetLastConflictCount() {
            if (debug == null) return 0;
            uint[] result = new uint[1];
            debug.GetData(result);
            return result[0];
        }

        public void Dispose() {
            DisposeBuffers();
            if (ownsShaderInstance && shader)
                UnityEngine.Object.Destroy(shader);
        }

        void DisposeBuffers() {
            color?.Dispose(); color = null;
            proposed?.Dispose(); proposed = null;
            prio?.Dispose(); prio = null;

            markEpoch?.Dispose(); markEpoch = null;
            workCount?.Dispose(); workCount = null;
            workList?.Dispose(); workList = null;
            workArgs?.Dispose(); workArgs = null;

            counts?.Dispose(); counts = null;
            starts?.Dispose(); starts = null;
            write?.Dispose(); write = null;
            orderOut?.Dispose(); orderOut = null;
            debug?.Dispose(); debug = null;

            relaxArgs?.Dispose(); relaxArgs = null;

            activeCount = 0;
            dtNeighborCount = 0;
            epoch = 0u;
            seed = 0u;
        }
    }
}