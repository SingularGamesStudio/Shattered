Shader "Unlit/TriangulationAccum"
{
    Properties
    {
        _UvScale("UV Scale", Float) = 0.25
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        Pass
        {
            ZWrite Off
            ZTest Always
            Cull Off
            Blend One One

            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex Vert
            #pragma fragment Frag
            #include "UnityCG.cginc"

            struct HalfEdge
            {
                int v;
                int next;
                int twin;
                int t;
            };

            StructuredBuffer<float2> _PositionsPrev;
            StructuredBuffer<float2> _PositionsCurr;
            float _RenderAlpha;

            StructuredBuffer<HalfEdge> _HalfEdges;
            StructuredBuffer<int> _TriToHE;
            StructuredBuffer<int> _OwnerByLocal;

            StructuredBuffer<int> _MaterialIds;
            int _MaterialCount;
            StructuredBuffer<float> _RestVolumes;

            StructuredBuffer<float2> _RestNormPositions;
            int _RealPointCount;
            float _LayerKernelH;
            float _WendlandSupportScale;

            UNITY_DECLARE_TEX2DARRAY(_AlbedoArray);

            float2 _NormCenter;
            float _NormInvHalfExtent;

            float _UvScale;

            static int Next(int he) { return _HalfEdges[he].next; }
            static int Dest(int he) { return _HalfEdges[Next(he)].v; }

            float2 GpuToWorld(float2 p) { return p / _NormInvHalfExtent + _NormCenter; }
            float Cross2(float2 a, float2 b) { return a.x * b.y - a.y * b.x; }
            float TriangleArea(float2 a, float2 b, float2 c) { return abs(Cross2(b - a, c - a)) * 0.5; }
            float2 SafeNormalize(float2 v)
            {
                float l2 = dot(v, v);
                return (l2 > 1e-12) ? (v * rsqrt(l2)) : float2(1, 0);
            }

            bool PatchTriangle(
            int sourceCorner,
            float support,
            float2 p0, float2 p1, float2 p2,
            float2 r0, float2 r1, float2 r2,
            float restV0, float restV1, float restV2,
            out float2 outPos,
            out float2 outRest,
            out float3 outBary)
            {
                const float kEpsLen = 1e-6;
                const float kEpsArea2 = 2e-10; // compares against |cross| (double-area), not TriangleArea()

                int localTri = sourceCorner / 3;
                int corner = sourceCorner - localTri * 3;

                float l01 = length(p1 - p0);
                float l12 = length(p2 - p1);
                float l20 = length(p0 - p2);

                bool b01 = l01 > support;
                bool b12 = l12 > support;
                bool b20 = l20 > support;
                int badCount = (b01 ? 1 : 0) + (b12 ? 1 : 0) + (b20 ? 1 : 0);

                float2 q0 = p0;
                float2 q1 = p1;
                float2 q2 = p2;
                outBary = (corner == 0) ? float3(1, 0, 0) : (corner == 1 ? float3(0, 1, 0) : float3(0, 0, 1));

                if (badCount == 0)
                {
                    if (localTri > 0) return false;
                    outPos = (corner == 0) ? q0 : (corner == 1 ? q1 : q2);
                    outRest = (corner == 0) ? r0 : (corner == 1 ? r1 : r2);
                    return true;
                }

                float areaRest = TriangleArea(r0, r1, r2);
                float areaNow = TriangleArea(p0, p1, p2);
                float deltaArea = max(0.0, areaRest - areaNow);

                if (badCount == 1)
                {
                    int ia = b01 ? 0 : (b12 ? 1 : 2);
                    int ib = b01 ? 1 : (b12 ? 2 : 0);
                    int ic = b01 ? 2 : (b12 ? 0 : 1);

                    float2 A = (ia == 0) ? p0 : (ia == 1 ? p1 : p2);
                    float2 B = (ib == 0) ? p0 : (ib == 1 ? p1 : p2);
                    float2 C = (ic == 0) ? p0 : (ic == 1 ? p1 : p2);
                    float2 RA = (ia == 0) ? r0 : (ia == 1 ? r1 : r2);
                    float2 RB = (ib == 0) ? r0 : (ib == 1 ? r1 : r2);
                    float2 RC = (ic == 0) ? r0 : (ic == 1 ? r1 : r2);

                    float2 mid = 0.5 * (A + B);
                    float L = max(length(B - A), kEpsLen);
                    float overshoot = max(0.0, L - support);
                    float t = saturate(overshoot / max(support, kEpsLen));
                    t = max(t, saturate(deltaArea / max(TriangleArea(p0, p1, p2), kEpsLen)));
                    float2 M = lerp(mid, C, 0.8 * t);
                    float2 RM = lerp(0.5 * (RA + RB), RC, 0.8 * t);

                    float2 tr0, tr1, tr2;
                    float3 tb0, tb1, tb2;

                    if (localTri > 1) return false;
                    if (localTri == 0)
                    {
                        q0 = A; q1 = M; q2 = C;
                        tr0 = RA; tr1 = RM; tr2 = RC;
                        tb0 = (ia == 0) ? float3(1, 0, 0) : (ia == 1) ? float3(0, 1, 0) : float3(0, 0, 1);
                        tb1 = 0.5 * ((ia == 0) ? float3(1, 0, 0) : (ia == 1) ? float3(0, 1, 0) : float3(0, 0, 1))
                        + 0.5 * ((ib == 0) ? float3(1, 0, 0) : (ib == 1) ? float3(0, 1, 0) : float3(0, 0, 1));
                        tb2 = (ic == 0) ? float3(1, 0, 0) : (ic == 1) ? float3(0, 1, 0) : float3(0, 0, 1);
                    }
                    else
                    {
                        q0 = M; q1 = B; q2 = C;
                        tr0 = RM; tr1 = RB; tr2 = RC;
                        tb0 = 0.5 * ((ia == 0) ? float3(1, 0, 0) : (ia == 1) ? float3(0, 1, 0) : float3(0, 0, 1))
                        + 0.5 * ((ib == 0) ? float3(1, 0, 0) : (ib == 1) ? float3(0, 1, 0) : float3(0, 0, 1));
                        tb1 = (ib == 0) ? float3(1, 0, 0) : (ib == 1) ? float3(0, 1, 0) : float3(0, 0, 1);
                        tb2 = (ic == 0) ? float3(1, 0, 0) : (ic == 1) ? float3(0, 1, 0) : float3(0, 0, 1);
                    }

                    outPos = (corner == 0) ? q0 : (corner == 1 ? q1 : q2);
                    outRest = (corner == 0) ? tr0 : (corner == 1 ? tr1 : tr2);
                    outBary = (corner == 0) ? tb0 : (corner == 1 ? tb1 : tb2);
                    return true;
                }

                if (badCount >= 2)
                {
                    // We always emit up to 5 micro-tris here and degenerate the ones we don't want,
                    // to avoid topology popping (no "return false" on skinny triangles).
                    if (localTri > 4) return false;

                    float invSupport = rcp(max(support, kEpsLen));
                    float s01 = l01 * invSupport;
                    float s12 = l12 * invSupport;
                    float s20 = l20 * invSupport;

                    // allBadT: continuous "we are approaching 3-bad" weight.
                    // It starts rising before badCount actually flips to 3 (which is discrete), preventing jumps.
                    float sMin = min(s01, min(s12, s20));
                    float allBadT = smoothstep(1.0, 1.0 + 0.25, sMin);

                    int ia, ib, ic;
                    float baseStretch = s01;
                    if (s01 <= s12 && s01 <= s20) { ia = 0; ib = 1; ic = 2; baseStretch = s01; }
                    else if (s12 <= s20) { ia = 1; ib = 2; ic = 0; baseStretch = s12; }
                    else { ia = 2; ib = 0; ic = 1; baseStretch = s20; }

                    float2 A = (ia == 0) ? p0 : (ia == 1 ? p1 : p2);
                    float2 B = (ib == 0) ? p0 : (ib == 1 ? p1 : p2);
                    float2 C = (ic == 0) ? p0 : (ic == 1 ? p1 : p2);

                    float2 RA = (ia == 0) ? r0 : (ia == 1 ? r1 : r2);
                    float2 RB = (ib == 0) ? r0 : (ib == 1 ? r1 : r2);
                    float2 RC = (ic == 0) ? r0 : (ic == 1 ? r1 : r2);

                    float support2 = support * support;

                    float L = length(B - A);
                    float halfL = 0.5 * L;
                    float hmax2 = sqrt(max(0.0, support2 - halfL * halfL));
                    float Amax2 = 0.5 * L * hmax2;                 // "2-bad" max strip area
                    float Amax3 = 0.4330127018922193 * support2;    // equilateral area ~ sqrt(3)/4 * s^2

                    // Blend the capacity model smoothly so we don't snap when the 3rd edge becomes "bad".
                    float Amax = lerp(Amax2, Amax3, allBadT);
                    if (Amax <= 1e-8)
                    {
                        if (localTri > 0) return false;
                        outPos = (corner == 0) ? p0 : (corner == 1 ? p1 : p2);
                        outRest = (corner == 0) ? r0 : (corner == 1 ? r1 : r2);
                        outBary = (corner == 0) ? float3(1, 0, 0) : (corner == 1 ? float3(0, 1, 0) : float3(0, 0, 1));
                        return true;
                    }

                    float minBadStretch = 1e9;
                    if (b01) minBadStretch = min(minBadStretch, s01);
                    if (b12) minBadStretch = min(minBadStretch, s12);
                    if (b20) minBadStretch = min(minBadStretch, s20);

                    // t3 ~ "how 3-bad are we" driven by overstretch, then pushed toward 1 as allBadT rises.
                    float t3raw = saturate((minBadStretch - 1.0) / 0.25);
                    float t3 = saturate(lerp(t3raw, 1.0, allBadT));

                    // lambdaStrip is the strip portion (goes to 0 as we become fully 3-bad).
                    float lambdaStrip = 1.0 - t3;
                    float Astrip = lambdaStrip * Amax;

                    float triAreaNow = TriangleArea(A, B, C);
                    float invTriAreaNow = rcp(max(triAreaNow, 1e-8));

                    float stripRatio = saturate(Astrip * invTriAreaNow);
                    float stripT = 1.0 - sqrt(max(0.0, 1.0 - stripRatio));

                    // Strip cut points (A->C and B->C). When Astrip→0 they collapse back toward A/B (degenerate).
                    float2 QA = lerp(A, C, stripT);
                    float2 QB = lerp(B, C, stripT);
                    float2 RQA = lerp(RA, RC, stripT);
                    float2 RQB = lerp(RB, RC, stripT);

                    float3 rv;
                    rv.x = max(restV0, 1e-8);
                    rv.y = max(restV1, 1e-8);
                    rv.z = max(restV2, 1e-8);
                    float rA = (ia == 0) ? rv.x : (ia == 1) ? rv.y : rv.z;
                    float rBv = (ib == 0) ? rv.x : (ib == 1) ? rv.y : rv.z;
                    float rC = (ic == 0) ? rv.x : (ic == 1) ? rv.y : rv.z;

                    float AblobTotal = max(0.0, Amax - stripRatio * triAreaNow);

                    // Blob weights: 2-bad sends blobs to C only; 3-bad distributes by rest volumes.
                    float sumABC = max(rA + rBv + rC, 1e-8);
                    float wA = lerp(0.0, rA / sumABC, allBadT);
                    float wB = lerp(0.0, rBv / sumABC, allBadT);
                    float wC = lerp(rC / sumABC, rC / sumABC, allBadT);

                    float AblobA = AblobTotal * wA;
                    float AblobB = AblobTotal * wB;
                    float AblobC = AblobTotal * wC;

                    float sinA = abs(Cross2(SafeNormalize(B - A), SafeNormalize(C - A)));
                    float sinB = abs(Cross2(SafeNormalize(A - B), SafeNormalize(C - B)));
                    float sinC = abs(Cross2(SafeNormalize(A - C), SafeNormalize(B - C)));

                    float cutA = sqrt((2.0 * AblobA) / max(sinA, 1e-4));
                    float cutB = sqrt((2.0 * AblobB) / max(sinB, 1e-4));
                    float cutC = sqrt((2.0 * AblobC) / max(sinC, 1e-4));

                    cutA = min(cutA, 0.49 * min(length(A - B), length(A - C)));
                    cutB = min(cutB, 0.49 * min(length(B - A), length(B - C)));
                    cutC = min(cutC, 0.49 * min(length(C - A), length(C - B)));

                    float rCutA = min(cutA, 0.49 * min(length(RA - RB), length(RA - RC)));
                    float rCutB = min(cutB, 0.49 * min(length(RB - RA), length(RB - RC)));
                    float rCutC = min(cutC, 0.49 * min(length(RC - RA), length(RC - RB)));

                    float2 Aab = A + SafeNormalize(B - A) * cutA;
                    float2 Aac = A + SafeNormalize(C - A) * cutA;
                    float2 Bab = B + SafeNormalize(A - B) * cutB;
                    float2 Bbc = B + SafeNormalize(C - B) * cutB;
                    float2 Cca = C + SafeNormalize(A - C) * cutC;
                    float2 Ccb = C + SafeNormalize(B - C) * cutC;

                    float2 RAab = RA + SafeNormalize(RB - RA) * rCutA;
                    float2 RAac = RA + SafeNormalize(RC - RA) * rCutA;
                    float2 RBab = RB + SafeNormalize(RA - RB) * rCutB;
                    float2 RBbc = RB + SafeNormalize(RC - RB) * rCutB;
                    float2 RCca = RC + SafeNormalize(RA - RC) * rCutC;
                    float2 RCcb = RC + SafeNormalize(RB - RC) * rCutC;

                    float3 maskA = (ia == 0) ? float3(1, 0, 0) : (ia == 1) ? float3(0, 1, 0) : float3(0, 0, 1);
                    float3 maskB = (ib == 0) ? float3(1, 0, 0) : (ib == 1) ? float3(0, 1, 0) : float3(0, 0, 1);
                    float3 maskC = (ic == 0) ? float3(1, 0, 0) : (ic == 1) ? float3(0, 1, 0) : float3(0, 0, 1);

                    // "2-bad body" layout:
                    // - localTri 0..1: body quad split
                    // - localTri 2: A-side filler
                    // - localTri 3: B-side filler
                    // - localTri 4: unused (kept degenerate)
                    float edgeT = smoothstep(0.5, 1.0, saturate(baseStretch));
                    float sumAB = max(rA + rBv, 1e-8);
                    float wAB_A = rA / sumAB;
                    float wAB_B = rBv / sumAB;

                    float aCut = edgeT * wAB_A;
                    float bCut = edgeT * wAB_B;

                    float bodyRatio = saturate(Amax * invTriAreaNow);
                    float bodyCut = 1.0 - sqrt(max(0.0, 1.0 - bodyRatio));

                    float2 PA = lerp(A, B, aCut);
                    float2 PB = lerp(B, A, bCut);
                    float2 QA2 = lerp(A, C, bodyCut);
                    float2 QB2 = lerp(B, C, bodyCut);

                    float2 RPA = lerp(RA, RB, aCut);
                    float2 RPB = lerp(RB, RA, bCut);
                    float2 RQA2 = lerp(RA, RC, bodyCut);
                    float2 RQB2 = lerp(RB, RC, bodyCut);

                    float3 bPA = lerp(maskA, maskB, aCut);
                    float3 bPB = lerp(maskB, maskA, bCut);
                    float3 bQA2 = lerp(maskA, maskC, bodyCut);
                    float3 bQB2 = lerp(maskB, maskC, bodyCut);

                    // We build both layouts into a fixed 5-triangle indexing and lerp per-vertex:
                    // - q*_2: "2-bad body" geometry
                    // - q*_3: "3-bad strip + corner blobs" geometry
                    // This makes topology stable and avoids popping from "triangle disappears" events.
                    float2 q0_2 = A, q1_2 = A, q2_2 = A;
                    float2 tr0_2 = RA, tr1_2 = RA, tr2_2 = RA;
                    float3 tb0_2 = maskA, tb1_2 = maskA, tb2_2 = maskA;

                    float2 q0_3 = A, q1_3 = A, q2_3 = A;
                    float2 tr0_3 = RA, tr1_3 = RA, tr2_3 = RA;
                    float3 tb0_3 = maskA, tb1_3 = maskA, tb2_3 = maskA;

                    if (localTri == 0)
                    {
                        q0_2 = PA; q1_2 = PB; q2_2 = QB2;
                        tr0_2 = RPA; tr1_2 = RPB; tr2_2 = RQB2;
                        tb0_2 = bPA; tb1_2 = bPB; tb2_2 = bQB2;
                        if (abs(Cross2(q1_2 - q0_2, q2_2 - q0_2)) <= kEpsArea2)
                        {
                            q0_2 = PA; q1_2 = PA; q2_2 = PA;
                            tr0_2 = RPA; tr1_2 = RPA; tr2_2 = RPA;
                            tb0_2 = bPA; tb1_2 = bPA; tb2_2 = bPA;
                        }

                        q0_3 = A; q1_3 = B; q2_3 = QB;
                        tr0_3 = RA; tr1_3 = RB; tr2_3 = RQB;
                        tb0_3 = maskA; tb1_3 = maskB; tb2_3 = maskB;
                        if (abs(Cross2(q1_3 - q0_3, q2_3 - q0_3)) <= kEpsArea2)
                        {
                            q0_3 = A; q1_3 = A; q2_3 = A;
                            tr0_3 = RA; tr1_3 = RA; tr2_3 = RA;
                            tb0_3 = maskA; tb1_3 = maskA; tb2_3 = maskA;
                        }
                    }
                    else if (localTri == 1)
                    {
                        q0_2 = PA; q1_2 = QB2; q2_2 = QA2;
                        tr0_2 = RPA; tr1_2 = RQB2; tr2_2 = RQA2;
                        tb0_2 = bPA; tb1_2 = bQB2; tb2_2 = bQA2;
                        if (abs(Cross2(q1_2 - q0_2, q2_2 - q0_2)) <= kEpsArea2)
                        {
                            q0_2 = PA; q1_2 = PA; q2_2 = PA;
                            tr0_2 = RPA; tr1_2 = RPA; tr2_2 = RPA;
                            tb0_2 = bPA; tb1_2 = bPA; tb2_2 = bPA;
                        }

                        q0_3 = A; q1_3 = QB; q2_3 = QA;
                        tr0_3 = RA; tr1_3 = RQB; tr2_3 = RQA;
                        tb0_3 = maskA; tb1_3 = maskB; tb2_3 = maskA;
                        if (abs(Cross2(q1_3 - q0_3, q2_3 - q0_3)) <= kEpsArea2)
                        {
                            q0_3 = A; q1_3 = A; q2_3 = A;
                            tr0_3 = RA; tr1_3 = RA; tr2_3 = RA;
                            tb0_3 = maskA; tb1_3 = maskA; tb2_3 = maskA;
                        }
                    }
                    else if (localTri == 2)
                    {
                        q0_2 = A; q1_2 = PA; q2_2 = QA2;
                        tr0_2 = RA; tr1_2 = RPA; tr2_2 = RQA2;
                        tb0_2 = maskA; tb1_2 = bPA; tb2_2 = bQA2;
                        if (abs(Cross2(q1_2 - q0_2, q2_2 - q0_2)) <= kEpsArea2)
                        {
                            q0_2 = A; q1_2 = A; q2_2 = A;
                            tr0_2 = RA; tr1_2 = RA; tr2_2 = RA;
                            tb0_2 = maskA; tb1_2 = maskA; tb2_2 = maskA;
                        }

                        if (AblobA <= 1e-10 || abs(Cross2(Aab - A, Aac - A)) <= kEpsArea2)
                        {
                            q0_3 = A; q1_3 = A; q2_3 = A;
                            tr0_3 = RA; tr1_3 = RA; tr2_3 = RA;
                            tb0_3 = maskA; tb1_3 = maskA; tb2_3 = maskA;
                        }
                        else
                        {
                            q0_3 = A; q1_3 = Aab; q2_3 = Aac;
                            tr0_3 = RA; tr1_3 = RAab; tr2_3 = RAac;
                            tb0_3 = maskA; tb1_3 = maskA; tb2_3 = maskA;
                        }
                    }
                    else if (localTri == 3)
                    {
                        q0_2 = B; q1_2 = QB2; q2_2 = PB;
                        tr0_2 = RB; tr1_2 = RQB2; tr2_2 = RPB;
                        tb0_2 = maskB; tb1_2 = bQB2; tb2_2 = bPB;
                        if (abs(Cross2(q1_2 - q0_2, q2_2 - q0_2)) <= kEpsArea2)
                        {
                            q0_2 = B; q1_2 = B; q2_2 = B;
                            tr0_2 = RB; tr1_2 = RB; tr2_2 = RB;
                            tb0_2 = maskB; tb1_2 = maskB; tb2_2 = maskB;
                        }

                        if (AblobB <= 1e-10 || abs(Cross2(Bab - B, Bbc - B)) <= kEpsArea2)
                        {
                            q0_3 = B; q1_3 = B; q2_3 = B;
                            tr0_3 = RB; tr1_3 = RB; tr2_3 = RB;
                            tb0_3 = maskB; tb1_3 = maskB; tb2_3 = maskB;
                        }
                        else
                        {
                            q0_3 = B; q1_3 = Bab; q2_3 = Bbc;
                            tr0_3 = RB; tr1_3 = RBab; tr2_3 = RBbc;
                            tb0_3 = maskB; tb1_3 = maskB; tb2_3 = maskB;
                        }
                    }
                    else
                    {
                        // localTri == 4:
                        // Keep the opposite-corner cap (C-blob) ALWAYS present
                        // Rationale: once edges are "bad" we treat force transmission as broken, so C must remain decoupled
                        // from what happens along the A-B side (no topology popping / disappearing cap).
                        if (AblobC <= 1e-10 || abs(Cross2(Cca - C, Ccb - C)) <= kEpsArea2)
                        {
                            q0_2 = C; q1_2 = C; q2_2 = C;
                            tr0_2 = RC; tr1_2 = RC; tr2_2 = RC;
                            tb0_2 = maskC; tb1_2 = maskC; tb2_2 = maskC;

                            q0_3 = C; q1_3 = C; q2_3 = C;
                            tr0_3 = RC; tr1_3 = RC; tr2_3 = RC;
                            tb0_3 = maskC; tb1_3 = maskC; tb2_3 = maskC;
                        }
                        else
                        {
                            // Identical geometry in both layouts → survives regardless of allBadT.
                            q0_2 = C; q1_2 = Cca; q2_2 = Ccb;
                            tr0_2 = RC; tr1_2 = RCca; tr2_2 = RCcb;
                            tb0_2 = maskC; tb1_2 = maskC; tb2_2 = maskC;

                            q0_3 = C; q1_3 = Cca; q2_3 = Ccb;
                            tr0_3 = RC; tr1_3 = RCca; tr2_3 = RCcb;
                            tb0_3 = maskC; tb1_3 = maskC; tb2_3 = maskC;
                        }
                    }


                    float2 fq0 = lerp(q0_2, q0_3, allBadT);
                    float2 fq1 = lerp(q1_2, q1_3, allBadT);
                    float2 fq2 = lerp(q2_2, q2_3, allBadT);

                    float2 ftr0 = lerp(tr0_2, tr0_3, allBadT);
                    float2 ftr1 = lerp(tr1_2, tr1_3, allBadT);
                    float2 ftr2 = lerp(tr2_2, tr2_3, allBadT);

                    float3 ftb0 = lerp(tb0_2, tb0_3, allBadT);
                    float3 ftb1 = lerp(tb1_2, tb1_3, allBadT);
                    float3 ftb2 = lerp(tb2_2, tb2_3, allBadT);

                    outPos = (corner == 0) ? fq0 : (corner == 1 ? fq1 : fq2);
                    outRest = (corner == 0) ? ftr0 : (corner == 1 ? ftr1 : ftr2);
                    outBary = (corner == 0) ? ftb0 : (corner == 1 ? ftb1 : ftb2);
                    return true;
                }

                if (badCount >= 3)
                return false;

                if (localTri > 0) return false;
                outPos = (corner == 0) ? p0 : (corner == 1 ? p1 : p2);
                outRest = (corner == 0) ? r0 : (corner == 1 ? r1 : r2);
                outBary = (corner == 0) ? float3(1, 0, 0) : (corner == 1 ? float3(0, 1, 0) : float3(0, 0, 1));
                return true;
            }

            int ClampMaterial(int m)
            {
                if (_MaterialCount <= 0) return 0;
                return clamp(m, 0, _MaterialCount - 1);
            }

            struct v2f
            {
                float4 pos : SV_POSITION;
                float2 restNorm : TEXCOORD0;
                float3 bary : TEXCOORD1;
                nointerpolation int3 mats : TEXCOORD2;
                nointerpolation int valid : TEXCOORD3;
            };

            v2f Vert(uint vid : SV_VertexID)
            {
                v2f o;

                uint tri = vid / 15;
                uint sourceCorner = vid - tri * 15;

                int he0 = _TriToHE[tri];
                if (he0 < 0)
                {
                    o.pos = 0;
                    o.restNorm = 0;
                    o.bary = 0;
                    o.mats = 0;
                    o.valid = 0;
                    return o;
                }

                int v0 = _HalfEdges[he0].v;
                int v1 = Dest(he0);
                int v2 = Dest(Next(he0));

                if (v0 >= _RealPointCount || v1 >= _RealPointCount || v2 >= _RealPointCount)
                {
                    o.pos = 0;
                    o.restNorm = 0;
                    o.bary = 0;
                    o.mats = 0;
                    o.valid = 0;
                    return o;
                }

                int owner0 = _OwnerByLocal[v0];
                int owner1 = _OwnerByLocal[v1];
                int owner2 = _OwnerByLocal[v2];
                if (owner0 < 0 || owner1 < 0 || owner2 < 0 || owner0 != owner1 || owner0 != owner2)
                {
                    o.pos = 0;
                    o.restNorm = 0;
                    o.bary = 0;
                    o.mats = 0;
                    o.valid = 0;
                    return o;
                }

                float a = saturate(_RenderAlpha);
                float2 p0 = lerp(_PositionsPrev[v0], _PositionsCurr[v0], a);
                float2 p1 = lerp(_PositionsPrev[v1], _PositionsCurr[v1], a);
                float2 p2 = lerp(_PositionsPrev[v2], _PositionsCurr[v2], a);

                float support = _LayerKernelH * _WendlandSupportScale;
                if (support <= 0.0)
                {
                    o.pos = 0;
                    o.restNorm = 0;
                    o.bary = 0;
                    o.mats = 0;
                    o.valid = 0;
                    return o;
                }

                float2 r0 = _RestNormPositions[v0];
                float2 r1 = _RestNormPositions[v1];
                float2 r2 = _RestNormPositions[v2];

                float2 p;
                float2 restP;
                float3 bary;
                if (!PatchTriangle((int)sourceCorner, support, p0, p1, p2, r0, r1, r2,
                _RestVolumes[v0], _RestVolumes[v1], _RestVolumes[v2],
                p, restP, bary))
                {
                    o.pos = 0;
                    o.restNorm = 0;
                    o.bary = 0;
                    o.mats = 0;
                    o.valid = 0;
                    return o;
                }

                float2 pW = GpuToWorld(p);
                o.pos = mul(UNITY_MATRIX_VP, float4(pW.x, pW.y, 0.0, 1.0));

                o.restNorm = restP;
                o.bary = bary;

                int m0 = ClampMaterial(_MaterialIds[v0]);
                int m1 = ClampMaterial(_MaterialIds[v1]);
                int m2 = ClampMaterial(_MaterialIds[v2]);
                o.mats = int3(m0, m1, m2);

                o.valid = 1;
                return o;
            }

            half4 Frag(v2f i) : SV_Target
            {
                if (i.valid == 0)
                discard;

                float2 uv = i.restNorm * _UvScale;

                half4 c0 = UNITY_SAMPLE_TEX2DARRAY(_AlbedoArray, float3(uv, i.mats.x));
                half4 c1 = UNITY_SAMPLE_TEX2DARRAY(_AlbedoArray, float3(uv, i.mats.y));
                half4 c2 = UNITY_SAMPLE_TEX2DARRAY(_AlbedoArray, float3(uv, i.mats.z));

                half4 col = c0 * i.bary.x + c1 * i.bary.y + c2 * i.bary.z;

                // Accum RT:
                // rgb = sum(color), a = sum(weight)
                return half4(col.rgb, 1.0h);
            }
            ENDHLSL
        }
    }
}
