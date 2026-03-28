Shader "Unlit/InternalSurfaceAccumTinted"
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
            #pragma require 2darray
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
            StructuredBuffer<uint> _TriInternal;

            StructuredBuffer<int> _MaterialIds;
            int _MaterialCount;
            StructuredBuffer<int> _UvModes;
            StructuredBuffer<float4> _SpriteTints;
            StructuredBuffer<float2> _RestNormPositions;

            int _TriCount;
            int _RealPointCount;

            float2 _NormCenter;
            float _NormInvHalfExtent;
            float _UvScale;

            UNITY_DECLARE_TEX2DARRAY(_AlbedoArray);

            static int Next(int he) { return _HalfEdges[he].next; }

            float2 GpuToWorld(float2 p)
            {
                return p / _NormInvHalfExtent + _NormCenter;
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
                nointerpolation int materialId : TEXCOORD1;
                nointerpolation int uvMode : TEXCOORD2;
                nointerpolation float4 tint : TEXCOORD3;
                nointerpolation int valid : TEXCOORD4;
            };

            v2f Vert(uint vid : SV_VertexID)
            {
                v2f o;
                o.pos = 0;
                o.restNorm = 0;
                o.materialId = 0;
                o.uvMode = 0;
                o.tint = 1;
                o.valid = 0;

                uint tri = vid / 3u;
                uint corner = vid - tri * 3u;

                if (tri >= (uint)_TriCount)
                    return o;

                if (_TriInternal[tri] == 0u)
                    return o;

                int he0 = _TriToHE[tri];
                if (he0 < 0)
                    return o;

                int he1 = Next(he0);
                int he2 = Next(he1);

                int v0 = _HalfEdges[he0].v;
                int v1 = _HalfEdges[he1].v;
                int v2 = _HalfEdges[he2].v;

                if ((uint)v0 >= (uint)_RealPointCount ||
                    (uint)v1 >= (uint)_RealPointCount ||
                    (uint)v2 >= (uint)_RealPointCount)
                    return o;

                int v = (corner == 0u) ? v0 : ((corner == 1u) ? v1 : v2);

                float a = saturate(_RenderAlpha);
                float2 p = lerp(_PositionsPrev[v], _PositionsCurr[v], a);
                float2 pW = GpuToWorld(p);

                o.pos = mul(UNITY_MATRIX_VP, float4(pW.x, pW.y, 0.0, 1.0));
                o.restNorm = _RestNormPositions[v];
                o.materialId = ClampMaterial(_MaterialIds[v]);
                o.uvMode = (_UvModes[v] != 0) ? 1 : 0;
                o.tint = _SpriteTints[v];
                o.valid = 1;
                return o;
            }

            half4 Frag(v2f i) : SV_Target
            {
                if (i.valid == 0)
                    discard;

                float2 uv = (i.uvMode != 0) ? i.restNorm : (i.restNorm * _UvScale);
                half4 col = UNITY_SAMPLE_TEX2DARRAY(_AlbedoArray, float3(uv, i.materialId));
                col *= half4(i.tint);

                half alpha = saturate(col.a);
                return half4(col.rgb * alpha, alpha);
            }
            ENDHLSL
        }
    }
}