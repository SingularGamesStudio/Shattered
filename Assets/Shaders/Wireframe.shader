Shader "Unlit/Wireframe"
{
    Properties{
        _WireColor("Wire Color", Color) = (1, 1, 1, 1)
            _WireWidthPx("Wire Width (px)", Float) = 1.5} SubShader
    {
        Tags{"RenderType" = "Transparent"
                            "Queue" = "Transparent+10"}

        Pass
        {
            ZWrite Off
                ZTest LEqual
                    Cull Off
                        Blend Off

                            HLSLPROGRAM
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

            int _RealPointCount;

            float2 _NormCenter;
            float _NormInvHalfExtent;

            float4 _WireColor;
            float _WireWidthPx;

            static int Next(int he) { return _HalfEdges[he].next; }
            static int Dest(int he) { return _HalfEdges[Next(he)].v; }

            float2 GpuToWorld(float2 p) { return p / _NormInvHalfExtent + _NormCenter; }

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 bary : TEXCOORD0;
                nointerpolation int valid : TEXCOORD1;
            };

            v2f Vert(uint vid : SV_VertexID)
            {
                v2f o;

                uint tri = vid / 3;
                uint corner = vid - tri * 3;

                int he0 = _TriToHE[tri];
                if (he0 < 0)
                {
                    o.pos = float4(0, 0, 0, 0);
                    o.bary = 0;
                    o.valid = 0;
                    return o;
                }

                int v0 = _HalfEdges[he0].v;
                int v1 = Dest(he0);
                int v2 = Dest(Next(he0));

                if (v0 >= _RealPointCount || v1 >= _RealPointCount || v2 >= _RealPointCount)
                {
                    o.pos = float4(0, 0, 0, 0);
                    o.bary = 0;
                    o.valid = 0;
                    return o;
                }

                int v = (corner == 0) ? v0 : (corner == 1 ? v1 : v2);

                float a = saturate(_RenderAlpha);
                float2 p = lerp(_PositionsPrev[v], _PositionsCurr[v], a);
                float2 pW = GpuToWorld(p);
                o.pos = mul(UNITY_MATRIX_VP, float4(pW.x, pW.y, 0.0, 1.0));

                o.bary = (corner == 0) ? float3(1, 0, 0) : (corner == 1 ? float3(0, 1, 0) : float3(0, 0, 1));
                o.valid = 1;
                return o;
            }

            float WireAlpha(float3 bary)
            {
                float e = min(bary.x, min(bary.y, bary.z));
                float w = fwidth(e) * _WireWidthPx;
                return 1.0 - smoothstep(0.0, w, e);
            }

            fixed4 Frag(v2f i) : SV_Target
            {
                if (i.valid == 0)
                    discard;

                float a = WireAlpha(i.bary);
                clip(a - 0.001);

                return fixed4(_WireColor.rgb, 1.0);
            }
            ENDHLSL
        }
    }
}
