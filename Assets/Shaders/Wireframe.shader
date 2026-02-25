Shader "Unlit/Wireframe"
{
    Properties{
        _WireColor("Wire Color", Color) = (1, 1, 1, 1)
        _WireWidthPx("Wire Width (px)", Float) = 1.5
    }

    SubShader
    {
        Tags { "RenderType" = "Transparent" "Queue" = "Transparent+10" }

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
            float _LayerKernelH;
            float _WendlandSupportScale;

            float2 _NormCenter;
            float _NormInvHalfExtent;

            float4 _WireColor;
            float _WireWidthPx;
            int _CullOverstretched;

            static int Next(int he) { return _HalfEdges[he].next; }
            static int Dest(int he) { return _HalfEdges[Next(he)].v; }

            float2 GpuToWorld(float2 p) { return p / _NormInvHalfExtent + _NormCenter; }

            struct v2f
            {
                float4 pos : SV_POSITION;
                nointerpolation int valid : TEXCOORD0;
            };

            float4 OffsetClip(float4 clipPos, float2 offsetNdc)
            {
                clipPos.xy += offsetNdc * clipPos.w;
                return clipPos;
            }

            v2f Vert(uint vid : SV_VertexID)
            {
                v2f o;

                // We draw: triCount * 3 edges, each edge as a quad = 2 triangles = 6 verts.
                uint tri = vid / 18;                 // 18 verts per triangle (3 edges * 6)
                uint triLocal = vid - tri * 18;
                uint edgeInTri = triLocal / 6;       // 0..2
                uint cornerInQuad = triLocal - edgeInTri * 6; // 0..5

                int he0 = _TriToHE[tri];
                if (he0 < 0)
                {
                    o.pos = float4(0, 0, 0, 0);
                    o.valid = 0;
                    return o;
                }

                int he = (edgeInTri == 0) ? he0 : (edgeInTri == 1 ? Next(he0) : Next(Next(he0)));

                // Deduplicate: for internal edges, draw only once.
                int twin = _HalfEdges[he].twin;
                if (twin >= 0 && he > twin)
                {
                    o.pos = float4(0, 0, 0, 0);
                    o.valid = 0;
                    return o;
                }

                int v0 = _HalfEdges[he].v;
                int v1 = Dest(he);

                if (v0 >= _RealPointCount || v1 >= _RealPointCount)
                {
                    o.pos = float4(0, 0, 0, 0);
                    o.valid = 0;
                    return o;
                }

                float a = saturate(_RenderAlpha);
                float2 p0 = lerp(_PositionsPrev[v0], _PositionsCurr[v0], a);
                float2 p1 = lerp(_PositionsPrev[v1], _PositionsCurr[v1], a);

                if (_CullOverstretched != 0) {
                    float support = _LayerKernelH * _WendlandSupportScale;
                    if (support <= 0.0 || dot(p1 - p0, p1 - p0) > support * support) {
                        o.pos = float4(0, 0, 0, 0);
                        o.valid = 0;
                        return o;
                    }
                }

                float2 p0W = GpuToWorld(p0);
                float2 p1W = GpuToWorld(p1);

                float4 c0 = mul(UNITY_MATRIX_VP, float4(p0W.x, p0W.y, 0.0, 1.0));
                float4 c1 = mul(UNITY_MATRIX_VP, float4(p1W.x, p1W.y, 0.0, 1.0));

                float2 n0 = c0.xy / max(1e-6, c0.w);
                float2 n1 = c1.xy / max(1e-6, c1.w);

                float2 dir = n1 - n0;
                float len2 = dot(dir, dir);
                dir = (len2 > 1e-10) ? (dir * rsqrt(len2)) : float2(1, 0);
                float2 perp = float2(-dir.y, dir.x);

                float2 pxToNdc = 2.0 / _ScreenParams.xy;
                float2 off = perp * (_WireWidthPx * pxToNdc);

                // Quad vertex pattern (two triangles):
                // (p0+off, p0-off, p1+off) and (p1+off, p0-off, p1-off)
                float4 baseClip = (cornerInQuad == 0 || cornerInQuad == 1) ? c0 : c1;
                float sign = (cornerInQuad == 0 || cornerInQuad == 2 || cornerInQuad == 3) ? 1.0 : -1.0;

                o.pos = OffsetClip(baseClip, off * sign);
                o.valid = 1;
                return o;
            }

            fixed4 Frag(v2f i) : SV_Target
            {
                if (i.valid == 0)
                return fixed4(0, 0, 0, 0);

                return fixed4(_WireColor.rgb, 1.0);
            }
            ENDHLSL
        }
    }
}
