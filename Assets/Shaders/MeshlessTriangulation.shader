Shader "Unlit/MeshlessTriangulation" {
    Properties {
        _UvScale ("UV Scale", Float) = 0.25
        _WireColor ("Wire Color", Color) = (0,0,0,1)
        _WireWidth ("Wire Width", Float) = 1.5
    }
    SubShader {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }

        // Pass 0: Fill (no wire).
        Pass {
            ZWrite On
            ZTest LEqual
            Cull Off

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment FragFill
            #include "UnityCG.cginc"

            struct HalfEdge { int v; int next; int twin; int t; };

            StructuredBuffer<float2> _Positions;
            StructuredBuffer<HalfEdge> _HalfEdges;
            StructuredBuffer<int> _TriToHE;

            StructuredBuffer<int> _MaterialIds;
            int _MaterialCount;

            UNITY_DECLARE_TEX2DARRAY(_AlbedoArray);

            float2 _NormCenter;
            float _NormInvHalfExtent;

            float _UvScale;

            static int Next(int he) { return _HalfEdges[he].next; }
            static int Dest(int he) { return _HalfEdges[Next(he)].v; }

            float2 GpuToWorld(float2 p) { return p / _NormInvHalfExtent + _NormCenter; }

            int ClampMaterial(int m) {
                if (_MaterialCount <= 0) return 0;
                return clamp(m, 0, _MaterialCount - 1);
            }

            struct v2f {
                float4 pos : SV_POSITION;
                float2 world : TEXCOORD0;
                float3 bary : TEXCOORD1;
                nointerpolation int3 mats : TEXCOORD2;
                nointerpolation int valid : TEXCOORD3;
            };

            v2f Vert(uint vid : SV_VertexID) {
                v2f o;

                uint tri = vid / 3;
                uint corner = vid - tri * 3;

                int he0 = _TriToHE[tri];
                if (he0 < 0) {
                    o.pos = float4(0, 0, 0, 0);
                    o.world = 0;
                    o.bary = 0;
                    o.mats = int3(0,0,0);
                    o.valid = 0;
                    return o;
                }

                int v0 = _HalfEdges[he0].v;
                int v1 = Dest(he0);
                int v2 = Dest(Next(he0));

                int v = (corner == 0) ? v0 : (corner == 1 ? v1 : v2);

                float2 pW = GpuToWorld(_Positions[v]);

                o.pos = mul(UNITY_MATRIX_VP, float4(pW.x, pW.y, 0.0, 1.0));
                o.world = pW;

                o.bary = (corner == 0) ? float3(1,0,0) : (corner == 1 ? float3(0,1,0) : float3(0,0,1));

                int m0 = ClampMaterial(_MaterialIds[v0]);
                int m1 = ClampMaterial(_MaterialIds[v1]);
                int m2 = ClampMaterial(_MaterialIds[v2]);
                o.mats = int3(m0, m1, m2);

                o.valid = 1;
                return o;
            }

            fixed4 FragFill(v2f i) : SV_Target {
                if (i.valid == 0) discard;

                float2 uv = i.world * _UvScale;

                fixed4 c0 = UNITY_SAMPLE_TEX2DARRAY(_AlbedoArray, float3(uv, i.mats.x));
                fixed4 c1 = UNITY_SAMPLE_TEX2DARRAY(_AlbedoArray, float3(uv, i.mats.y));
                fixed4 c2 = UNITY_SAMPLE_TEX2DARRAY(_AlbedoArray, float3(uv, i.mats.z));

                return c0 * i.bary.x + c1 * i.bary.y + c2 * i.bary.z;
            }
            ENDHLSL
        }

        // Pass 1: Wire overlay.
        Pass {
            ZWrite Off
            ZTest Always
            Cull Off
            Blend SrcAlpha OneMinusSrcAlpha

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment FragWire
            #include "UnityCG.cginc"

            struct HalfEdge { int v; int next; int twin; int t; };

            StructuredBuffer<float2> _Positions;
            StructuredBuffer<HalfEdge> _HalfEdges;
            StructuredBuffer<int> _TriToHE;

            StructuredBuffer<int> _MaterialIds;
            int _MaterialCount;

            float2 _NormCenter;
            float _NormInvHalfExtent;

            float4 _WireColor;
            float _WireWidth;

            static int Next(int he) { return _HalfEdges[he].next; }
            static int Dest(int he) { return _HalfEdges[Next(he)].v; }

            float2 GpuToWorld(float2 p) { return p / _NormInvHalfExtent + _NormCenter; }

            struct v2f {
                float4 pos : SV_POSITION;
                float2 world : TEXCOORD0;
                float3 bary : TEXCOORD1;
                nointerpolation int3 mats : TEXCOORD2;
                nointerpolation int valid : TEXCOORD3;
            };

            v2f Vert(uint vid : SV_VertexID) {
                v2f o;

                uint tri = vid / 3;
                uint corner = vid - tri * 3;

                int he0 = _TriToHE[tri];
                if (he0 < 0) {
                    o.pos = float4(0, 0, 0, 0);
                    o.world = 0;
                    o.bary = 0;
                    o.mats = int3(0,0,0);
                    o.valid = 0;
                    return o;
                }

                int v0 = _HalfEdges[he0].v;
                int v1 = Dest(he0);
                int v2 = Dest(Next(he0));

                int v = (corner == 0) ? v0 : (corner == 1 ? v1 : v2);

                float2 pW = GpuToWorld(_Positions[v]);

                o.pos = mul(UNITY_MATRIX_VP, float4(pW.x, pW.y, 0.0, 1.0));
                o.world = pW;

                o.bary = (corner == 0) ? float3(1,0,0) : (corner == 1 ? float3(0,1,0) : float3(0,0,1));

                o.mats = int3(0,0,0);
                o.valid = 1;
                return o;
            }

            float EdgeMask(float3 bary) {
                float3 d = fwidth(bary);
                float3 a3 = smoothstep(0.0, d * _WireWidth, bary);
                return 1.0 - min(min(a3.x, a3.y), a3.z);
            }

            fixed4 FragWire(v2f i) : SV_Target {
                if (i.valid == 0) discard;

                float edge = EdgeMask(i.bary);
                clip(edge - 0.001);
                return _WireColor;
            }
            ENDHLSL
        }
    }
}
