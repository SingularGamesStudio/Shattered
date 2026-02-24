Shader "Hidden/TriangulationComposite"
{
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Overlay" }
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex Vert
            #pragma fragment Frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _MainTex_TexelSize;

            sampler2D _AccumTex;
            sampler2D _SdfTex;
            float _RoundPixels;

            struct v2f { float4 pos:SV_POSITION; float2 uv:TEXCOORD0; };

            v2f Vert(uint id : SV_VertexID)
            {
                v2f o;
                o.uv = float2((id << 1) & 2, id & 2);
                o.pos = float4(o.uv * 2.0 - 1.0, 0, 1);
                return o;
            }

            float2 FixUv(float2 uv)
            {
                // Unity: only flip when needed (platform / AA / image effects). [web:114][web:131]
                #if UNITY_UV_STARTS_AT_TOP
                    if (_MainTex_TexelSize.y < 0)
                    uv.y = 1.0 - uv.y;
                #endif
                return uv;
            }

            fixed4 Frag(v2f i) : SV_Target
            {
                float2 uv = FixUv(i.uv);

                fixed4 bg = tex2D(_MainTex, uv);

                float sdf = tex2D(_SdfTex, uv).r; // signed distance in pixels, inside negative

                if (sdf > -_RoundPixels)
                return bg;

                float4 acc = tex2D(_AccumTex, uv);
                float w = max(acc.a, 1e-4);
                fixed3 col = acc.rgb / w;

                return fixed4(col, 1.0);
            }
            ENDHLSL
        }

        Pass
        {
            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex Vert
            #pragma fragment FragDbgAccum
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _MainTex_TexelSize;
            sampler2D _AccumTex;

            struct v2f { float4 pos:SV_POSITION; float2 uv:TEXCOORD0; };

            v2f Vert(uint id : SV_VertexID)
            {
                v2f o;
                o.uv = float2((id << 1) & 2, id & 2);
                o.pos = float4(o.uv * 2.0 - 1.0, 0, 1);
                return o;
            }

            float2 FixUv(float2 uv)
            {
                #if UNITY_UV_STARTS_AT_TOP
                    if (_MainTex_TexelSize.y < 0)
                    uv.y = 1.0 - uv.y;
                #endif
                return uv;
            }

            fixed4 FragDbgAccum(v2f i) : SV_Target
            {
                float2 uv = FixUv(i.uv);
                float a = tex2D(_AccumTex, uv).a;
                return fixed4(a, a, a, 1.0);
            }
            ENDHLSL
        }

        Pass
        {
            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex Vert
            #pragma fragment FragDbgSdf
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _MainTex_TexelSize;
            sampler2D _SdfTex;
            sampler2D _AccumTex;

            struct v2f { float4 pos:SV_POSITION; float2 uv:TEXCOORD0; };

            v2f Vert(uint id : SV_VertexID)
            {
                v2f o;
                o.uv = float2((id << 1) & 2, id & 2);
                o.pos = float4(o.uv * 2.0 - 1.0, 0, 1);
                return o;
            }

            float2 FixUv(float2 uv)
            {
                #if UNITY_UV_STARTS_AT_TOP
                    if (_MainTex_TexelSize.y < 0)
                    uv.y = 1.0 - uv.y;
                #endif
                return uv;
            }

            fixed4 FragDbgSdf(v2f i) : SV_Target
            {
                float2 uv = FixUv(i.uv);
                float sd = tex2D(_SdfTex, uv).r;
                float inside = tex2D(_AccumTex, uv).a > 0.0 ? 1.0 : 0.0;

                float v = saturate(abs(sd) / 16.0);
                return fixed4(v, inside, 1.0 - inside, 1.0);
            }
            ENDHLSL
        }
    }
}
