Shader "ClothSimulation/UnlitClothShader"
{
    Properties
    {
        _Color ("Color", Color) = (1,1,1,1)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100
        Cull Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work

            #include "UnityCG.cginc"
            #include "UnityLightingCommon.cginc" // for _LightColor0


            StructuredBuffer<float4> _positions;
            StructuredBuffer<float4> _normals;

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float3 worldPos: TEXCOORD0;
                float3 normal: TEXCOORD1;
            };

            fixed4 _Color;

            v2f vert (uint id : SV_VertexID)
            {
                v2f o;
                float4 vertex = _positions[id];
                o.normal = _normals[id].xyz;
                o.vertex = UnityObjectToClipPos(vertex); 
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float3 worldNormal = i.normal;
                half nl = abs(dot(worldNormal, _WorldSpaceLightPos0.xyz));
                return nl * _Color;
            }
            ENDCG
        }
    }
}
