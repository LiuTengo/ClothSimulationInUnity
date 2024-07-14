Shader "Custom/SilkShader"
{
    Properties
    {
        _MainTex ("Albedo (RGB)", 2D) = "white" {}
        _Roughness("Roughness",Float) = 0.5
        _Anisotropy("Anisotropy",Float) = 0.5
        _SpecularColor("SpecularColoe",Color) = (68,70,72)
        _Tint("Tint",Float) = 0.5
    }
    SubShader
    {
        Pass
        {
            Tags { "RenderType"="Opaque" }
        LOD 200

        CGPROGRAM
        // Physically based Standard lighting model, and enable shadows on all light types
        #pragma vertex vert;
        #pragma fragment frag;

        #include "UnityCG.cginc"

        struct appdata
        {
            float4 vertex : POSITION;
            float2 uv : TEXCOORD0;
            float3 normal : NORMAL;
            float4 tangent : TANGENT;
        };

        struct v2f
        {
            float2 uv : TEXCOORD0;
            float4 vertex : SV_POSITION;
            float3 normal : TEXCOORD1;
            float3 tangent : TEXCOORD2;
            float3 bitangent : TEXCOORD3;
            float4 wPos: TEXCOORD4;
        };

        float3 F_Schlick(float3 f0, float VoH)
        {
            float f = pow(1.0 - VoH, 5.0);
            return f + f0 * (1.0 - f);
        }

        float D_GGX_Anisotropic(float at, float ab, float ToH, float BoH, float NoH)
        {
            float a2 = at * ab;
            float3 d = float3(ab * ToH, at * BoH, a2 * NoH);
            float d2 = dot(d, d);
            float b2 = a2 / d2;
            return a2 * b2 * b2 * (1 / UNITY_PI);
        }
        
        float V_SmithGGXCorrelated_Anisotropic(float at, float ab, float ToV, float BoV, float ToL, float BoL, float NoV, float NoL)
        {
            float lambdaV = NoL * length(float3(at * ToV, ab * BoV, NoV));
            float lambdaL = NoV * length(float3(at * ToL, ab * BoL, NoL));
            float v = 0.5 / (lambdaV + lambdaL);
            return v;
        }

        sampler2D _MainTex;
        float4 _MainTex_ST;
        float _Roughness;
        float _Anisotropy;
        float3 _SpecularColor;
        float _Tint;
        
        v2f vert (appdata v)
        {
            v2f o;
            // UNITY_INITIALIZE_OUTPUT(v2f, o);
            o.vertex = UnityObjectToClipPos(v.vertex);
            o.uv = TRANSFORM_TEX(v.uv, _MainTex);
            o.normal = UnityObjectToWorldNormal(v.normal);
            o.tangent = UnityObjectToWorldDir(v.tangent.xyz);
            o.bitangent = UnityObjectToWorldDir(cross(v.normal, v.tangent) * v.tangent.w); //v.tangent.w决定副切线方向
            o.wPos = mul(unity_ObjectToWorld, v.vertex);
            return o;
        }

        half4 frag(v2f i):SV_Target
        {
            float3 N = normalize(i.normal);
            float3 T = normalize(i.tangent);
            float3 B = normalize(i.bitangent);
            float3 L = normalize(UnityWorldSpaceLightDir(i.wPos));
            float3 V = normalize(UnityWorldSpaceViewDir(i.wPos));
            float3 H = normalize(L + V);
              
            float NoH = dot(N, H); float ToH = dot(T, H); float BoH = dot(B, H);
            float NoL = dot(N, L); float ToL = dot(T, L); float BoL = dot(B, L);
            float NoV = dot(N, V); float ToV = dot(T, V); float BoV = dot(B, V);
            float VoH = dot(V, H);
              
            float roughness = _Roughness * _Roughness; //粗糙度的映射有益于粗糙度直观化
              
            float at = max(roughness * (1 + _Anisotropy), 0.001);
            float ab = max(roughness * (1 - _Anisotropy), 0.001);
              
            float NDF = D_GGX_Anisotropic(at, ab, ToH, BoH, NoH);
            float Vis = V_SmithGGXCorrelated_Anisotropic(at, ab, ToV, BoV, ToL, BoL, NoV, NoL);
            float3 Fresnel = F_Schlick(_SpecularColor, VoH); //布料的_SpecularColor一般是(68,70,72)
              
            float3 specularLobe = NDF * Fresnel;
            float3 diffuseColor = tex2D(_MainTex, i.uv) * _Tint * (1.0 / UNITY_PI); //这里直接把Lambert合并进去了，可以考虑将(1.0 / UNITY_PI)作为常数来乘，节省计算
              
            fixed3 ambient = UNITY_LIGHTMODEL_AMBIENT.xyz;
              
            // sample the texture
            float4 col = float4(1, 1, 1, 1);
            col.xyz *= (diffuseColor + specularLobe + ambient) * unity_LightColor0.xyz * NoL;
            return col;
        }
        
        ENDCG
        }
    }
    FallBack "Diffuse"
}
