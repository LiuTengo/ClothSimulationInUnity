using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using TMPro;
using UnityEngine;

public class ClothExplictTest : MonoBehaviour
{
    private const int VERTICES_STRIDE =  3*sizeof(float);
    private Vector3[] clothPosition;
    private Vector3[] clothVelocity;
    
    [SerializeField] private ComputeShader shader;
    [SerializeField] private Vector3 springKEffector = new Vector3(25000,25000,25000);
    [SerializeField] private float mass = 1;
    [SerializeField] private float damping = 0.6f;
    [SerializeField] private int deltaTimeStep = 30;//单位时间步长
    
    
    private Mesh clothMesh;
    private int clothVerticesLength;
    private int[] clothTrianglesIndex;
    private int clothTrianglesLength;
    private Vector3[] meshVertices;
    private int threadX = 6;
    private int threadY = 6;
    private int edgeSize;
        
    private ComputeBuffer clothPositionBuffer;
    private ComputeBuffer clothVelocityBuffer;
    
    private int kernelInitSpring;
    private int kernelCalculatePosition;
    private int kernelCalculateVelocity;

    private int dtId;
    
    void Awake()
    {
        InitMesh();
        InitBuffer();
        InitShader();
        
        shader.Dispatch(kernelInitSpring,clothVerticesLength/threadX,clothVerticesLength/threadY,1);
    }

    private void InitMesh()
    {
        clothMesh = GetComponent<MeshFilter>().mesh;

        if (clothMesh == null)
        {
            Debug.LogError("Didn't find MeshFilter!");
            return;
        }
        
        meshVertices = clothMesh.vertices;
        clothVerticesLength = meshVertices.Length;
        edgeSize = (int)Mathf.Sqrt(clothVerticesLength);//布料一边顶点数
        clothTrianglesLength = (edgeSize - 1) * (edgeSize - 1)*6;
        
        Debug.Log(edgeSize);
        Debug.Log(clothTrianglesLength);
        Debug.Log(clothVerticesLength);
        
        clothPosition = new Vector3[clothVerticesLength];
        clothVelocity = new Vector3[clothVerticesLength];
        Vector2[] uv = new Vector2[clothVerticesLength];
        int[] triangles = new int[clothTrianglesLength];

        //重新排布顶点坐标以及UV
        for (int y=0;y<edgeSize;y++)
        {
            for (int x=0;x<edgeSize;x++)
            {
                clothPosition[y * edgeSize + x] = 
                    new Vector3(5.0f-10.0f*x/(edgeSize-1),0.0f,5.0f-10.0f*y/(edgeSize-1));
                uv[y * edgeSize + x] = new Vector2(x/(edgeSize-1.0f),y/(edgeSize-1.0f));
            }
        }

        int t = 0;
        for (int j=0;j<edgeSize-1;j++)
        {
            for (int i=0;i<edgeSize-1;i++)
            {
                triangles[t * 6 + 0] = j*edgeSize+i;
                triangles[t * 6 + 1] = j*edgeSize+i+1;
                triangles[t * 6 + 2] = (j+1)*edgeSize+i;
                triangles[t * 6 + 3] = j*edgeSize+i;
                triangles[t * 6 + 4] = (j+1)*edgeSize+i+1;
                triangles[t * 6 + 5] = (j+1)*edgeSize+i;
                t++;
            }
        }

        clothMesh.vertices = clothPosition;
        clothMesh.uv = uv;
        clothMesh.triangles = triangles;
        clothMesh.RecalculateNormals();

        for (int i=0;i<clothVerticesLength;i++)
        {
            clothVelocity[i] = Vector3.zero;
        }
    }
    
    
    
    private void InitBuffer()
    {
        clothPositionBuffer = new ComputeBuffer(clothVerticesLength,VERTICES_STRIDE);
        clothPositionBuffer.SetData(clothPosition);

        clothVelocityBuffer = new ComputeBuffer(clothVerticesLength,VERTICES_STRIDE);
        clothVelocityBuffer.SetData(clothVelocity);
    }
    
    private void InitShader()
    {
        dtId = Shader.PropertyToID("dt");
        
        kernelInitSpring = shader.FindKernel("InitialMassSpring");
        kernelCalculatePosition = shader.FindKernel("CalculatePosition");
        kernelCalculateVelocity = shader.FindKernel("CalculateVelocity");
        
        //shader.SetInt("threadX",threadX);
        //shader.SetInt("threadY",threadY);
        
        shader.SetFloat("mass",mass);
        shader.SetFloat("Damping",damping);
        int[] edge = new int[2];
        edge[0] = edgeSize;
        edge[1] = edgeSize;
        shader.SetInts("size",edge);
        
        shader.SetBuffer(kernelInitSpring,"positionBuffer",clothPositionBuffer);
        shader.SetBuffer(kernelCalculatePosition,"positionBuffer",clothPositionBuffer);
        shader.SetBuffer(kernelCalculateVelocity,"positionBuffer",clothPositionBuffer);
        
        shader.SetBuffer(kernelInitSpring,"velocityBuffer",clothVelocityBuffer);
        shader.SetBuffer(kernelCalculatePosition,"velocityBuffer",clothVelocityBuffer);
        shader.SetBuffer(kernelCalculateVelocity,"velocityBuffer",clothVelocityBuffer);
    }

    void Update()
    {
        float dt = Time.deltaTime / deltaTimeStep;
        for (int i=0;i<deltaTimeStep;i++)
        {
            shader.SetFloat(dtId,dt);
            shader.Dispatch(kernelCalculateVelocity,edgeSize/threadX,edgeSize/threadY,1);
            shader.Dispatch(kernelCalculatePosition,edgeSize/threadX,edgeSize/threadY,1);
        }
        
        clothPositionBuffer.GetData(clothPosition);
        for (int i=0;i<clothPosition.Length;i++)
        {
            Debug.Log(i+":"+clothPosition[i]);
        }
        //clothMesh.vertices = clothPosition;
        //clothMesh.RecalculateNormals();
    }

    private void OnDisable()
    {
        if (clothPositionBuffer != null)
        {
            clothPositionBuffer.Release();
            clothPositionBuffer = null;
        }
        if (clothVelocityBuffer != null)
        {
            clothVelocityBuffer.Release();
            clothVelocityBuffer = null;
        }
    }
}
