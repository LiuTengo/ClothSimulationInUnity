using System;
using UnityEngine;

/// <summary>
/// Explicit Euler Method Cloth Simulation
/// Basic Method
/// </summary>
public class ExplicitCloth_GPU : MonoBehaviour
{
    private static int threadX = 8;
    private static int threadY = 8;
    
    //Effector
    [SerializeField] private int timeStep = 5;
    [SerializeField] private float mass = 1.0f;
    [SerializeField] private float springK = 25000;
    [SerializeField] private float damping = 0.99f;
    [SerializeField] private int edgeVertices = 12;
    [SerializeField] private GameObject ball;
    //[SerializeField] private int groupSize = 2;
    //Compute Shader
    [SerializeField] private ComputeShader computeShader;
    private ComputeBuffer positionBuffer;
    private ComputeBuffer velocityBuffer;
    
    //Buffer Data
    private Vector3[] Position;
    private Vector3[] Velocity;
    private Vector3 originalLength;
    
    //kernel Handle;
    private int kernelUpdatePosition;
    private int kernelUpdateVelocity;
    
    
    private Mesh mesh;
    
    //ball
    private Mesh ballMesh;
    private float ballRadius;

    private int groupSize;
    private void Start()
    {
        InitMeshAndArray();
        InitBuffer();
        InitShader();

        ballMesh = ball.GetComponent<MeshFilter>().mesh;
        ballRadius = ball.transform.localScale.x/2;
    }
    private void InitMeshAndArray()
    {
        mesh = GetComponent<MeshFilter>().mesh;
    
        if (!mesh)
        {
            throw new NotImplementedException();
        }
    
        Position = new Vector3[edgeVertices*edgeVertices];
        Velocity = new Vector3[edgeVertices*edgeVertices];
        Vector2[] uv = new Vector2[edgeVertices*edgeVertices];
        int[] trianglesEdges = new int[(edgeVertices-1)*(edgeVertices-1)*6];
        
        for (int y=0;y<edgeVertices;y++)
        {
            for (int x=0;x<edgeVertices;x++)
            {
                Position[y * edgeVertices + x] = 
                    new Vector3(5.0f-10.0f*x/(edgeVertices-1),0.0f,3.0f-6.0f*y/(edgeVertices-1));
                uv[y * edgeVertices + x] = new Vector2(x/(edgeVertices-1.0f),y/(edgeVertices-1.0f));
            }
        }

        float spacing = new Vector3(10.0f / (edgeVertices - 1),0,6.0f / (edgeVertices - 1)).magnitude;
        originalLength = new Vector3(spacing,spacing*Mathf.Sqrt(2),spacing*2);
    
        int t = 0;
        for (int j=0;j<edgeVertices-1;j++)
        {
            for (int i=0;i<edgeVertices-1;i++)
            {
                trianglesEdges[t * 6 + 0] = j*edgeVertices+i;
                trianglesEdges[t * 6 + 1] = j*edgeVertices+i+1;
                trianglesEdges[t * 6 + 2] = (j+1)*edgeVertices+i+1;
                trianglesEdges[t * 6 + 3] = j*edgeVertices+i;
                trianglesEdges[t * 6 + 4] = (j+1)*edgeVertices+i+1;
                trianglesEdges[t * 6 + 5] = (j+1)*edgeVertices+i;
                t++;
            }
        }
    
        mesh.vertices = Position;
        mesh.triangles = trianglesEdges;
        mesh.uv = uv;
        mesh.RecalculateNormals();
        
        for (int i=0;i<Velocity.Length;i++)
        {
            Velocity[i] = Vector3.zero;
        }

        groupSize = edgeVertices / threadX;
    }
    
    private void InitBuffer()
    { 
        positionBuffer = new ComputeBuffer(Position.Length,3*sizeof(float));
        velocityBuffer = new ComputeBuffer(Velocity.Length,3*sizeof(float));
        
        
        positionBuffer.SetData(Position);
        velocityBuffer.SetData(Velocity);
    }

    private void InitShader()
    {
        //kernelInital = computeShader.FindKernel("InitialMassSpring");
        kernelUpdateVelocity = computeShader.FindKernel("CalculateVelocity");
        kernelUpdatePosition = computeShader.FindKernel("CalculatePosition");
        
        //Pass Const Value
        //computeShader.SetInt("thread",threadNumber);
        computeShader.SetInt("edgeCount",edgeVertices);
        computeShader.SetFloat("damping",damping);
        computeShader.SetFloat("springK",springK);
        computeShader.SetFloat("mass",mass);
        computeShader.SetVector("originalLength",originalLength);
        
        //Pass Buffer
        computeShader.SetBuffer(kernelUpdateVelocity,"positionBuffer",positionBuffer);
        computeShader.SetBuffer(kernelUpdateVelocity,"velocityBuffer",velocityBuffer);

        computeShader.SetBuffer(kernelUpdatePosition,"positionBuffer",positionBuffer);
        computeShader.SetBuffer(kernelUpdatePosition,"velocityBuffer",velocityBuffer);
    }
    
    private void Update()
    {
        float dt = Time.deltaTime / timeStep;
        computeShader.SetFloat("dt",dt);
        for (int i=0;i<timeStep;i++)
        {
            Vector4 ballInfo = ball.transform.position;
            ballInfo.w = 1;
            Matrix4x4 mat= WorldToClothObject();
            ballInfo = mat * ballInfo;
            ballInfo.w = ballRadius;
            computeShader.SetVector("ballInfo",ballInfo);
            
            computeShader.Dispatch(kernelUpdatePosition,groupSize,groupSize,1);
            computeShader.Dispatch(kernelUpdateVelocity,groupSize,groupSize,1);
            
            positionBuffer.GetData(Position);
            
            mesh.vertices = Position;
            mesh.RecalculateNormals();
            
            //Graphics.DrawMeshNow(mesh, Matrix4x4.identity, 0);
        }
    }

    private void OnDestroy()
    {
        if (positionBuffer!=null)
        {
            positionBuffer.Release();
        }

        if (velocityBuffer != null)
        {
            velocityBuffer.Release();
        }
    }

    private Matrix4x4 WorldToClothObject()
    {
        Matrix4x4 mat = Matrix4x4.identity;
        for (int i=0;i<3;i++)
        {
            mat[3*4+i] = -transform.position[i];
        }
        return mat;
    }
}
