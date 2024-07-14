using System;
using System.Collections.Generic;
using UnityEngine;

namespace Cloth.Scripts.gpu.GaussSeidel
{
    [Serializable]
    public struct Edge
    {
        public int i, j;
        public float restLength;

        public Edge(int _i,int _j)
        {
            i = _i;
            j = _j;
            restLength = 0.0f;
        }

        public void set_restLength(Vector3[] position)
        {
            restLength = (position[i]-position[j]).magnitude;
        }
    }
    public struct EdgeTriangle
    {
        public int i,j;
        public int triangle;
    }

    
    public class JacobiClothPBD_GPU : MonoBehaviour
    { 
        private static int Thread = 16; 
    
        [SerializeField]
        private ComputeShader computeShader;
        //可设置的参数
        [SerializeField]
        private float deltaTime;
        [SerializeField]
        private float damping;
        [SerializeField]
        private float thickness;
        [SerializeField]
        private Vector3 g;
        [SerializeField]
        private Vector3 forceField;
        [SerializeField]
        private int iterationTime;
        [SerializeField]
        private int[] pinPoint;
        [SerializeField]
        private SphereColliderProxy[] collisionBalls;
        //布料顶点
        private Vector3[] vertices;
        private Vector3[] velocity;
        private Edge[] m_Edges;
        private Vector3[] sumPosition;
        private int[] sumNumber;
        private int[] pin;

        public BallInfo[] ballInfos;
        
        private Vector3Int[] deltaInt;
        private int[] m_TriangleIndices;
        
        //ComputeBuffer
        private ComputeBuffer velocityBuffer;
        private ComputeBuffer edgeBuffer;
        private ComputeBuffer positionBuffer;
        private ComputeBuffer sumPositionBuffer;
        private ComputeBuffer sumNumberBuffer;
        private ComputeBuffer deltaIntBuffer;
        private ComputeBuffer pinBuffer;
        
        private ComputeBuffer ballInfoBuffer;
        
        //kernelHandle
        private int kernelPredictPosition;
        private int kernelInitialStrainLimiting;
        private int kernelClothConstrainLimiting;
        private int kernelUpdatePositionAndVelocity;
        private int kernelBallCollision;
        
        //Private Parameters
        private int totalEdge;
        private int totalVertecies;
        
        private int collisionBallCount;
        
        private int verticesGroup;
        private int edgeGroup;
        private Mesh m_Mesh;
        private Matrix4x4 transformMatrix;
        private static readonly int DT = Shader.PropertyToID("dt");
        
        private void Awake()
        {
            m_Mesh = GetComponent<MeshFilter>().mesh;
            transformMatrix = WorldToClothObject();
            
            InitialArray();
            InitialEdges();
            InitParameter();
            InitBuffer();
            InitShader();
        }
        
        #region QuickSort

        private void Swap(ref int a,ref int b)
        {
            (a, b) = (b, a);
        }

        private void Swap(ref EdgeTriangle a, ref EdgeTriangle b)
        {
            int temp;

            temp = a.i;
            a.i = b.i;
            b.i = temp;

            temp = a.j;
            a.j = b.j;
            b.j = temp;

            temp = a.triangle;
            a.triangle = b.triangle;
            b.triangle = temp;
        }

        private void QuickSort(EdgeTriangle[] a,int l,int r)
        {
            int j;
            if (l<r)
            {
                j = QuickSortPartition(a,l,r);
                QuickSort(a,l,j-1);
                QuickSort(a,j+1,r);
            }
        }

        private bool IsSameEdge(ref EdgeTriangle a,ref EdgeTriangle b)
        {
            return a.i==b.i && a.j==b.j;
        }
    
        private bool EdgeTriangleLarger(ref EdgeTriangle a, ref EdgeTriangle b)
        {
            return (a.i > b.i) || (a.i == b.i && a.j > b.j);
        }
    
        private int QuickSortPartition(EdgeTriangle[] a, int left, int right)
        {
            EdgeTriangle pivot;
            int i, j;
            pivot = a[left];
            i = left;
            j = right + 1;
            while (true)
            {
                do ++i; while (i <= right && EdgeTriangleLarger(ref pivot, ref a[i]));
                do --j; while (EdgeTriangleLarger(ref a[j], ref pivot));
                if (i >= j)
                    break;
                Swap(ref a[i], ref a[j]);
            }
            Swap(ref a[left], ref a[j]);
            return j;
        }

        #endregion
        
        #region Initialization
        
    private void InitialArray()
    {
        totalVertecies = m_Mesh.vertexCount;
        collisionBallCount = collisionBalls.Length;
        
        var pos = m_Mesh.vertices;
        m_TriangleIndices = new int[m_Mesh.triangles.Length];
        
        vertices = new Vector3[totalVertecies];
        velocity = new Vector3[totalVertecies];
        sumPosition = new Vector3[totalVertecies];
        sumNumber = new int[totalVertecies];
        pin = new int[totalVertecies];

        deltaInt = new Vector3Int[totalVertecies];

        ballInfos = new BallInfo[collisionBalls.Length];

        for (int t=0;t<m_Mesh.triangles.Length;t++)
        {
            m_TriangleIndices[t] = m_Mesh.triangles[t];
        }
        
        for (int i=0;i<totalVertecies;i++)
        {
            vertices[i] =  pos[i];
            velocity[i] = Vector3.zero;
            sumPosition[i] = Vector3.zero;
            sumNumber[i] = 0;
            pin[i] = 0;
            deltaInt[i] = Vector3Int.zero;
        }
        
        foreach (var p in pinPoint)
        {
            pin[p] = 1;
        }

        for (int i=0;i<collisionBallCount;i++)
        {
            ballInfos[i] = collisionBalls[i].GetBallInfo();
        }
    }

    private void InitialEdges()
    {
        EdgeTriangle[] edgeTriangles = new EdgeTriangle[m_TriangleIndices.Length];
        int triangleCount = 0;
        for (int i=0;i<m_TriangleIndices.Length;i+=3)
        {
            edgeTriangles[i].i = m_TriangleIndices[i];
            edgeTriangles[i].j = m_TriangleIndices[i+1];
            edgeTriangles[i+1].i = m_TriangleIndices[i+1];
            edgeTriangles[i+1].j = m_TriangleIndices[i+2];
            edgeTriangles[i+2].i = m_TriangleIndices[i+2];
            edgeTriangles[i+2].j = m_TriangleIndices[i];
            edgeTriangles[i].triangle = triangleCount;
            edgeTriangles[i+1].triangle = triangleCount;
            edgeTriangles[i+2].triangle = triangleCount;
            triangleCount++;
        }

        for (int i=0;i<m_TriangleIndices.Length;i++)
        {
            if (edgeTriangles[i].i > edgeTriangles[i].j)
            {
                (edgeTriangles[i].i, edgeTriangles[i].j) = (edgeTriangles[i].j, edgeTriangles[i].i);
            }
        }
        
        QuickSort(edgeTriangles,0,edgeTriangles.Length-1);

        int edgeCount = 0;
        for (int i=0;i<edgeTriangles.Length;i++)
        {
            if (i==0 || !IsSameEdge(ref edgeTriangles[i],ref edgeTriangles[i-1]))
            {
                edgeCount++;
            }
        }

        m_Edges = new Edge[edgeCount];
        for (int i=0,e=0;i<edgeTriangles.Length;i++)
        {
            Edge edge = new Edge();
            if (i==0 || !IsSameEdge(ref edgeTriangles[i],ref edgeTriangles[i-1]))
            {
                edge.i = edgeTriangles[i].i;
                edge.j = edgeTriangles[i].j;
                edge.set_restLength(vertices);
                    
                m_Edges[e] = edge;
                e++;
            }
        }

        totalEdge = m_Edges.Length;
    }

    private void InitParameter()
    {
        verticesGroup = (int)Math.Ceiling(((float)totalVertecies/Thread));
        edgeGroup = (int)Math.Ceiling(((float)totalEdge/Thread));
    }
        
        private void InitBuffer()
        {
            positionBuffer = new ComputeBuffer(vertices.Length,3*sizeof(float));
            positionBuffer.SetData(vertices);
            
            velocityBuffer = new ComputeBuffer(vertices.Length,3*sizeof(float));
            velocityBuffer.SetData(velocity);
        
            sumPositionBuffer = new ComputeBuffer(vertices.Length,3*sizeof(float));
            sumPositionBuffer.SetData(sumPosition);
            
            sumNumberBuffer = new ComputeBuffer(vertices.Length,1*sizeof(int));
            sumNumberBuffer.SetData(sumNumber);
            
            edgeBuffer = new ComputeBuffer(m_Edges.Length,2*sizeof(int)+1*sizeof(float));
            edgeBuffer.SetData(m_Edges);

            pinBuffer = new ComputeBuffer(vertices.Length,1*sizeof(int));
            pinBuffer.SetData(pin);
            
            deltaIntBuffer = new ComputeBuffer(deltaInt.Length,3*sizeof(int));
            deltaIntBuffer.SetData(deltaInt);

            ballInfoBuffer = new ComputeBuffer(ballInfos.Length,4*sizeof(float));
        }
        
        private void InitShader()
        {
            kernelPredictPosition = computeShader.FindKernel("PredictPosition");
            kernelInitialStrainLimiting= computeShader.FindKernel("InitialStrainLimiting");
            kernelClothConstrainLimiting= computeShader.FindKernel("ClothConstrainLimiting");
            kernelUpdatePositionAndVelocity= computeShader.FindKernel("UpdatePositionAndVelocity");
            kernelBallCollision = computeShader.FindKernel("CollisionHandling");
            
            computeShader.SetFloat("damping",damping);
            computeShader.SetFloat("thickness",thickness);
            computeShader.SetInt("totalVertex",totalVertecies);
            computeShader.SetVector("forceField",forceField);
            computeShader.SetVector("g",g);
            computeShader.SetInt("sphereLength",collisionBallCount);
        
            computeShader.SetBuffer(kernelPredictPosition,"position",positionBuffer);
            computeShader.SetBuffer(kernelPredictPosition,"velocity",velocityBuffer);
            computeShader.SetBuffer(kernelPredictPosition,"pin",pinBuffer);
        
            computeShader.SetBuffer(kernelInitialStrainLimiting,"sum_p",sumPositionBuffer);
            computeShader.SetBuffer(kernelInitialStrainLimiting,"sum_n",sumNumberBuffer);
        
            computeShader.SetBuffer(kernelClothConstrainLimiting,"edges",edgeBuffer);
            computeShader.SetBuffer(kernelClothConstrainLimiting,"position",positionBuffer);
            computeShader.SetBuffer(kernelClothConstrainLimiting,"sum_p",sumPositionBuffer);
            computeShader.SetBuffer(kernelClothConstrainLimiting,"sum_n",sumNumberBuffer);
            computeShader.SetBuffer(kernelClothConstrainLimiting,"deltaIntBuffer",deltaIntBuffer);
            
            computeShader.SetBuffer(kernelUpdatePositionAndVelocity,"position",positionBuffer);
            computeShader.SetBuffer(kernelUpdatePositionAndVelocity,"velocity",velocityBuffer);
            computeShader.SetBuffer(kernelUpdatePositionAndVelocity,"sum_p",sumPositionBuffer);
            computeShader.SetBuffer(kernelUpdatePositionAndVelocity,"sum_n",sumNumberBuffer);
            computeShader.SetBuffer(kernelUpdatePositionAndVelocity,"deltaIntBuffer",deltaIntBuffer);
            computeShader.SetBuffer(kernelUpdatePositionAndVelocity,"pin",pinBuffer);
            
            computeShader.SetBuffer(kernelBallCollision,"pin",pinBuffer);
            computeShader.SetBuffer(kernelBallCollision,"sphere",ballInfoBuffer);
            computeShader.SetBuffer(kernelBallCollision,"position",positionBuffer);
            computeShader.SetBuffer(kernelBallCollision,"velocity",velocityBuffer);
        }
        #endregion
        
        private void Update()
        {
           computeShader.SetFloat(DT,deltaTime);
           computeShader.SetFloat("damping",damping);
           computeShader.SetVector("forceField",forceField);
           computeShader.SetVector("g",g);

           for (int i=0;i<collisionBallCount;i++)
           {
               collisionBalls[i].UpdateBallInfo();
               ballInfos[i] = collisionBalls[i].GetBallInfo();
               ballInfos[i].ToLocalSpace(ref transformMatrix);
           }
           ballInfoBuffer.SetData(ballInfos);
           computeShader.SetBuffer(kernelBallCollision,"sphere",ballInfoBuffer);
           
           computeShader.Dispatch(kernelPredictPosition,verticesGroup,1,1);
           for (int i=0;i<iterationTime;i++)
           {
               computeShader.Dispatch(kernelInitialStrainLimiting,verticesGroup,1,1);
               computeShader.Dispatch(kernelClothConstrainLimiting, edgeGroup, 1,1);
               computeShader.Dispatch(kernelUpdatePositionAndVelocity,verticesGroup,1,1);
           }
           computeShader.Dispatch(kernelBallCollision,verticesGroup,1,1);
           
           positionBuffer.GetData(vertices);
           m_Mesh.vertices = vertices;
           m_Mesh.RecalculateNormals();
        }
        
        private void OnDestroy()
        {
            velocityBuffer?.Release();
            positionBuffer?.Release();
            sumPositionBuffer?.Release();
            sumNumberBuffer?.Release();
            edgeBuffer?.Release();
            deltaIntBuffer?.Release();
            pinBuffer?.Release();
            ballInfoBuffer?.Release();
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
}
