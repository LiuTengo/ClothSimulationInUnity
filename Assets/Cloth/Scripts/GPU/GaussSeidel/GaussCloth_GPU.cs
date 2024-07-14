using System;
using Cloth.Scripts.gpu.GaussSeidel;
using UnityEngine;

namespace Cloth.Scripts.gpu.Jacobi
{
    public struct Edge
    {
        public int I, J;
        public float restLength;

        public int nT1;
        public int nT2;

        public int nv1;
        public int nv2;
        public float restAngle;

        public void SelfInit(Vector3[] vertices, int[] triangleVertexIndices)
        {
            SetRestLength(vertices);
            SetThirdVertexs(triangleVertexIndices);
            SetRestAngle(vertices);
        }

        public bool HaveTwoNeighborTriangle()
        {
            return nT2 >= 0;
        }

        private void SetRestLength(Vector3[] vertices)
        {
            restLength = (vertices[I] - vertices[J]).magnitude;
        }

        private int GetThirdVertexIndex(int triangleIndex, int v1, int v2, int[] triangleVertexIndices)
        {
            for (var i = 0; i < 3; i++)
            {
                var v = triangleVertexIndices[triangleIndex * 3 + i];
                if (v != v1 && v != v2)
                    return v;
            }

            return -1;
        }

        private void SetThirdVertexs(int[] triangleVertexIndices)
        {
            if (HaveTwoNeighborTriangle())
            {
                nv1 = GetThirdVertexIndex(nT1, I, J, triangleVertexIndices);
                nv2 = GetThirdVertexIndex(nT2, I, J, triangleVertexIndices);
                if (nv1 < 0 || nv2 < 0)
                    Debug.LogError("mesh data error");
            }
            else
            {
                nv1 = GetThirdVertexIndex(nT1, I, J, triangleVertexIndices);
                if (nv1 < 0)
                    Debug.LogError("mesh data error");
                nv2 = -1;
            }
        }

        private void SetRestAngle(Vector3[] vertices)
        {
            if (!HaveTwoNeighborTriangle())
            {
                restAngle = 0;
                return;
            }

            var p1 = vertices[I];
            var p2 = vertices[J] - p1;
            var p3 = vertices[nv1] - p1;
            var p4 = vertices[nv2] - p1;

            var n1 = Vector3.Normalize(Vector3.Cross(p2, p3));
            var n2 = Vector3.Normalize(Vector3.Cross(p2, p4));
            restAngle = Mathf.Acos(Vector3.Dot(n1, n2));
        }
    }

    public struct CollideInfo
    {
        public Vector3 position;
        public Vector3 normal;
    }

    public struct EdgeTriangle
    {
        public int I, J;
        public int triangle;
    }

    [RequireComponent(typeof(MeshFilter))]
    [RequireComponent(typeof(MeshRenderer))]
    public class GaussCloth_GPU : MonoBehaviour
    {
        private static readonly int Thread = 16;
        private static readonly int Dt = Shader.PropertyToID("dt");
        private static readonly int Step = Shader.PropertyToID("s");

        [SerializeField] private ComputeShader shader;
        [SerializeField] private Vector3 G = new(0.0f, -9.8f, 0.0f);
        [SerializeField] private Vector3 forceField = new(0.0f, 0.0f, 0.0f);
        [SerializeField] [Range(0, 1)] private float deltaTime;
        [SerializeField] [Range(0, 1)] private float damper;
        [SerializeField] [Range(0, 1)] private float stiffness;
        [SerializeField] private int iterateCount;
        [SerializeField] private float density;
        [SerializeField] private float thickness;
        [SerializeField] private int[] pinPoint;
        [SerializeField] private bool enableCollision;
        [SerializeField] private SphereColliderProxy[] collisionBall;

        private Mesh m_Mesh;

        private int m_VerticesCount;

        private int[] m_TriangleIndices;
        public int[] pin;
        private Edge[] m_Edges;
        private Vector3[] position;
        private Vector3[] predictPosition;
        private Vector3[] velocity;
        private Vector3Int[] deltaInt;
        private BallInfo[] m_SphereData;
        private CollideInfo[] m_CollideInfos;

        private float[] vertexMass;

        private ComputeBuffer edges;
        private ComputeBuffer positionBuffer;
        private ComputeBuffer predictPositionBuffer;
        private ComputeBuffer velocityBuffer;
        private ComputeBuffer massBuffer;
        private ComputeBuffer pinPointBuffer;
        private ComputeBuffer deltaIntBuffer;
        private ComputeBuffer sphereBuffer;
        private ComputeBuffer collideInfoBuffer;

        private int kernelInitSimulation;
        private int kernelDistanceLimiting;
        private int kernelBendingLimitiing;
        private int kernelUpdatePosition;
        private int kernelAddCollisionConstrain;
        private int kernelCollisionConstain;

        private int totalEdge;
        private int totalVertecies;
        private int totalSphere;

        private int verticesGroup;
        private int edgeGroup;
        private Matrix4x4 mat;

        private void Awake()
        {
            InitialComponent();
            InitialArray();
            InitMass();
            InitialEdges();
            InitParameter();

            InitBuffer();
            InitShader();
        }

        #region QuickSort

        private void Swap(ref int a, ref int b)
        {
            (a, b) = (b, a);
        }

        private void Swap(ref EdgeTriangle a, ref EdgeTriangle b)
        {
            int temp;

            temp = a.I;
            a.I = b.I;
            b.I = temp;

            temp = a.J;
            a.J = b.J;
            b.J = temp;

            temp = a.triangle;
            a.triangle = b.triangle;
            b.triangle = temp;
        }

        private void QuickSort(EdgeTriangle[] a, int l, int r)
        {
            int j;
            if (l < r)
            {
                j = QuickSortPartition(a, l, r);
                QuickSort(a, l, j - 1);
                QuickSort(a, j + 1, r);
            }
        }

        private bool IsSameEdge(ref EdgeTriangle a, ref EdgeTriangle b)
        {
            return a.I == b.I && a.J == b.J;
        }

        private bool EdgeTriangleLarger(ref EdgeTriangle a, ref EdgeTriangle b)
        {
            return a.I > b.I || (a.I == b.I && a.J > b.J);
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
                do
                {
                    ++i;
                } while (i <= right && EdgeTriangleLarger(ref pivot, ref a[i]));

                do
                {
                    --j;
                } while (EdgeTriangleLarger(ref a[j], ref pivot));

                if (i >= j)
                    break;
                Swap(ref a[i], ref a[j]);
            }

            Swap(ref a[left], ref a[j]);
            return j;
        }

        #endregion

        #region Initialization

        private void InitialComponent()
        {
            m_Mesh = GetComponent<MeshFilter>().mesh;
            mat = WorldToClothObject();
        }

        private Matrix4x4 WorldToClothObject()
        {
            var mat = Matrix4x4.identity;
            for (var i = 0; i < 3; i++) mat[3 * 4 + i] = -transform.position[i];
            return mat;
        }

        private void InitialArray()
        {
            totalVertecies = m_Mesh.vertexCount;
            totalSphere = collisionBall.Length;

            var pos = m_Mesh.vertices;
            m_TriangleIndices = new int[m_Mesh.triangles.Length];

            position = new Vector3[totalVertecies];
            velocity = new Vector3[totalVertecies];
            predictPosition = new Vector3[totalVertecies];
            vertexMass = new float[totalVertecies];
            pin = new int[totalVertecies];
            deltaInt = new Vector3Int[totalVertecies];
            m_CollideInfos = new CollideInfo[totalVertecies];

            if (enableCollision)
            {
                m_SphereData = new BallInfo[totalSphere];

                for (var i = 0; i < totalSphere; i++)
                {
                    m_SphereData[i] = collisionBall[i].GetBallInfo();
                    m_SphereData[i].ToLocalSpace(ref mat);
                }
            }


            for (var t = 0; t < m_Mesh.triangles.Length; t++) m_TriangleIndices[t] = m_Mesh.triangles[t];

            for (var i = 0; i < totalVertecies; i++)
            {
                position[i] = pos[i];
                predictPosition[i] = position[i];
                velocity[i] = Vector3.zero;
                vertexMass[i] = 0.0f;
                pin[i] = 0;
                deltaInt[i] = Vector3Int.zero;
            }

            foreach (var t in pinPoint) pin[t] = 1;
        }

        private void InitMass()
        {
            for (var i = 0; i < m_TriangleIndices.Length; i += 3)
            {
                var p1 = m_TriangleIndices[i];
                var p2 = m_TriangleIndices[i + 1];
                var p3 = m_TriangleIndices[i + 2];

                var v1 = position[p1];
                var v2 = position[p2];
                var v3 = position[p3];

                var e1 = v2 - v1;
                var e2 = v3 - v1;

                var area = Vector3.Cross(e1, e2).magnitude * 0.5f;
                var m = area * density;
                var avg = m / 3;
                vertexMass[p1] += avg;
                vertexMass[p2] += avg;
                vertexMass[p3] += avg;
            }
        }

        private void InitialEdges()
        {
            var edgeTriangles = new EdgeTriangle[m_TriangleIndices.Length];
            var triangleCount = 0;
            for (var i = 0; i < m_TriangleIndices.Length; i += 3)
            {
                edgeTriangles[i].I = m_TriangleIndices[i];
                edgeTriangles[i].J = m_TriangleIndices[i + 1];
                edgeTriangles[i + 1].I = m_TriangleIndices[i + 1];
                edgeTriangles[i + 1].J = m_TriangleIndices[i + 2];
                edgeTriangles[i + 2].I = m_TriangleIndices[i + 2];
                edgeTriangles[i + 2].J = m_TriangleIndices[i];
                edgeTriangles[i].triangle = triangleCount;
                edgeTriangles[i + 1].triangle = triangleCount;
                edgeTriangles[i + 2].triangle = triangleCount;
                triangleCount++;
            }

            for (var i = 0; i < m_TriangleIndices.Length; i++)
                if (edgeTriangles[i].I > edgeTriangles[i].J)
                    (edgeTriangles[i].I, edgeTriangles[i].J) = (edgeTriangles[i].J, edgeTriangles[i].I);

            QuickSort(edgeTriangles, 0, edgeTriangles.Length - 1);

            var edgeCount = 0;
            for (var i = 0; i < edgeTriangles.Length; i++)
            {
                var isRepeat = i != 0 && IsSameEdge(ref edgeTriangles[i], ref edgeTriangles[i - 1]);
                if (!isRepeat) edgeCount++;
            }

            m_Edges = new Edge[edgeCount];
            for (int i = 0, e = 0; i < edgeTriangles.Length; i++)
            {
                var edge = new Edge();
                edge.I = edgeTriangles[i].I;
                edge.J = edgeTriangles[i].J;

                var isRepeat = i != 0 && IsSameEdge(ref edgeTriangles[i], ref edgeTriangles[i - 1]);

                var updateEdgeIndex = 0;
                if (isRepeat)
                {
                    updateEdgeIndex = e - 1;
                    var preEdgeIndex = e - 1;
                    edge.nT1 = m_Edges[preEdgeIndex].nT1;
                    edge.nT2 = edgeTriangles[i].triangle;
                }
                else
                {
                    updateEdgeIndex = e;
                    edge.nT1 = edgeTriangles[i].triangle;
                    edge.nT2 = -1;
                    e++;
                }

                edge.SelfInit(position, m_TriangleIndices);

                m_Edges[updateEdgeIndex] = edge;
            }

            totalEdge = m_Edges.Length;
        }

        private void InitParameter()
        {
            verticesGroup = (int)Math.Ceiling((float)totalVertecies / Thread);
            edgeGroup = (int)Math.Ceiling((float)totalEdge / Thread);
        }

        private void InitBuffer()
        {
            edges = new ComputeBuffer(m_Edges.Length, 6 * sizeof(int) + 2 * sizeof(float));
            edges.SetData(m_Edges);

            positionBuffer = new ComputeBuffer(position.Length, 3 * sizeof(float));
            positionBuffer.SetData(position);

            predictPositionBuffer = new ComputeBuffer(predictPosition.Length, 3 * sizeof(float));
            predictPositionBuffer.SetData(position);

            velocityBuffer = new ComputeBuffer(velocity.Length, 3 * sizeof(float));
            velocityBuffer.SetData(velocity);

            massBuffer = new ComputeBuffer(vertexMass.Length, 1 * sizeof(float));
            massBuffer.SetData(vertexMass);

            pinPointBuffer = new ComputeBuffer(pin.Length, 1 * sizeof(int));
            pinPointBuffer.SetData(pin);

            deltaIntBuffer = new ComputeBuffer(deltaInt.Length, 3 * sizeof(int));
            deltaIntBuffer.SetData(deltaInt);

            sphereBuffer = new ComputeBuffer(totalSphere, 4 * sizeof(float));
            sphereBuffer.SetData(m_SphereData);

            collideInfoBuffer = new ComputeBuffer(totalVertecies, 6 * sizeof(float));
            collideInfoBuffer.SetData(m_CollideInfos);
        }

        private void InitShader()
        {
            kernelInitSimulation = shader.FindKernel("InitSimulation");
            kernelDistanceLimiting = shader.FindKernel("DistanceConstrain");
            kernelBendingLimitiing = shader.FindKernel("BendingConstrain");
            kernelAddCollisionConstrain = shader.FindKernel("AddCollisionConstrain");
            kernelCollisionConstain = shader.FindKernel("CollisionConstrain");
            kernelUpdatePosition = shader.FindKernel("UpdatePositionAndVelocity");

            shader.SetInt("totalVertices", totalVertecies);
            shader.SetInt("totalEdges", totalEdge);
            shader.SetInt("sphereBufferLength", totalSphere);
            shader.SetVector("forceField", forceField);
            shader.SetVector("g", G);
            shader.SetInt("iterateCount", iterateCount);
            shader.SetFloat("damper", damper);
            shader.SetFloat("stiffness", stiffness);
            shader.SetFloat("thickness", thickness);

            shader.SetBuffer(kernelInitSimulation, "pin", pinPointBuffer);
            shader.SetBuffer(kernelInitSimulation, "velocity", velocityBuffer);
            shader.SetBuffer(kernelInitSimulation, "position", positionBuffer);
            shader.SetBuffer(kernelInitSimulation, "masses", massBuffer);
            shader.SetBuffer(kernelInitSimulation, "predictPosition", predictPositionBuffer);

            shader.SetBuffer(kernelAddCollisionConstrain, "pin", pinPointBuffer);
            shader.SetBuffer(kernelAddCollisionConstrain, "collideInfo", collideInfoBuffer);
            shader.SetBuffer(kernelAddCollisionConstrain, "sphere_buffer", sphereBuffer);
            shader.SetBuffer(kernelAddCollisionConstrain, "position", positionBuffer);
            shader.SetBuffer(kernelAddCollisionConstrain, "predictPosition", predictPositionBuffer);

            shader.SetBuffer(kernelDistanceLimiting, "edges", edges);
            shader.SetBuffer(kernelDistanceLimiting, "masses", massBuffer);
            shader.SetBuffer(kernelDistanceLimiting, "predictPosition", predictPositionBuffer);
            shader.SetBuffer(kernelDistanceLimiting, "deltaIntBuffer", deltaIntBuffer);

            shader.SetBuffer(kernelBendingLimitiing, "edges", edges);
            shader.SetBuffer(kernelBendingLimitiing, "masses", massBuffer);
            shader.SetBuffer(kernelBendingLimitiing, "predictPosition", predictPositionBuffer);
            shader.SetBuffer(kernelBendingLimitiing, "deltaIntBuffer", deltaIntBuffer);

            shader.SetBuffer(kernelCollisionConstain, "pin", pinPointBuffer);
            shader.SetBuffer(kernelCollisionConstain, "velocity", velocityBuffer);
            shader.SetBuffer(kernelCollisionConstain, "position", positionBuffer);
            shader.SetBuffer(kernelCollisionConstain, "predictPosition", predictPositionBuffer);
            shader.SetBuffer(kernelCollisionConstain, "collideInfo", collideInfoBuffer);

            shader.SetBuffer(kernelUpdatePosition, "pin", pinPointBuffer);
            shader.SetBuffer(kernelUpdatePosition, "velocity", velocityBuffer);
            shader.SetBuffer(kernelUpdatePosition, "predictPosition", predictPositionBuffer);
            shader.SetBuffer(kernelUpdatePosition, "deltaIntBuffer", deltaIntBuffer);
            shader.SetBuffer(kernelUpdatePosition, "position", positionBuffer);
        }

        #endregion

        private void Update()
        {
            //Update Collider Data
            if (enableCollision)
            {
                for (var i = 0; i < totalSphere; i++)
                {
                    collisionBall[i].UpdateBallInfo();
                    m_SphereData[i] = collisionBall[i].GetBallInfo();
                    m_SphereData[i].ToLocalSpace(ref mat);
                }

                sphereBuffer.SetData(m_SphereData);
            }

            //Simulation
            shader.SetFloat(Dt, deltaTime);
            shader.SetVector("forceField", forceField);
            shader.SetVector("g", G);

            shader.Dispatch(kernelInitSimulation, verticesGroup, 1, 1);
            if (enableCollision)
                shader.Dispatch(kernelAddCollisionConstrain, verticesGroup, 1, 1);
            for (var i = 0; i < iterateCount; i++)
            {
                shader.SetInt(Step, i);
                shader.Dispatch(kernelDistanceLimiting, edgeGroup, 1, 1);
                shader.Dispatch(kernelBendingLimitiing, edgeGroup, 1, 1);
                if (enableCollision)
                    shader.Dispatch(kernelCollisionConstain, verticesGroup, 1, 1);
            }

            shader.Dispatch(kernelUpdatePosition, verticesGroup, 1, 1);

            positionBuffer.GetData(position);
            m_Mesh.SetVertices(position);
            m_Mesh.RecalculateNormals();
        }

        private void OnDisable()
        {
            edges?.Dispose();
            positionBuffer?.Dispose();
            predictPositionBuffer?.Dispose();
            velocityBuffer?.Dispose();
            massBuffer?.Dispose();
            deltaIntBuffer?.Dispose();
            pinPointBuffer?.Dispose();
            sphereBuffer?.Dispose();
            collideInfoBuffer?.Dispose();
        }
    }
}