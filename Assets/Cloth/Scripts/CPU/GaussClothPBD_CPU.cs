using System;
using Cloth.Scripts.gpu.GaussSeidel;
using UnityEngine;

namespace myCloth
{
    public struct DistanceEdge
    {
        public int I, J;
        public float restLength;

        public int nT1;
        public int nT2;

        public int nv1;
        public int nv2;
        public float restAngle;

        public DistanceEdge(int i, int j)
        {
            I = i;
            J = j;
            restLength = 0.0f;
            restAngle = 0.0f;
            nT1 = 0;
            nT2 = 0;
            nv1 = 0;
            nv2 = 0;
        }

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

    public struct EdgeTriangle
    {
        public int I, J;
        public int triangle;
    }

    public struct CollisionInfo
    {
        public Vector3 normal;
        public Vector3 position;
    }

    [Serializable]
    public struct CollisionConstrain
    {
        public Vector3 position;
        public Vector3 velocity;
    }

    [RequireComponent(typeof(MeshFilter))]
    [RequireComponent(typeof(MeshRenderer))]
    public class GaussClothPBD_CPU : MonoBehaviour
    {
        [SerializeField] private Vector3 G = new(0.0f, -9.8f, 0.0f);
        [SerializeField] private Vector3 forceField = new(0.0f, 0.0f, 0.0f);
        [SerializeField] private float deltaTime;
        [SerializeField] private float damper;
        [Range(0, 1)] [SerializeField] private float stiffness;
        [SerializeField] private int iterateCount;
        [SerializeField] private float density;
        [SerializeField] private float thickness;
        [SerializeField] private int[] pinPoint;
        [SerializeField] private SphereColliderProxy Sphere;

        private Mesh m_Mesh;
        private Matrix4x4 mat;
        private int m_VerticesCount;

        private int[] m_TriangleIndices;
        private DistanceEdge[] m_Edges;
        private Vector3[] position;
        private Vector3[] predictPosition;
        private Vector3[] velocity;
        private float[] vertexMass;
        private int[] pin;
        private CollisionInfo[] collisionInfos;
        private CollisionConstrain[] constrainsInfos;

        private int totalEdge;
        private int totalVertecies;
        private int totalQuad;

        public BallInfo collisionBall;

        private void Awake()
        {
            InitialComponent();
            InitialArray();
            InitialEdges();
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

        private void InitialArray()
        {
            totalVertecies = m_Mesh.vertexCount;
            var pos = m_Mesh.vertices;
            var mat = transform.localToWorldMatrix;
            m_TriangleIndices = new int[m_Mesh.triangles.Length];

            position = new Vector3[totalVertecies];
            velocity = new Vector3[totalVertecies];
            predictPosition = new Vector3[totalVertecies];
            vertexMass = new float[totalVertecies];
            pin = new int[totalVertecies];
            collisionInfos = new CollisionInfo[totalVertecies];
            constrainsInfos = new CollisionConstrain[totalVertecies];

            for (var t = 0; t < m_Mesh.triangles.Length; t++) m_TriangleIndices[t] = m_Mesh.triangles[t];

            for (var i = 0; i < totalVertecies; i++)
            {
                position[i] = pos[i];
                predictPosition[i] = position[i];
                velocity[i] = Vector3.zero;
                vertexMass[i] = 0.0f;
                pin[i] = 0;
            }

            foreach (var p in pinPoint) pin[p] = 1;

            InitMass();
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

            foreach (var pIndex in pinPoint) vertexMass[pIndex] = float.MaxValue;
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

            m_Edges = new DistanceEdge[edgeCount];
            for (int i = 0, e = 0; i < edgeTriangles.Length; i++)
            {
                var edge = new DistanceEdge();
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
        }

        #endregion


        private void Update()
        {
            Sphere.UpdateBallInfo();
            collisionBall = Sphere.GetBallInfo();
            collisionBall.center = mat.MultiplyPoint3x4(collisionBall.center);

            InitializeSimulate();
            AddCollisionConstrain();
            for (var i = 0; i < iterateCount; i++) StrainLimiting(i);
            UpdatePositionAndVelocity();

            m_Mesh.SetVertices(position);
            m_Mesh.RecalculateBounds();
            m_Mesh.RecalculateNormals();
        }

        private void InitializeSimulate()
        {
            for (var i = 0; i < totalVertecies; i++)
            {
                if (pin[i] == 1) continue;

                var v = velocity[i];
                var p = position[i];
                var a = G + forceField / vertexMass[i];
                v += a * deltaTime;
                v *= Mathf.Max(0.0f, 1 - damper * deltaTime / vertexMass[i]);
                p += v * deltaTime;

                velocity[i] = v;
                predictPosition[i] = p;
            }
        }

        private void StrainLimiting(int step)
        {
            DistanceConstrain(step);
            BendingConstrain(step);
            CollisionConstrain(step);
        }

        private void AddCollisionConstrain()
        {
            for (var i = 0; i < totalVertecies; i++)
            {
                if (pin[i] == 1) continue;

                var p = predictPosition[i];
                if (IsInSphere(p))
                {
                    pin[i] = 2;

                    var info = new CollisionInfo();
                    var normal = Vector3.Normalize(p - collisionBall.center);
                    var pos = normal * collisionBall.radius + collisionBall.center;
                    info.normal = normal;
                    info.position = pos + normal * thickness;

                    collisionInfos[i] = info;
                }
                else
                {
                    pin[i] = 0;
                }
            }
        }

        private bool IsInSphere(Vector3 p)
        {
            var d = p - collisionBall.center;
            var distance = d.magnitude - collisionBall.radius;
            return distance < 0;
        }

        private void CollisionConstrain(int step)
        {
            for (var i = 0; i < totalVertecies; i++)
            {
                if (pin[i] != 2) continue;

                var pos = predictPosition[i];
                var collisionInfo = collisionInfos[i];
                var deltaP = collisionInfo.position - pos;
                pos += deltaP * (step * 1.0f / iterateCount);
                predictPosition[i] = pos;
                velocity[i] = (predictPosition[i] - position[i]) / deltaTime;
            }
        }

        private void DistanceConstrain(int step)
        {
            var di = 1.0f / (iterateCount - step);
            var pow = 1.0f / iterateCount;
            var k = 1 - Mathf.Pow(1 - stiffness, pow);

            for (var i = 0; i < m_Edges.Length; i++)
            {
                var e = m_Edges[i];
                var p0 = predictPosition[e.I];
                var p1 = predictPosition[e.J];

                var m0 = vertexMass[e.I];
                var m1 = vertexMass[e.J];
                var distV = p1 - p0;
                var normal = Vector3.Normalize(distV);
                var length = distV.magnitude;
                var err = length - e.restLength;
                Vector3 correct;
                correct = normal * (k * err);
                var totalM = m0 + m1;

                predictPosition[e.I] += correct * (di * m1) / totalM;
                predictPosition[e.J] -= correct * (di * m0) / totalM;
            }
        }

        private void BendingConstrain(int step)
        {
            var di = 1.0f / (iterateCount - step);
            var pow = 1.0f / iterateCount;
            var k = 1 - Mathf.Pow(1 - stiffness, pow);

            for (var i = 0; i < m_Edges.Length; i++)
            {
                var edge = m_Edges[i];

                if (!edge.HaveTwoNeighborTriangle()) continue;

                var p1 = predictPosition[edge.I];
                var p2 = predictPosition[edge.J] - p1;
                var p3 = predictPosition[edge.nv1] - p1;
                var p4 = predictPosition[edge.nv2] - p1;

                var n1 = Vector3.Normalize(Vector3.Cross(p2, p3));
                var n2 = Vector3.Normalize(Vector3.Cross(p2, p4));

                var d = Vector3.Dot(n1, n2);

                var n1M = Vector3.Cross(p2, p3).magnitude;
                var n2M = Vector3.Cross(p2, p4).magnitude;

                var q4 = (Vector3.Cross(p2, n1) + Vector3.Cross(n2, p2) * d) / n2M;
                var q3 = (Vector3.Cross(p2, n2) + Vector3.Cross(n1, p2) * d) / n1M;
                var q2 = -(Vector3.Cross(p3, n2) + Vector3.Cross(n1, p3) * d) / n1M -
                         (Vector3.Cross(p4, n1) + Vector3.Cross(n2, p4) * d) / n2M;
                var q1 = -q2 - q3 - q4;

                var w1 = 1.0f / vertexMass[edge.I];
                var w2 = 1.0f / vertexMass[edge.J];
                var w3 = 1.0f / vertexMass[edge.nv1];
                var w4 = 1.0f / vertexMass[edge.nv2];

                var totalWeight = w1 * q1.magnitude * q1.magnitude + w2 * q2.magnitude * q2.magnitude +
                                  w3 * q3.magnitude * q3.magnitude + w4 * q4.magnitude * q4.magnitude;
                totalWeight = Mathf.Max(0.01f, totalWeight);

                var arcd = -(Mathf.Sqrt(1 - d * d) * (Mathf.Acos(d) - edge.restAngle)) / totalWeight;

                if (!float.IsNaN(arcd))
                {
                    predictPosition[edge.I] += q1 * (di * k * (w1 * arcd));
                    predictPosition[edge.J] += q2 * (di * k * (w2 * arcd));
                    predictPosition[edge.nv1] += q3 * (di * k * (w3 * arcd));
                    predictPosition[edge.nv2] += q4 * (di * k * (w4 * arcd));
                }
            }
        }

        private void UpdatePositionAndVelocity()
        {
            for (var i = 0; i < totalVertecies; i++)
            {
                if (pin[i] == 1) continue;
                velocity[i] = (predictPosition[i] - position[i]) / deltaTime;
                position[i] = predictPosition[i];
            }
        }

        private Matrix4x4 WorldToClothObject()
        {
            var mat = Matrix4x4.identity;
            for (var i = 0; i < 3; i++) mat[3 * 4 + i] = -transform.position[i];
            return mat;
        }
    }
}