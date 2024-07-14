using System;
using UnityEngine;

namespace Cloth.Scripts.gpu.GaussSeidel
{
    [Serializable]
    public struct BallInfo
    {
        public Vector3 center;
        public float radius;

        public void ToLocalSpace(ref Matrix4x4 mat)
        {
            center = mat.MultiplyPoint3x4(center);
        }
    }
    
    public class SphereColliderProxy : MonoBehaviour
    {
        [SerializeField]
        private BallInfo info;
        private void Awake()
        {
            info = new BallInfo();
            info.center = transform.position;
            info.radius = transform.localScale.x*0.5f+0.2f;
        }

        public BallInfo GetBallInfo()
        {
            return info;
        }
        
        public void UpdateBallInfo()
        {
            info.center = transform.position;
        }
    }
}