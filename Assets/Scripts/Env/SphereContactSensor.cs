using UnityEngine;

namespace RLPlatform.Env
{
    // 구체의 충돌 이벤트를 환경 스크립트로 전달한다 (Collision 콜백은 해당 Rigidbody 오브젝트에서만 호출되므로 분리).
    [RequireComponent(typeof(Rigidbody))]
    public class SphereContactSensor : MonoBehaviour
    {
        public CubeSphereEnv env;

        private void OnCollisionEnter(Collision collision)
        {
            if (env != null) env.OnSphereCollision(collision.collider);
        }

        private void OnCollisionStay(Collision collision)
        {
            if (env != null) env.OnSphereCollision(collision.collider);
        }
    }
}

