using System;
using RLPlatform.Net;
using UnityEngine;

namespace RLPlatform.Env
{
    // 바닥 위 구체를 Force 로 굴려 장애물 벽을 우회해 정육면체에 접촉시키는 환경 (10차원 관측).
    public class CubeSphereEnv : MonoBehaviour, IRLEnvironment
    {
        public const int ObsDim = 10;
        public const int ActDim = 2;

        public Transform floor;
        public Transform target;       // 정육면체(목표)
        public Transform wall;         // 장애물 벽
        public Rigidbody sphere;

        private Collider targetCollider;
        private Collider wallCollider;
        private SphereCollider sphereCollider;
        private EnvParams p = new EnvParams();
        private int stepCount;
        private float prevDistance;
        private float bestDistance;   // 에피소드 내 달성한 역대 최단 거리 (보상 파밍 원천 방지)
        private bool success;
        private bool fell;
        private bool hitWall;
        private float sphereRadius = 0.5f;

        public string EnvName => "CubeSphereEnv";
        public float MaxStepsPerSecond => p.max_steps_per_sec;

        private void Awake()
        {
            // 직접 Simulate 를 호출하므로 자동 물리 갱신을 끈다.
            Physics.simulationMode = SimulationMode.Script;
            if (target != null) targetCollider = target.GetComponent<Collider>();
            if (wall != null) wallCollider = wall.GetComponent<Collider>();
            if (sphere != null) sphereCollider = sphere.GetComponent<SphereCollider>();
            ApplyParams();
        }

        public void Configure(EnvParams newParams)
        {
            if (newParams == null) throw new ArgumentException("env params is null");
            if (newParams.decision_interval < 1) throw new ArgumentException("decision_interval must be >= 1");
            if (newParams.floor_size <= 2f * newParams.spawn_margin + 1f)
                throw new ArgumentException("floor_size too small for spawn_margin");
            p = newParams;
            ApplyParams();
        }

        private void ApplyParams()
        {
            floor.localScale = new Vector3(p.floor_size, 1f, p.floor_size);
            floor.position = new Vector3(0f, -0.5f, 0f);   // 윗면이 y=0
            if (wall != null)
            {
                wall.localScale = new Vector3(p.wall_width, 1.2f, p.wall_thickness);
            }
            sphere.mass = p.sphere_mass;
            sphere.drag = p.sphere_drag;
            sphere.angularDrag = p.sphere_angular_drag;
            sphere.interpolation = RigidbodyInterpolation.None;
            sphereRadius = 0.5f * sphere.transform.localScale.x;
        }

        public Response Hello()
        {
            return new Response
            {
                env_name = EnvName,
                obs_dim = ObsDim,
                act_dim = ActDim,
                act_low = new[] { -1f, -1f },
                act_high = new[] { 1f, 1f },
                obs = new float[ObsDim],
            };
        }

        public Response Reset(int seed)
        {
            if (seed >= 0) UnityEngine.Random.InitState(seed);

            float half = p.floor_size * 0.5f - p.spawn_margin;
            Vector3 s = Vector3.zero, c = Vector3.zero;
            bool placed = false;

            // 공과 박스 사이의 거리가 최소 거리 조건을 만족할 때까지 샘플링
            for (int i = 0; i < 200 && !placed; i++)
            {
                s = new Vector3(UnityEngine.Random.Range(-half, half), 0f, UnityEngine.Random.Range(-half, half));
                c = new Vector3(UnityEngine.Random.Range(-half, half), 0f, UnityEngine.Random.Range(-half, half));
                placed = Vector3.Distance(s, c) >= p.min_spawn_distance;
            }
            if (!placed)
            {
                s = new Vector3(-half * 0.7f, 0f, -half * 0.7f);
                c = new Vector3(half * 0.7f, 0f, half * 0.7f);
            }

            sphere.velocity = Vector3.zero;
            sphere.angularVelocity = Vector3.zero;
            sphere.position = new Vector3(s.x, sphereRadius + 0.01f, s.z);
            sphere.rotation = Quaternion.identity;
            target.position = new Vector3(c.x, 0.5f * target.localScale.y, c.z);

            // 공과 박스의 정중앙에 벽 배치
            if (wall != null)
            {
                Vector3 midPoint = (s + c) * 0.5f;
                wall.position = new Vector3(midPoint.x, 0.6f, midPoint.z);

                // 공 -> 박스 방향을 법선으로 삼아 벽의 가로(X축)가 정확히 수직을 향하도록 회전 설정
                Vector3 toTarget = c - s;
                toTarget.y = 0f;
                if (toTarget.sqrMagnitude > 0.001f)
                {
                    wall.rotation = Quaternion.LookRotation(toTarget.normalized, Vector3.up);
                }
            }

            Physics.SyncTransforms();

            stepCount = 0;
            success = false;
            fell = false;
            hitWall = false;
            prevDistance = PlanarDistance();
            bestDistance = prevDistance;
            return Build(0f, false, false);
        }

        public Response Step(float[] action)
        {
            if (action == null || action.Length != ActDim)
                throw new ArgumentException($"action must have {ActDim} elements");

            float ax = Mathf.Clamp(action[0], -1f, 1f);
            float az = Mathf.Clamp(action[1], -1f, 1f);
            Vector3 force = new Vector3(ax, 0f, az) * p.max_force;

            for (int i = 0; i < p.decision_interval; i++)
            {
                sphere.AddForce(force, ForceMode.Force);
                Physics.Simulate(p.fixed_dt);
                if (sphere.position.y < p.fall_y) fell = true;
                if (success || fell) break;
            }
            stepCount++;

            float dist = PlanarDistance();
            if (dist <= 0.05f) success = true;

            float approachReward = 0f;
            if (dist < bestDistance)
            {
                approachReward = p.w_approach * (bestDistance - dist);
                bestDistance = dist;
            }

            float reward = approachReward
                           - p.c_step
                           - p.c_action * (ax * ax + az * az);
            if (hitWall)
            {
                reward -= p.r_wall_hit;
                hitWall = false;
            }
            prevDistance = dist;
            if (success) reward += p.r_success;
            if (fell) reward -= p.r_fall;

            bool done = success || fell;
            bool truncated = !done && stepCount >= p.max_steps;
            return Build(reward, done, truncated);
        }

        public void OnSphereCollision(Collider other)
        {
            if (target != null && other.gameObject == target.gameObject) success = true;
            if (wall != null && other.gameObject == wall.gameObject) hitWall = true;
        }

        // 큐브 표면과 구체 표면 사이의 평면(XZ) 최단 거리
        private float PlanarDistance()
        {
            if (targetCollider != null)
            {
                Vector3 closestPointOnTarget = targetCollider.ClosestPoint(sphere.position);
                Vector3 d = closestPointOnTarget - sphere.position;
                d.y = 0f;
                float surfaceDist = d.magnitude - sphereRadius;
                return Mathf.Max(0f, surfaceDist);
            }
            else
            {
                Vector3 d = target.position - sphere.position;
                d.y = 0f;
                return Mathf.Max(0f, d.magnitude - sphereRadius - 0.5f);
            }
        }

        // 관측 (10차원)
        private float[] Observe()
        {
            Vector3 relTarget = target.position - sphere.position;
            Vector3 vel = sphere.velocity;
            Vector3 pos = sphere.position;
            float halfFloor = p.floor_size * 0.5f;

            Vector3 closestOnWall = wallCollider != null ? wallCollider.ClosestPoint(sphere.position) : wall.position;
            Vector3 relWall = closestOnWall - sphere.position;

            float halfWall = p.wall_width * 0.5f + sphereRadius + 0.2f;
            Vector3 edgeLeft = wall.TransformPoint(new Vector3(-halfWall, 0f, 0f));
            Vector3 edgeRight = wall.TransformPoint(new Vector3(halfWall, 0f, 0f));

            Vector3 chosenEdge = (edgeLeft - sphere.position).sqrMagnitude < (edgeRight - sphere.position).sqrMagnitude
                ? edgeLeft : edgeRight;
            Vector3 relOpening = chosenEdge - sphere.position;

            return new[]
            {
                Clip(relTarget.x / p.floor_size), Clip(relTarget.z / p.floor_size),
                Clip(vel.x / p.max_speed), Clip(vel.z / p.max_speed),
                Clip(pos.x / halfFloor), Clip(pos.z / halfFloor),
                Clip(relWall.x / p.floor_size), Clip(relWall.z / p.floor_size),
                Clip(relOpening.x / p.floor_size), Clip(relOpening.z / p.floor_size)
            };
        }

        private static float Clip(float v) => Mathf.Clamp(v, -1f, 1f);

        private Response Build(float reward, bool done, bool truncated)
        {
            return new Response
            {
                env_name = EnvName,
                obs_dim = ObsDim,
                act_dim = ActDim,
                obs = Observe(),
                reward = reward,
                done = done,
                truncated = truncated,
                success = success,
                fell = fell,
                step = stepCount,
                distance = PlanarDistance(),
            };
        }
    }
}
