using System;

namespace RLPlatform
{
    // 환경/Reward 파라미터. 실제 값의 단일 출처는 Python의 .env 이며, 연결 시 config 명령으로 주입된다.
    // 아래 초기값은 config 수신 전 에디터 단독 실행(수동 확인)용 폴백이다.
    [Serializable]
    public class EnvParams
    {
        public float floor_size = 10f;
        public float min_spawn_distance = 3f;
        public float spawn_margin = 1f;
        public float max_force = 10f;
        public float max_speed = 10f;
        public float fall_y = -1f;
        public int decision_interval = 5;
        public int max_steps = 300;
        public float fixed_dt = 0.02f;
        public float sphere_mass = 1f;
        public float sphere_drag = 1f;
        public float sphere_angular_drag = 0.5f;
        public float w_approach = 1f;
        public float r_success = 10f;
        public float r_fall = 10f;
        public float r_wall_hit = 1f;       // 벽 충돌 페널티
        public float c_step = 0.01f;
        public float c_action = 0.001f;
        public float wall_width = 3f;       // 벽 가로 길이 (반으로 축소: 6m -> 3m)
        public float wall_thickness = 0.5f; // 벽 두께
        // 0 이면 제한 없음(학습용 최고속도), >0 이면 초당 step 수를 제한(관찰/평가용)
        public float max_steps_per_sec = 0f;
    }
}

