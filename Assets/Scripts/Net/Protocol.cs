using System;

namespace RLPlatform.Net
{
    // Python <-> Unity 메시지 규격. JsonUtility가 Dictionary/가변 타입을 지원하지 않으므로
    // 모든 필드를 가진 고정 클래스로 정의한다 (사용하지 않는 필드는 Python이 무시).
    public static class Protocol
    {
        public const int Version = 1;
        public const string CmdHello = "hello";
        public const string CmdConfig = "config";
        public const string CmdReset = "reset";
        public const string CmdStep = "step";
        public const string CmdClose = "close";
    }

    [Serializable]
    public class Request
    {
        public string cmd;
        public int seed = -1;       // -1 이면 시드 미지정
        public float[] action;      // step 전용
        public EnvParams env;       // config 전용 (C# 예약어 params 를 피하기 위해 env 사용)
    }

    [Serializable]
    public class Response
    {
        public bool ok = true;
        public string error = "";
        public string env_name = "";
        public int protocol_version = Protocol.Version;
        public int obs_dim;
        public int act_dim;
        public float[] act_low;
        public float[] act_high;
        public float[] obs;
        public float reward;
        public bool done;          // 환경 규칙에 의한 종료 (성공/추락)
        public bool truncated;     // 최대 step 초과
        public bool success;
        public bool fell;
        public int step;
        public float distance;
    }

    // 환경 구현체가 따라야 하는 계약 - 다른 RL 환경에서도 TcpEnvServer 를 재사용하기 위함.
    public interface IRLEnvironment
    {
        string EnvName { get; }
        void Configure(EnvParams p);
        Response Hello();
        Response Reset(int seed);
        Response Step(float[] action);
        float MaxStepsPerSecond { get; }
    }
}

