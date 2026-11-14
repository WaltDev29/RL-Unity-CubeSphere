using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

namespace RLPlatform.Net
{
    // Unity 측 TCP 서버. 프레임: [uint32 big-endian 길이][UTF-8 JSON].
    // 소켓 I/O 는 백그라운드 스레드, 환경 처리(Unity API)는 메인 스레드 Update 에서만 수행한다.
    // IRLEnvironment 만 구현하면 다른 RL 환경에도 그대로 재사용 가능.
    public class TcpEnvServer : MonoBehaviour
    {
        public int port = 5005;                        // -rlport <n> 커맨드라인 인자로 덮어쓰기 가능
        public float frameBudgetMs = 12f;              // 프레임당 요청 처리 시간 상한 (렌더링 정지 방지)
        public MonoBehaviour environmentBehaviour;     // IRLEnvironment 구현체

        private IRLEnvironment env;
        private TcpListener listener;
        private Thread acceptThread;
        private volatile bool running;
        private readonly ConcurrentQueue<Pending> queue = new ConcurrentQueue<Pending>();
        private readonly SemaphoreSlim signal = new SemaphoreSlim(0);
        private float nextStepTime;
        private const int MaxFrameBytes = 1 << 20;

        private class Pending
        {
            public string requestJson;
            public string responseJson;
            public readonly ManualResetEventSlim done = new ManualResetEventSlim(false);
        }

        private void Awake()
        {
            Application.runInBackground = true;       // 에디터/창 비활성 상태에서도 통신 유지
            QualitySettings.vSyncCount = 0;
            env = environmentBehaviour as IRLEnvironment;
            if (env == null) Debug.LogError("[Server] environmentBehaviour 가 IRLEnvironment 를 구현하지 않음");

            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
                if (args[i] == "-rlport" && int.TryParse(args[i + 1], out int pp)) port = pp;
        }

        private void OnEnable()
        {
            running = true;
            try
            {
                listener = new TcpListener(IPAddress.Loopback, port);
                listener.Start();
                acceptThread = new Thread(AcceptLoop) { IsBackground = true, Name = "RLAccept" };
                acceptThread.Start();
                Debug.Log($"[Server] listening on 127.0.0.1:{port}");
            }
            catch (Exception e)
            {
                Debug.LogError($"[Server] failed to start on port {port}: {e.Message}");
            }
        }

        private void OnDisable()
        {
            running = false;
            try { listener?.Stop(); } catch { }
        }

        // ---- 네트워크 스레드 ----
        private void AcceptLoop()
        {
            while (running)
            {
                TcpClient client = null;
                try
                {
                    client = listener.AcceptTcpClient();
                    client.NoDelay = true;
                    Debug.Log("[Server] client connected");
                    Serve(client);
                }
                catch (Exception e)
                {
                    if (running) Debug.LogWarning($"[Server] connection ended: {e.Message}");
                }
                finally
                {
                    try { client?.Close(); } catch { }
                    Debug.Log("[Server] client disconnected");
                }
            }
        }

        private void Serve(TcpClient client)
        {
            NetworkStream stream = client.GetStream();
            byte[] lenBuf = new byte[4];
            while (running)
            {
                if (!ReadExact(stream, lenBuf, 4)) return;
                int len = (lenBuf[0] << 24) | (lenBuf[1] << 16) | (lenBuf[2] << 8) | lenBuf[3];
                if (len <= 0 || len > MaxFrameBytes) throw new IOException($"invalid frame length {len}");
                byte[] body = new byte[len];
                if (!ReadExact(stream, body, len)) return;

                var pending = new Pending { requestJson = Encoding.UTF8.GetString(body) };
                queue.Enqueue(pending);
                signal.Release();
                while (running && !pending.done.Wait(100)) { }
                if (!running) return;

                byte[] resp = Encoding.UTF8.GetBytes(pending.responseJson);
                byte[] header =
                {
                    (byte)(resp.Length >> 24), (byte)(resp.Length >> 16), (byte)(resp.Length >> 8), (byte)resp.Length
                };
                stream.Write(header, 0, 4);
                stream.Write(resp, 0, resp.Length);
            }
        }

        private static bool ReadExact(Stream s, byte[] buf, int count)
        {
            int read = 0;
            while (read < count)
            {
                int n = s.Read(buf, read, count - read);
                if (n <= 0) return false;      // 상대가 연결을 닫음
                read += n;
            }
            return true;
        }

        // ---- 메인 스레드 ----
        private void Update()
        {
            if (env == null) return;
            float start = Time.realtimeSinceStartup;
            bool processed = false;
            while (true)
            {
                if (!queue.TryPeek(out Pending next))
                {
                    // 직전에 처리했다면 곧 다음 요청이 오므로 짧게 대기해 프레임당 여러 step 을 처리
                    if (processed && (Time.realtimeSinceStartup - start) * 1000f < frameBudgetMs && signal.Wait(1))
                        continue;
                    break;
                }
                // 관찰용 속도 제한: step 요청은 설정된 초당 횟수만 처리
                if (env.MaxStepsPerSecond > 0f && Time.realtimeSinceStartup < nextStepTime && next.requestJson.Contains("\"step\""))
                    break;

                queue.TryDequeue(out Pending cur);
                signal.Wait(0);   // 처리한 요청만큼 신호 카운트를 소모
                cur.responseJson = Handle(cur.requestJson);
                cur.done.Set();
                processed = true;
                if (env.MaxStepsPerSecond > 0f) nextStepTime = Time.realtimeSinceStartup + 1f / env.MaxStepsPerSecond;
                if ((Time.realtimeSinceStartup - start) * 1000f > frameBudgetMs) break;
            }
        }

        private string Handle(string json)
        {
            try
            {
                Request req = JsonUtility.FromJson<Request>(json);
                Response resp;
                switch (req.cmd)
                {
                    case Protocol.CmdHello: resp = env.Hello(); break;
                    case Protocol.CmdConfig: env.Configure(req.env); resp = new Response(); break;
                    case Protocol.CmdReset: resp = env.Reset(req.seed); break;
                    case Protocol.CmdStep: resp = env.Step(req.action); break;
                    case Protocol.CmdClose: resp = new Response(); break;
                    default: resp = new Response { ok = false, error = $"unknown cmd '{req.cmd}'" }; break;
                }
                return JsonUtility.ToJson(resp);
            }
            catch (Exception e)
            {
                Debug.LogError($"[Server] request failed: {e}");
                return JsonUtility.ToJson(new Response { ok = false, error = e.Message });
            }
        }
    }
}
