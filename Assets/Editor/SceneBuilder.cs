#if UNITY_EDITOR
using System.IO;
using RLPlatform.Env;
using RLPlatform.Net;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace RLPlatform.EditorTools
{
    // 씬/빌드를 코드로 재현 가능하게 생성한다 (메뉴 RL/* 또는 -executeMethod 배치 실행).
    public static class SceneBuilder
    {
        private const string ScenePath = "Assets/Scenes/RLEnv.unity";
        private const string BuildPath = "Builds/RLEnv/RLEnv.exe";

        [MenuItem("RL/Build Scene")]
        public static void BuildScene()
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            // 바닥: 윗면이 y=0 이 되도록 크기/위치는 런타임에 EnvParams 로 재설정됨
            GameObject floor = GameObject.CreatePrimitive(PrimitiveType.Cube);
            floor.name = "Floor";
            floor.transform.localScale = new Vector3(10f, 1f, 10f);
            floor.transform.position = new Vector3(0f, -0.5f, 0f);
            floor.GetComponent<Renderer>().sharedMaterial = MakeMaterial("Floor", new Color(0.35f, 0.4f, 0.45f));

            GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            cube.name = "TargetCube";
            cube.transform.position = new Vector3(3f, 0.5f, 3f);
            cube.GetComponent<Renderer>().sharedMaterial = MakeMaterial("Target", new Color(0.9f, 0.25f, 0.2f));
            var cubeRb = cube.AddComponent<Rigidbody>();
            cubeRb.isKinematic = true;     // 구체와 부딪혀도 밀리지 않는 고정 목표

            // 장애물 벽 (기본 길이 3m, 두께 0.5m)
            GameObject wall = GameObject.CreatePrimitive(PrimitiveType.Cube);
            wall.name = "ObstacleWall";
            wall.transform.localScale = new Vector3(3f, 1.2f, 0.5f);
            wall.transform.position = new Vector3(0f, 0.6f, 0f);
            wall.GetComponent<Renderer>().sharedMaterial = MakeMaterial("Wall", new Color(0.6f, 0.6f, 0.65f));
            var wallRb = wall.AddComponent<Rigidbody>();
            wallRb.isKinematic = true;

            GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.name = "Agent";
            sphere.transform.position = new Vector3(-3f, 0.51f, -3f);
            sphere.GetComponent<Renderer>().sharedMaterial = MakeMaterial("Agent", new Color(0.2f, 0.6f, 0.95f));
            var rb = sphere.AddComponent<Rigidbody>();
            rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;

            GameObject envObj = new GameObject("RLEnvironment");
            var env = envObj.AddComponent<CubeSphereEnv>();
            env.floor = floor.transform;
            env.target = cube.transform;
            env.wall = wall.transform;
            env.sphere = rb;
            var sensor = sphere.AddComponent<SphereContactSensor>();
            sensor.env = env;
            var server = envObj.AddComponent<TcpEnvServer>();
            server.environmentBehaviour = env;

            GameObject camObj = new GameObject("Main Camera");
            camObj.tag = "MainCamera";
            var cam = camObj.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.12f, 0.12f, 0.14f);
            camObj.transform.position = new Vector3(0f, 14f, -9f);
            camObj.transform.rotation = Quaternion.Euler(58f, 0f, 0f);

            GameObject lightObj = new GameObject("Directional Light");
            var light = lightObj.AddComponent<Light>();
            light.type = LightType.Directional;
            lightObj.transform.rotation = Quaternion.Euler(50f, -30f, 0f);

            Directory.CreateDirectory("Assets/Scenes");
            EditorSceneManager.SaveScene(scene, ScenePath);
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
            AssetDatabase.SaveAssets();
            Debug.Log($"[SceneBuilder] scene saved: {ScenePath}");
        }

        [MenuItem("RL/Build Windows Player")]
        public static void BuildPlayer()
        {
            BuildScene(); // 씬을 항상 최신 코드로 다시 빌드 후 플레이어 빌드 진행
            Directory.CreateDirectory(Path.GetDirectoryName(BuildPath));
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = BuildPath,
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None,
            });
            Debug.Log($"[SceneBuilder] build result: {report.summary.result} -> {BuildPath}");
            if (Application.isBatchMode)
                EditorApplication.Exit(report.summary.result == UnityEditor.Build.Reporting.BuildResult.Succeeded ? 0 : 1);
        }

        // 배치 모드 진입점: 씬 생성 후 종료
        public static void BuildSceneBatch()
        {
            BuildScene();
            EditorApplication.Exit(0);
        }

        private static Material MakeMaterial(string name, Color color)
        {
            Directory.CreateDirectory("Assets/Materials");
            string path = $"Assets/Materials/{name}.mat";
            var mat = new Material(Shader.Find("Standard")) { color = color };
            AssetDatabase.CreateAsset(mat, path);
            return mat;
        }
    }
}
#endif

