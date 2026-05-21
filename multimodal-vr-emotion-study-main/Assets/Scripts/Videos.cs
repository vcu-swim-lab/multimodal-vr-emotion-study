using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.Video;
using UnityEngine.UI;
using UnityEngine.XR;


public enum Emotion
{
    Anger,
    Disgust,
    Fear,
    Happiness,
    Neutral,
    Sadness,
    Surprise,
    Max,
}


internal class VideoList
{
    public VideoList(string folder)
    {
        this.folder = folder;

        const int max = (int)Emotion.Max;
        emotions = new Emotion[max];

        for (int i = 0; i < max; ++i)
            emotions[i] = (Emotion)i;

        // shuffle the emotions for fairness
        for (int i = max - 1; i > 0; --i)
        {
            var j = Random.Range(0, i);
            (emotions[j], emotions[i]) = (emotions[i], emotions[j]);
        }
    }

    public string Next(out Emotion emotion)
    {
        if (current == emotions.Length)
        {
            emotion = Emotion.Max;
            return null;
        }

        emotion = emotions[current++];
        return Path.Combine(folder, $"{emotion}.mp4");
    }

    public int Count { get => emotions.Length - current; }

    private readonly string folder;
    private readonly Emotion[] emotions;
    private int current = 0;
}

public class Videos : MonoBehaviour
{
    private enum VideoDecision
    {
        None,
        Play,
        Next,
        Replay,
        Quit,
    }

    //Variable to hold "Video Player" component. Assigned in start function
    private VideoPlayer videoPlayer;
    private AudioSource vrAudioSource;

    [Header("UI Panels")]
    [SerializeField] private GameObject startPanel;
    [SerializeField] private GameObject betweenVideosPanel;
    [SerializeField] private GameObject finishedPanel;

    [Header("UI Buttons")]
    [SerializeField] private Button playButton;
    [SerializeField] private Button quitButton;
    [SerializeField] private Button nextButton;
    [SerializeField] private Button replayButton;
    [SerializeField] private Button finalQuitButton;

    [Header("Signal Logging")]
    [SerializeField] private XRSignalLogger xrSignalLogger;

    private VideoDecision decision = VideoDecision.None;
    private readonly List<InputDevice> xrInputDevices = new();
    private bool wasXrAdvancePressed;
    private bool videoHadError;

    // The collection of videos to play on the first part.
    private VideoList videos;
    // The collection of videos to play on the second part.
    private VideoList emotionsText;

    private void Awake()
    {
        videoPlayer = GetComponent<VideoPlayer>();
        vrAudioSource = GetComponent<AudioSource>();

        videoPlayer.audioOutputMode = VideoAudioOutputMode.AudioSource;
        videoPlayer.SetTargetAudioSource(0, vrAudioSource);
        videoPlayer.errorReceived += (_, message) =>
        {
            videoHadError = true;
            Debug.LogError($"[Videos] VideoPlayer error for '{videoPlayer.url}': {message}");
        };

        vrAudioSource.ignoreListenerPause = true;

        if (xrSignalLogger == null)
            xrSignalLogger = FindFirstObjectByType<XRSignalLogger>();

        RegisterButton(playButton, VideoDecision.Play);
        RegisterButton(quitButton, VideoDecision.Quit);
        RegisterButton(nextButton, VideoDecision.Next);
        RegisterButton(replayButton, VideoDecision.Replay);
        RegisterButton(finalQuitButton, VideoDecision.Quit);

        ShowOnly(startPanel);
    }

    private IEnumerator Start()
    {
        yield return StartCoroutine(EnsureEmotionVideosAvailable());

        videos = new VideoList(VIDEOS_FOLDER);
        emotionsText = new VideoList(TEXT_VIDEOS_FOLDER);

        yield return StartCoroutine(AnimationCoro());
    }

    public VideoPlayer GetVideoPlayer() => videoPlayer;

    private IEnumerator AnimationCoro()
    {
        ShowOnly(startPanel);
        decision = VideoDecision.None;

        yield return new WaitUntil(() => StartRequested());

        if (decision == VideoDecision.Quit)
        {
            QuitApplication();
            yield break;
        }

        ShowOnly(null);

        yield return StartCoroutine(PlayVideoList(videos));

        if (File.Exists(INTRO_PATH))
        {
            yield return StartCoroutine(PlayVideoPath(INTRO_PATH, "intro"));
            yield return StartCoroutine(AwaitBetweenVideoDecision(INTRO_PATH, "intro"));
        }
        else
        {
            Debug.LogWarning($"[Videos] Intro video not found, skipping acting intro: {INTRO_PATH}");
        }

        if (Directory.Exists(TEXT_VIDEOS_FOLDER) && Directory.GetFiles(TEXT_VIDEOS_FOLDER, "*.mp4").Length > 0)
        {
            yield return StartCoroutine(PlayVideoList(emotionsText));
        }
        else
        {
            Debug.LogWarning($"[Videos] Acting prompt videos not found, skipping prompt phase: {TEXT_VIDEOS_FOLDER}");
        }

        videoPlayer.Stop();
        ShowOnly(finishedPanel);
        decision = VideoDecision.None;
        yield return new WaitUntil(() => FinishRequested());
        QuitApplication();
    }

    private IEnumerator PlayVideoList(VideoList list)
    {
        while (list.Count > 0)
        {
            var videoPath = list.Next(out var emotion);
            if (string.IsNullOrEmpty(videoPath) || !File.Exists(videoPath))
            {
                Debug.LogError($"[Videos] Missing video for {emotion}: {videoPath}");
                continue;
            }

            yield return StartCoroutine(PlayVideoPath(videoPath, emotion.ToString()));
            yield return StartCoroutine(AwaitBetweenVideoDecision(videoPath, emotion.ToString()));
        }
    }

    private IEnumerator PlayVideoPath(string videoPath, string label)
    {
        ShowOnly(null);
        videoHadError = false;
        videoPlayer.url = videoPath;
        Debug.LogError($"[Videos] Preparing {label}: {videoPath}");

        videoPlayer.Prepare();
        var prepareStart = Time.realtimeSinceStartup;
        yield return new WaitUntil(() => videoPlayer.isPrepared || videoHadError || Time.realtimeSinceStartup - prepareStart > 10f);
        if (!videoPlayer.isPrepared || videoHadError)
        {
            Debug.LogError($"[Videos] Failed to prepare {label}: {videoPath}");
            yield break;
        }

        videoPlayer.Play();
        var playStart = Time.realtimeSinceStartup;
        yield return new WaitUntil(() => videoPlayer.isPlaying || videoHadError || Time.realtimeSinceStartup - playStart > 5f);
        if (!videoPlayer.isPlaying || videoHadError)
        {
            Debug.LogError($"[Videos] Failed to start {label}: {videoPath}");
            yield break;
        }

        Debug.LogError($"[Videos] Playing {label}");
        xrSignalLogger?.BeginLogging(label);
        yield return new WaitUntil(() => !videoPlayer.isPlaying || videoHadError);
        xrSignalLogger?.EndLogging();
        Debug.LogError($"[Videos] Finished {label}");
    }

    private IEnumerator AwaitBetweenVideoDecision(string videoPath, string label)
    {
        do
        {
            decision = VideoDecision.None;
            wasXrAdvancePressed = IsXrAdvancePressed();
            ShowOnly(betweenVideosPanel);

            yield return new WaitUntil(() => BetweenVideoDecisionRequested());

            if (decision == VideoDecision.Quit)
            {
                QuitApplication();
                yield break;
            }

            if (decision == VideoDecision.Replay)
                yield return StartCoroutine(PlayVideoPath(videoPath, label));
        }
        while (decision == VideoDecision.Replay);

        ShowOnly(null);
    }

    private bool StartRequested()
    {
        if (decision == VideoDecision.Play || decision == VideoDecision.Quit)
            return true;

        if (Input.GetKeyDown(KeyCode.Space) || Input.GetKeyDown(KeyCode.Return))
        {
            decision = VideoDecision.Play;
            return true;
        }

        if (Input.GetKeyDown(KeyCode.Escape))
        {
            decision = VideoDecision.Quit;
            return true;
        }

        return false;
    }

    private bool BetweenVideoDecisionRequested()
    {
        if (decision == VideoDecision.Next || decision == VideoDecision.Replay || decision == VideoDecision.Quit)
            return true;

        if (Input.GetKeyDown(KeyCode.Space) || Input.GetKeyDown(KeyCode.Return) || XrAdvancePressedThisFrame())
        {
            decision = VideoDecision.Next;
            return true;
        }

        if (Input.GetKeyDown(KeyCode.R))
        {
            decision = VideoDecision.Replay;
            return true;
        }

        if (Input.GetKeyDown(KeyCode.Escape))
        {
            decision = VideoDecision.Quit;
            return true;
        }

        return false;
    }

    private bool FinishRequested()
    {
        if (decision == VideoDecision.Quit)
            return true;

        if (Input.GetKeyDown(KeyCode.Space) || Input.GetKeyDown(KeyCode.Return) || Input.GetKeyDown(KeyCode.Escape))
        {
            decision = VideoDecision.Quit;
            return true;
        }

        return false;
    }

    private bool XrAdvancePressedThisFrame()
    {
        var xrAdvancePressed = IsXrAdvancePressed();
        var requested = xrAdvancePressed && !wasXrAdvancePressed;
        wasXrAdvancePressed = xrAdvancePressed;
        return requested;
    }

    private bool IsXrAdvancePressed()
    {
        xrInputDevices.Clear();
        InputDevices.GetDevicesWithCharacteristics(InputDeviceCharacteristics.Controller, xrInputDevices);

        foreach (var device in xrInputDevices)
        {
            if (device.TryGetFeatureValue(CommonUsages.primaryButton, out var primaryButton) && primaryButton)
                return true;

            if (device.TryGetFeatureValue(CommonUsages.triggerButton, out var triggerButton) && triggerButton)
                return true;
        }

        return false;
    }

    private void RegisterButton(Button button, VideoDecision buttonDecision)
    {
        if (button == null)
            return;

        button.onClick.AddListener(() => decision = buttonDecision);
    }

    private void ShowOnly(GameObject visiblePanel)
    {
        SetPanelVisible(startPanel, visiblePanel == startPanel);
        SetPanelVisible(betweenVideosPanel, visiblePanel == betweenVideosPanel);
        SetPanelVisible(finishedPanel, visiblePanel == finishedPanel);
    }

    private static void SetPanelVisible(GameObject panel, bool visible)
    {
        if (panel != null)
            panel.SetActive(visible);
    }

    private void QuitApplication()
    {
        Debug.LogError("[Videos] Quitting application.");
        Application.Quit();
    }

    private IEnumerator EnsureEmotionVideosAvailable()
    {
        Directory.CreateDirectory(VIDEOS_FOLDER);

        for (int i = 0; i < (int)Emotion.Max; ++i)
        {
            var emotion = (Emotion)i;
            var fileName = $"{emotion}.mp4";
            var destinationPath = Path.Combine(VIDEOS_FOLDER, fileName);

            if (File.Exists(destinationPath) && new FileInfo(destinationPath).Length > 0)
                continue;

            var sourceUrl = $"{STREAMING_EMOTION_VIDEOS_URL}/{fileName}";
            Debug.LogError($"[Videos] Copying video from StreamingAssets: {sourceUrl}");

            using var request = UnityWebRequest.Get(sourceUrl);
            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[Videos] Failed to copy {fileName}: {request.error} from {sourceUrl}");
                continue;
            }

            File.WriteAllBytes(destinationPath, request.downloadHandler.data);
            Debug.LogError($"[Videos] Copied {fileName} to {destinationPath}");
        }
    }

    private static string VIDEOS_FOLDER => Path.Combine(Application.persistentDataPath, "Videos", "Emotions");

    private static string INTRO_PATH => Path.Combine(Application.persistentDataPath, "Videos", "Intro.mp4");

    private static string TEXT_VIDEOS_FOLDER => Path.Combine(Application.persistentDataPath, "Videos", "EmotionTextVids");

    private static string STREAMING_EMOTION_VIDEOS_URL => $"{Application.streamingAssetsPath}/Videos/Emotions";
}



