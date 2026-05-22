using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.Video;
using UnityEngine.UI;
using UnityEngine.XR;
using TMPro;


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

internal class EmotionList
{
    public EmotionList()
    {
        const int max = (int)Emotion.Max;
        emotions = new Emotion[max];

        for (int i = 0; i < max; ++i)
            emotions[i] = (Emotion)i;

        for (int i = max - 1; i > 0; --i)
        {
            var j = Random.Range(0, i);
            (emotions[j], emotions[i]) = (emotions[i], emotions[j]);
        }
    }

    public Emotion Next()
    {
        if (current == emotions.Length)
            return Emotion.Max;

        return emotions[current++];
    }

    public int Count { get => emotions.Length - current; }

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

    private const string GoToActingButtonText = "Go to Acting Phase";
    private const float ActingPromptDurationSeconds = 10f;

    //Variable to hold "Video Player" component. Assigned in start function
    private VideoPlayer videoPlayer;
    private AudioSource vrAudioSource;

    [Header("UI Panels")]
    [SerializeField] private GameObject startPanel;
    [SerializeField] private GameObject betweenVideosPanel;
    [SerializeField] private GameObject actingPromptPanel;
    [SerializeField] private GameObject finishedPanel;

    [Header("Acting Prompt")]
    [SerializeField] private TMP_Text actingPromptText;

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
    private readonly Dictionary<Button, string> originalButtonLabels = new();
    private bool wasXrAdvancePressed;
    private bool videoHadError;

    // The collection of videos to play on the first part.
    private VideoList videos;

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

        CacheButtonLabel(nextButton);
        CacheButtonLabel(quitButton);

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

        yield return StartCoroutine(PlayActingPhase());

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

            bool isFinalVideo = list.Count == 0;
            yield return StartCoroutine(PlayVideoPath(videoPath, emotion.ToString()));
            yield return StartCoroutine(AwaitBetweenVideoDecision(videoPath, emotion.ToString(), isFinalVideo));
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
        xrSignalLogger?.BeginLogging(label, RecordingPhase.Video);
        yield return new WaitUntil(() => !videoPlayer.isPlaying || videoHadError);
        xrSignalLogger?.EndLogging();
        Debug.LogError($"[Videos] Finished {label}");
    }

    private IEnumerator PlayActingPhase()
    {
        var actingEmotions = new EmotionList();
        videoPlayer.Stop();

        if (actingPromptPanel == null || actingPromptText == null)
            Debug.LogWarning("[Videos] Acting prompt UI is not fully assigned. Acting recordings will still run, but the prompt may not be visible.");

        while (actingEmotions.Count > 0)
        {
            Emotion emotion = actingEmotions.Next();
            if (emotion == Emotion.Max)
                yield break;

            decision = VideoDecision.None;
            wasXrAdvancePressed = IsXrAdvancePressed();

            if (actingPromptText != null)
                actingPromptText.text = emotion.ToString();

            ShowOnly(actingPromptPanel);

            Debug.LogError($"[Videos] Acting prompt: {emotion}");
            xrSignalLogger?.BeginLogging(emotion.ToString(), RecordingPhase.Acting);

            yield return new WaitForSeconds(ActingPromptDurationSeconds);

            xrSignalLogger?.EndLogging();
            Debug.LogError($"[Videos] Finished acting prompt: {emotion}");

            yield return StartCoroutine(AwaitActingAdvanceDecision());
        }

        ShowOnly(null);
    }

    private IEnumerator AwaitBetweenVideoDecision(string videoPath, string label, bool isFinalVideo)
    {
        do
        {
            decision = VideoDecision.None;
            wasXrAdvancePressed = IsXrAdvancePressed();
            ConfigureBetweenVideoButtons(isFinalVideo);
            ShowOnly(betweenVideosPanel);

            yield return new WaitUntil(() => BetweenVideoDecisionRequested());

            if (decision == VideoDecision.Replay)
                yield return StartCoroutine(PlayVideoPath(videoPath, label));
        }
        while (decision == VideoDecision.Replay);

        RestoreBetweenVideoButtons();
        ShowOnly(null);
    }

    private IEnumerator AwaitActingAdvanceDecision()
    {
        decision = VideoDecision.None;
        wasXrAdvancePressed = IsXrAdvancePressed();
        ConfigureActingAdvanceButtons();
        ShowOnly(betweenVideosPanel);

        yield return new WaitUntil(() => ActingDecisionRequested());

        RestoreBetweenVideoButtons();
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
        if (decision == VideoDecision.Next || decision == VideoDecision.Replay)
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

        return false;
    }

    private bool ActingDecisionRequested()
    {
        if (decision == VideoDecision.Next)
            return true;

        if (Input.GetKeyDown(KeyCode.Space) || Input.GetKeyDown(KeyCode.Return) || XrAdvancePressedThisFrame())
        {
            decision = VideoDecision.Next;
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

    private void ConfigureBetweenVideoButtons(bool isFinalVideo)
    {
        SetButtonVisible(nextButton, true);
        SetButtonVisible(replayButton, true);

        if (isFinalVideo)
            SetButtonText(nextButton, GoToActingButtonText);
        else
            RestoreButtonText(nextButton);
    }

    private void RestoreBetweenVideoButtons()
    {
        SetButtonVisible(nextButton, true);
        SetButtonVisible(replayButton, true);
        RestoreButtonText(nextButton);
    }

    private void ConfigureActingAdvanceButtons()
    {
        SetButtonVisible(nextButton, true);
        SetButtonVisible(replayButton, false);
        RestoreButtonText(nextButton);
    }

    private static void SetButtonVisible(Button button, bool visible)
    {
        if (button != null)
            button.gameObject.SetActive(visible);
    }

    private void CacheButtonLabel(Button button)
    {
        if (button == null || originalButtonLabels.ContainsKey(button))
            return;

        TMP_Text tmpText = button.GetComponentInChildren<TMP_Text>(true);
        if (tmpText != null)
        {
            originalButtonLabels[button] = tmpText.text;
            return;
        }

        Text uiText = button.GetComponentInChildren<Text>(true);
        if (uiText != null)
            originalButtonLabels[button] = uiText.text;
    }

    private void SetButtonText(Button button, string text)
    {
        if (button == null)
            return;

        TMP_Text tmpText = button.GetComponentInChildren<TMP_Text>(true);
        if (tmpText != null)
        {
            tmpText.text = text;
            return;
        }

        Text uiText = button.GetComponentInChildren<Text>(true);
        if (uiText != null)
            uiText.text = text;
    }

    private void RestoreButtonText(Button button)
    {
        if (button == null || !originalButtonLabels.TryGetValue(button, out string label))
            return;

        SetButtonText(button, label);
    }

    private void ShowOnly(GameObject visiblePanel)
    {
        SetPanelVisible(startPanel, visiblePanel == startPanel);
        SetPanelVisible(betweenVideosPanel, visiblePanel == betweenVideosPanel);
        SetPanelVisible(actingPromptPanel, visiblePanel == actingPromptPanel);
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

    private static string STREAMING_EMOTION_VIDEOS_URL => $"{Application.streamingAssetsPath}/Videos/Emotions";
}



