using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

public class XRSignalLogger : MonoBehaviour
{
    [Header("Head")]
    [SerializeField] private Transform centerEyeAnchor;
    [SerializeField] private Transform ovrHead;

    [Header("Hands")]
    [SerializeField] private OVRHand leftHand;
    [SerializeField] private OVRHand rightHand;

    [Header("Quest Pro Optional Signals")]
    [SerializeField] private OVRFaceExpressions faceExpressions;
    [SerializeField] private Transform eyeGazeTransform;

    [Header("CSV Logging")]
    [SerializeField] private bool logPositionCsv = true;
    private float csvSampleInterval = 1f / 60f; // 60 Hz

    private Vector3 lastHeadPosition;
    private Quaternion lastHeadRotation;
    private bool hasPreviousHeadSample;

    private StreamWriter csvWriter;
    private string csvPath;
    private string sessionTimestamp;
    private readonly Dictionary<RecordingPhase, string> sessionDirectories = new();
    private string currentEmotion;
    private float nextCsvSampleTime;
    private bool isLogging;
    private readonly List<string> csvRow = new(128);
    private readonly XRSignalEyeGazeSampler eyeGazeSampler = new();
    private readonly Dictionary<RecordingPhase, int> emotionFlowCounts = new();

    public void StartDetectingSignals()
    {
        Debug.Log("=== Quest Signal Availability ===");

        DetectHeadSignals();
        DetectHandSignals();
        DetectControllerSignals();
        DetectFaceSignals();
        DetectEyeGazeSignals();
    }

    public void BeginLogging(string label, RecordingPhase phase = RecordingPhase.Video)
    {
        if (!logPositionCsv)
        {
            return;
        }

        if (!IsEmotionLabel(label))
        {
            return;
        }

        EndLogging();

        currentEmotion = XRSignalPaths.SanitizePathPart(label);
        AppendEmotionFlow(currentEmotion, phase);
        string emotionDirectory = Path.Combine(GetSessionDirectory(phase), currentEmotion);
        Directory.CreateDirectory(emotionDirectory);

        csvPath = Path.Combine(emotionDirectory, XRSignalPaths.GetCsvFileName(phase));
        csvWriter = new StreamWriter(csvPath, false, Encoding.UTF8);
        csvWriter.WriteLine(XRSignalCsvFormatter.BuildCsvHeader());
        csvWriter.Flush();

        isLogging = true;
        nextCsvSampleTime = Time.time;

        Debug.Log($"[XRSignalLogger] Started {phase} CSV for {currentEmotion}: {csvPath}");
    }

    public void EndLogging()
    {
        if (!isLogging && csvWriter == null)
        {
            return;
        }

        WritePositionCsvSample();
        CloseCsvWriter();
        isLogging = false;
        currentEmotion = null;
    }

    private void Update()
    {
        if (!isLogging || !logPositionCsv || Time.time < nextCsvSampleTime)
        {
            return;
        }

        nextCsvSampleTime += csvSampleInterval;
        WritePositionCsvSample();
    }

    private void OnDisable()
    {
        EndLogging();
    }

    private void OnDestroy()
    {
        EndLogging();
    }

    private void WritePositionCsvSample()
    {
        if (csvWriter == null || string.IsNullOrEmpty(currentEmotion))
        {
            return;
        }

        Transform head = GetHeadTransform();
        Vector3 headPosition = head != null ? head.position : Vector3.zero;
        Quaternion headRotation = head != null ? head.rotation : Quaternion.identity;
        Vector3 headEuler = headRotation.eulerAngles;
        bool eyeGazeTracked = eyeGazeSampler.TryGetEyeGazeSample(
            eyeGazeTransform,
            headPosition,
            out Vector3 eyeGazeOrigin,
            out Vector3 eyeGazeDirection,
            out Quaternion eyeGazeRotation);
        Vector3 eyeGazeEuler = eyeGazeRotation.eulerAngles;

        bool leftControllerTracked = IsRealTouchControllerTracked(OVRInput.Controller.LTouch);
        bool rightControllerTracked = IsRealTouchControllerTracked(OVRInput.Controller.RTouch);
        Vector3 leftControllerPosition = leftControllerTracked ? OVRInput.GetLocalControllerPosition(OVRInput.Controller.LTouch) : Vector3.zero;
        Vector3 rightControllerPosition = rightControllerTracked ? OVRInput.GetLocalControllerPosition(OVRInput.Controller.RTouch) : Vector3.zero;
        Vector3 leftHandPosition = leftHand != null ? leftHand.transform.position : Vector3.zero;
        Vector3 rightHandPosition = rightHand != null ? rightHand.transform.position : Vector3.zero;

        OVRPlugin.SystemHeadset headset = OVRPlugin.GetSystemHeadsetType();
        bool isQuestPro =
            headset == OVRPlugin.SystemHeadset.Meta_Quest_Pro ||
            headset == OVRPlugin.SystemHeadset.Meta_Link_Quest_Pro;
        bool faceValid = isQuestPro &&
            faceExpressions != null &&
            faceExpressions.ValidExpressions;

        csvRow.Clear();
        csvRow.Add(XRSignalCsvFormatter.FormatFloat(Time.time));
        csvRow.Add(XRSignalCsvFormatter.FormatFloat(Time.realtimeSinceStartup));
        csvRow.Add(currentEmotion);
        csvRow.Add(XRSignalCsvFormatter.BoolToCsv(head != null));
        XRSignalCsvFormatter.AddVector3Fields(csvRow, headPosition);
        XRSignalCsvFormatter.AddEulerAndQuaternionFields(csvRow, headEuler, headRotation);
        XRSignalCsvFormatter.AddEyeGazeFields(csvRow, eyeGazeTracked, eyeGazeOrigin, eyeGazeDirection, eyeGazeEuler, eyeGazeRotation);
        XRSignalCsvFormatter.AddHandFields(
            csvRow,
            leftHand != null && leftHand.IsTracked,
            leftHandPosition,
            rightHand != null && rightHand.IsTracked,
            rightHandPosition);
        XRSignalCsvFormatter.AddControllerFields(csvRow, leftControllerTracked, leftControllerPosition, rightControllerTracked, rightControllerPosition);
        csvRow.Add(XRSignalCsvFormatter.BoolToCsv(faceValid));
        XRSignalCsvFormatter.AddFaceExpressionFields(csvRow, faceExpressions, faceValid);

        csvWriter.WriteLine(string.Join(",", csvRow));
    }

    private void DetectHeadSignals()
    {
        Transform head = GetHeadTransform();

        if (head == null)
        {
            Debug.Log("Head: NOT AVAILABLE - assign CenterEyeAnchor or OVRHead");
            return;
        }

        Vector3 position = head.position;
        Quaternion rotation = head.rotation;

        Debug.Log("Head: AVAILABLE");
        Debug.Log($"head_position: AVAILABLE | {position}");
        Debug.Log($"head_rotation: AVAILABLE | {rotation.eulerAngles}");

        if (hasPreviousHeadSample)
        {
            float dt = Mathf.Max(Time.deltaTime, Mathf.Epsilon);
            Vector3 headVelocity = (position - lastHeadPosition) / dt;
            float angle = Quaternion.Angle(lastHeadRotation, rotation);
            float headAngularVelocity = angle / dt;

            Debug.Log($"head_velocity: AVAILABLE | {headVelocity.magnitude} m/s");
            Debug.Log($"head_angular_velocity: AVAILABLE | {headAngularVelocity} deg/s");
        }
        else
        {
            Debug.Log("head_velocity: AVAILABLE AFTER SECOND SAMPLE");
            Debug.Log("head_angular_velocity: AVAILABLE AFTER SECOND SAMPLE");
        }

        lastHeadPosition = position;
        lastHeadRotation = rotation;
        hasPreviousHeadSample = true;

        Debug.Log($"tracking_validity: {(OVRManager.isHmdPresent ? "AVAILABLE / HMD PRESENT" : "NOT PRESENT")}");
    }

    private void DetectHandSignals()
    {
        Debug.Log("Hand Tracking:");

        if (leftHand == null)
        {
            Debug.Log("left_hand: NOT ASSIGNED");
        }
        else
        {
            Debug.Log($"left_hand: AVAILABLE | tracked={leftHand.IsTracked} | confidence={leftHand.HandConfidence}");
        }

        if (rightHand == null)
        {
            Debug.Log("right_hand: NOT ASSIGNED");
        }
        else
        {
            Debug.Log($"right_hand: AVAILABLE | tracked={rightHand.IsTracked} | confidence={rightHand.HandConfidence}");
        }
    }

    private void DetectControllerSignals()
    {
        OVRInput.Controller connected = OVRInput.GetConnectedControllers();
        OVRInput.Controller active = OVRInput.GetActiveController();

        Debug.Log("Controller Tracking:");
        Debug.Log($"controllers_connected: {connected}");
        Debug.Log($"active_controller: {active}");
        Debug.Log($"left_controller_position: {OVRInput.GetLocalControllerPosition(OVRInput.Controller.LTouch)}");
        Debug.Log($"right_controller_position: {OVRInput.GetLocalControllerPosition(OVRInput.Controller.RTouch)}");
        Debug.Log($"left_controller_rotation: {OVRInput.GetLocalControllerRotation(OVRInput.Controller.LTouch).eulerAngles}");
        Debug.Log($"right_controller_rotation: {OVRInput.GetLocalControllerRotation(OVRInput.Controller.RTouch).eulerAngles}");
        Debug.Log($"left_index_trigger: {OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, OVRInput.Controller.LTouch)}");
        Debug.Log($"right_index_trigger: {OVRInput.Get(OVRInput.Axis1D.PrimaryIndexTrigger, OVRInput.Controller.RTouch)}");
        Debug.Log($"left_grip: {OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger, OVRInput.Controller.LTouch)}");
        Debug.Log($"right_grip: {OVRInput.Get(OVRInput.Axis1D.PrimaryHandTrigger, OVRInput.Controller.RTouch)}");
        Debug.Log($"left_thumbstick: {OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, OVRInput.Controller.LTouch)}");
        Debug.Log($"right_thumbstick: {OVRInput.Get(OVRInput.Axis2D.PrimaryThumbstick, OVRInput.Controller.RTouch)}");
    }

    private void DetectFaceSignals()
    {
        if (faceExpressions == null)
        {
            Debug.Log("Face tracking: OPTIONAL / NOT ASSIGNED");
            return;
        }

        Debug.Log($"Face tracking: assigned={faceExpressions != null} | enabled={faceExpressions.FaceTrackingEnabled} | valid={faceExpressions.ValidExpressions}");
    }

    private void DetectEyeGazeSignals()
    {
        if (eyeGazeTransform == null)
        {
            Debug.Log($"Eye gaze transform logging: OPTIONAL / NOT ASSIGNED | supported={OVRPlugin.eyeTrackingSupported} | enabled={OVRPlugin.eyeTrackingEnabled}");
            return;
        }

        Debug.Log($"eye_gaze: AVAILABLE | position={eyeGazeTransform.position} | rotation={eyeGazeTransform.rotation.eulerAngles}");
    }

    private string GetSessionDirectory(RecordingPhase phase)
    {
        return XRSignalPaths.GetSessionDirectory(phase, sessionDirectories, ref sessionTimestamp);
    }

    private void CloseCsvWriter()
    {
        if (csvWriter == null)
        {
            return;
        }

        Debug.Log($"[XRSignalLogger] Closed CSV: {csvPath}");
        csvWriter.Flush();
        csvWriter.Close();
        csvWriter = null;
        csvPath = null;
    }

    private Transform GetHeadTransform()
    {
        return centerEyeAnchor != null ? centerEyeAnchor : ovrHead;
    }

    private static bool IsRealTouchControllerTracked(OVRInput.Controller controller)
    {
        OVRInput.Controller activeController = OVRInput.GetActiveController();
        if (HasControllerFlag(activeController, OVRInput.Controller.Hands) ||
            HasControllerFlag(activeController, OVRInput.Controller.LHand) ||
            HasControllerFlag(activeController, OVRInput.Controller.RHand))
        {
            return false;
        }

        OVRInput.Controller connectedControllers = OVRInput.GetConnectedControllers();
        bool controllerConnected = HasControllerFlag(connectedControllers, controller);
        return controllerConnected && OVRInput.GetControllerPositionTracked(controller);
    }

    private static bool HasControllerFlag(OVRInput.Controller value, OVRInput.Controller flag)
    {
        return (value & flag) == flag;
    }

    private static bool IsEmotionLabel(string label)
    {
        return Enum.TryParse(label, out Emotion emotion) && emotion != Emotion.Max;
    }

    private void AppendEmotionFlow(string emotion, RecordingPhase phase)
    {
        string sessionDirectory = GetSessionDirectory(phase);
        string flowPath = Path.Combine(sessionDirectory, "emotion_flow.txt");

        if (!emotionFlowCounts.ContainsKey(phase))
        {
            emotionFlowCounts[phase] = 0;
        }

        emotionFlowCounts[phase]++;

        EmotionFlowWriter.AppendEmotionFlowEntry(
            flowPath,
            emotionFlowCounts[phase],
            phase,
            emotion,
            DateTime.Now);
    }
}
