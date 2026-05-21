using System;
using System.Collections.Generic;
using System.Globalization;
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

    [Header("CSV Logging")]
    [SerializeField] private bool logPositionCsv = true;
    [SerializeField, Min(0.02f)] private float csvSampleInterval = 0.1f;

    private static readonly OVRFaceExpressions.FaceExpression[] FaceExpressionColumns = BuildFaceExpressionColumns();

    private Vector3 lastHeadPosition;
    private Quaternion lastHeadRotation;
    private bool hasPreviousHeadSample;

    private StreamWriter csvWriter;
    private string csvPath;
    private string sessionDirectory;
    private string currentEmotion;
    private float nextCsvSampleTime;
    private bool isLogging;

    public void StartDetectingSignals()
    {
        Debug.Log("=== Quest Signal Availability ===");

        DetectHeadSignals();
        DetectHandSignals();
        DetectControllerSignals();
        DetectFaceSignals();
        DetectUnsupportedSignals();
    }

    public void BeginLogging(string label)
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

        currentEmotion = SanitizePathPart(label);
        string emotionDirectory = Path.Combine(GetSessionDirectory(), currentEmotion);
        Directory.CreateDirectory(emotionDirectory);

        csvPath = Path.Combine(emotionDirectory, "weights.csv");
        csvWriter = new StreamWriter(csvPath, false, Encoding.UTF8);
        csvWriter.WriteLine(BuildCsvHeader());
        csvWriter.Flush();

        isLogging = true;
        nextCsvSampleTime = Time.time;

        Debug.Log($"[XRSignalLogger] Started weights CSV for {currentEmotion}: {csvPath}");
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

        nextCsvSampleTime = Time.time + csvSampleInterval;
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

    private void DetectUnsupportedSignals()
    {
        Debug.Log("Eye gaze transform logging: NOT CONFIGURED in XRSignalLogger");
    }

    private void WritePositionCsvSample()
    {
        if (csvWriter == null || string.IsNullOrEmpty(currentEmotion))
        {
            return;
        }

        Transform head = GetHeadTransform();
        Vector3 headPosition = head != null ? head.position : Vector3.zero;
        bool leftControllerTracked = IsRealTouchControllerTracked(OVRInput.Controller.LTouch);
        bool rightControllerTracked = IsRealTouchControllerTracked(OVRInput.Controller.RTouch);
        Vector3 leftControllerPosition = leftControllerTracked ? OVRInput.GetLocalControllerPosition(OVRInput.Controller.LTouch) : Vector3.zero;
        Vector3 rightControllerPosition = rightControllerTracked ? OVRInput.GetLocalControllerPosition(OVRInput.Controller.RTouch) : Vector3.zero;
        Vector3 leftHandPosition = leftHand != null ? leftHand.transform.position : Vector3.zero;
        Vector3 rightHandPosition = rightHand != null ? rightHand.transform.position : Vector3.zero;
        bool faceValid = faceExpressions != null && faceExpressions.ValidExpressions;

        var row = new List<string>
        {
            FormatFloat(Time.time),
            FormatFloat(Time.realtimeSinceStartup),
            currentEmotion,
            BoolToCsv(head != null),
            FormatFloat(headPosition.x),
            FormatFloat(headPosition.y),
            FormatFloat(headPosition.z),
            BoolToCsv(leftHand != null && leftHand.IsTracked),
            FormatFloat(leftHandPosition.x),
            FormatFloat(leftHandPosition.y),
            FormatFloat(leftHandPosition.z),
            BoolToCsv(rightHand != null && rightHand.IsTracked),
            FormatFloat(rightHandPosition.x),
            FormatFloat(rightHandPosition.y),
            FormatFloat(rightHandPosition.z),
            BoolToCsv(leftControllerTracked),
            FormatFloat(leftControllerPosition.x),
            FormatFloat(leftControllerPosition.y),
            FormatFloat(leftControllerPosition.z),
            BoolToCsv(rightControllerTracked),
            FormatFloat(rightControllerPosition.x),
            FormatFloat(rightControllerPosition.y),
            FormatFloat(rightControllerPosition.z),
            BoolToCsv(faceValid)
        };

        foreach (OVRFaceExpressions.FaceExpression expression in FaceExpressionColumns)
        {
            float weight = 0f;
            if (faceValid)
            {
                faceExpressions.TryGetFaceExpressionWeight(expression, out weight);
            }

            row.Add(FormatFloat(weight));
        }

        csvWriter.WriteLine(string.Join(",", row));
        csvWriter.Flush();
    }

    private string GetSessionDirectory()
    {
        if (!string.IsNullOrEmpty(sessionDirectory))
        {
            return sessionDirectory;
        }

        string timestamp = DateTime.Now.ToString("yyyyMMdd-HHmm", CultureInfo.InvariantCulture);
        sessionDirectory = Path.Combine(GetRecordingsRootDirectory(), timestamp);
        Directory.CreateDirectory(sessionDirectory);
        return sessionDirectory;
    }

    private static string GetRecordingsRootDirectory()
    {
#if UNITY_EDITOR || UNITY_STANDALONE_WIN
        string projectRoot = Directory.GetParent(Application.dataPath)?.FullName;
        if (!string.IsNullOrEmpty(projectRoot))
        {
            return Path.Combine(projectRoot, "VideoRecordings");
        }
#endif
        return Path.Combine(Application.persistentDataPath, "VideoRecordings");
    }
    private void CloseCsvWriter()
    {
        if (csvWriter == null)
        {
            return;
        }

        Debug.Log($"[XRSignalLogger] Closed weights CSV: {csvPath}");
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

    private static string BuildCsvHeader()
    {
        var header = new List<string>
        {
            "Timestamp",
            "RealtimeSeconds",
            "Emotion",
            "HeadTracked",
            "HeadX",
            "HeadY",
            "HeadZ",
            "LeftHandTracked",
            "LeftHandX",
            "LeftHandY",
            "LeftHandZ",
            "RightHandTracked",
            "RightHandX",
            "RightHandY",
            "RightHandZ",
            "LeftControllerTracked",
            "LeftControllerX",
            "LeftControllerY",
            "LeftControllerZ",
            "RightControllerTracked",
            "RightControllerX",
            "RightControllerY",
            "RightControllerZ",
            "FaceValid"
        };

        foreach (OVRFaceExpressions.FaceExpression expression in FaceExpressionColumns)
        {
            header.Add(expression.ToString());
        }

        return string.Join(",", header);
    }

    private static OVRFaceExpressions.FaceExpression[] BuildFaceExpressionColumns()
    {
        var columns = new List<OVRFaceExpressions.FaceExpression>();
        int first = (int)OVRFaceExpressions.FaceExpression.BrowLowererL;
        int last = (int)OVRFaceExpressions.FaceExpression.TongueRetreat;

        for (int i = first; i <= last; i++)
        {
            columns.Add((OVRFaceExpressions.FaceExpression)i);
        }

        return columns.ToArray();
    }

    private static bool IsEmotionLabel(string label)
    {
        return Enum.TryParse(label, out Emotion emotion) && emotion != Emotion.Max;
    }

    private static string SanitizePathPart(string value)
    {
        foreach (char invalidChar in Path.GetInvalidFileNameChars())
        {
            value = value.Replace(invalidChar, '_');
        }

        return value;
    }

    private static string FormatFloat(float value)
    {
        return value.ToString("G9", CultureInfo.InvariantCulture);
    }

    private static string BoolToCsv(bool value)
    {
        return value ? "1" : "0";
    }
}





