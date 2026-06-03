using System.Collections.Generic;
using System.Globalization;
using UnityEngine;

internal static class XRSignalCsvFormatter
{
    private static readonly OVRFaceExpressions.FaceExpression[] FaceExpressionColumns = BuildFaceExpressionColumns();

    public static string BuildCsvHeader()
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
            "HeadPitch",
            "HeadYaw",
            "HeadRoll",
            "HeadRotX",
            "HeadRotY",
            "HeadRotZ",
            "HeadRotW",
            "EyeGazeTracked",
            "EyeGazeOriginX",
            "EyeGazeOriginY",
            "EyeGazeOriginZ",
            "EyeGazeDirX",
            "EyeGazeDirY",
            "EyeGazeDirZ",
            "EyeGazePitch",
            "EyeGazeYaw",
            "EyeGazeRoll",
            "EyeGazeRotX",
            "EyeGazeRotY",
            "EyeGazeRotZ",
            "EyeGazeRotW",
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

    public static string FormatFloat(float value)
    {
        return value.ToString("G9", CultureInfo.InvariantCulture);
    }

    public static string BoolToCsv(bool value)
    {
        return value ? "1" : "0";
    }

    public static void AddVector3Fields(List<string> row, Vector3 value)
    {
        row.Add(FormatFloat(value.x));
        row.Add(FormatFloat(value.y));
        row.Add(FormatFloat(value.z));
    }

    public static void AddEulerAndQuaternionFields(List<string> row, Vector3 euler, Quaternion rotation)
    {
        AddVector3Fields(row, euler);
        AddQuaternionFields(row, rotation);
    }

    public static void AddEyeGazeFields(
        List<string> row,
        bool tracked,
        Vector3 origin,
        Vector3 direction,
        Vector3 euler,
        Quaternion rotation)
    {
        row.Add(BoolToCsv(tracked));
        AddVector3Fields(row, origin);
        AddVector3Fields(row, direction);
        AddEulerAndQuaternionFields(row, euler, rotation);
    }

    public static void AddHandFields(
        List<string> row,
        bool leftTracked,
        Vector3 leftPosition,
        bool rightTracked,
        Vector3 rightPosition)
    {
        row.Add(BoolToCsv(leftTracked));
        AddVector3Fields(row, leftPosition);
        row.Add(BoolToCsv(rightTracked));
        AddVector3Fields(row, rightPosition);
    }

    public static void AddControllerFields(
        List<string> row,
        bool leftTracked,
        Vector3 leftPosition,
        bool rightTracked,
        Vector3 rightPosition)
    {
        row.Add(BoolToCsv(leftTracked));
        AddVector3Fields(row, leftPosition);
        row.Add(BoolToCsv(rightTracked));
        AddVector3Fields(row, rightPosition);
    }

    public static void AddFaceExpressionFields(List<string> row, OVRFaceExpressions faceExpressions, bool faceValid)
    {
        foreach (OVRFaceExpressions.FaceExpression expression in FaceExpressionColumns)
        {
            float weight = 0f;
            if (faceValid)
            {
                faceExpressions.TryGetFaceExpressionWeight(expression, out weight);
            }

            row.Add(FormatFloat(weight));
        }
    }

    private static void AddQuaternionFields(List<string> row, Quaternion value)
    {
        row.Add(FormatFloat(value.x));
        row.Add(FormatFloat(value.y));
        row.Add(FormatFloat(value.z));
        row.Add(FormatFloat(value.w));
    }

    private static OVRFaceExpressions.FaceExpression[] BuildFaceExpressionColumns()
    {
        int max = (int)OVRFaceExpressions.FaceExpression.Max;
        var columns = new OVRFaceExpressions.FaceExpression[max];

        for (int i = 0; i < max; i++)
        {
            columns[i] = (OVRFaceExpressions.FaceExpression)i;
        }

        return columns;
    }
}
