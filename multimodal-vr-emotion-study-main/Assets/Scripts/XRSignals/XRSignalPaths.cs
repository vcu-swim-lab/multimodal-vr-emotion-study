using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

internal static class XRSignalPaths
{
    public static string GetSessionDirectory(
        RecordingPhase phase,
        Dictionary<RecordingPhase, string> sessionDirectories,
        ref string sessionTimestamp)
    {
        if (sessionDirectories.TryGetValue(phase, out string existingDirectory))
        {
            return existingDirectory;
        }

        if (string.IsNullOrEmpty(sessionTimestamp))
        {
            sessionTimestamp = DateTime.Now.ToString("yyyyMMdd-HHmm", CultureInfo.InvariantCulture);
        }

        string sessionDirectory = Path.Combine(GetRecordingsRootDirectory(phase), sessionTimestamp);
        Directory.CreateDirectory(sessionDirectory);
        sessionDirectories[phase] = sessionDirectory;
        return sessionDirectory;
    }

    public static string GetCsvFileName(RecordingPhase phase)
    {
        return phase == RecordingPhase.Acting ? "acting.csv" : "weights.csv";
    }

    public static string SanitizePathPart(string value)
    {
        foreach (char invalidChar in Path.GetInvalidFileNameChars())
        {
            value = value.Replace(invalidChar, '_');
        }

        return value;
    }

    private static string GetRecordingsRootDirectory(RecordingPhase phase)
    {
        string rootFolder = phase == RecordingPhase.Acting ? "ActingRecordings" : "VideoRecordings";

#if UNITY_EDITOR || UNITY_STANDALONE_WIN
        string projectRoot = Directory.GetParent(Application.dataPath)?.FullName;
        if (!string.IsNullOrEmpty(projectRoot))
        {
            return Path.Combine(projectRoot, rootFolder);
        }
#endif
        return Path.Combine(Application.persistentDataPath, rootFolder);
    }
}
