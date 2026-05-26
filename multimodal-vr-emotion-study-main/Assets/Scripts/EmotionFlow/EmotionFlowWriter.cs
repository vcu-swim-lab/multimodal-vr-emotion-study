using System;
using System.IO;
using System.Text;

public enum RecordingPhase
{
    Video,
    Acting,
}

public static class EmotionFlowWriter
{
    public static void AppendEmotionFlowEntry(
        string flowPath,
        int order,
        RecordingPhase phase,
        string emotion,
        DateTime timestamp)
    {
        if (!File.Exists(flowPath))
        {
            File.WriteAllText(
                flowPath,
                $"Emotion flow for {phase}{Environment.NewLine}",
                Encoding.UTF8);
        }

        string line = $"{order}. {phase} - {emotion} - {timestamp:yyyy-MM-dd HH:mm:ss}";
        File.AppendAllText(flowPath, line + Environment.NewLine, Encoding.UTF8);
    }
}