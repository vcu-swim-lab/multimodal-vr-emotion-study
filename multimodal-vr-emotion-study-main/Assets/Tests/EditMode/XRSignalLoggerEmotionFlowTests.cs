using System;
using System.IO;
using NUnit.Framework;


public class XRSignalLoggerTests
{
    private string tempDirectory;

    [SetUp]
    public void SetUp()
    {
        tempDirectory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDirectory);
    }

    [TearDown]
    public void TearDown()
    {
        if (Directory.Exists(tempDirectory))
        {
            Directory.Delete(tempDirectory, true);
        }
    }

    [Test]
    public void AppendEmotionFlowEntry_AppendsEmotionsInOrder()
    {
        string flowPath = Path.Combine(tempDirectory, "emotion_flow.txt");

        EmotionFlowWriter.AppendEmotionFlowEntry(
            flowPath,
            1,
            RecordingPhase.Video,
            "Happiness",
            new DateTime(2026, 5, 26, 14, 30, 0));

        EmotionFlowWriter.AppendEmotionFlowEntry(
            flowPath,
            2,
            RecordingPhase.Video,
            "Sadness",
            new DateTime(2026, 5, 26, 14, 31, 0));

        string[] lines = File.ReadAllLines(flowPath);

        Assert.That(lines[0], Is.EqualTo("Emotion flow for Video"));
        Assert.That(lines[1], Is.EqualTo("1. Video - Happiness - 2026-05-26 14:30:00"));
        Assert.That(lines[2], Is.EqualTo("2. Video - Sadness - 2026-05-26 14:31:00"));
    }

    [Test]
    public void AppendEmotionFlowEntry_AllowsRepeatedEmotionForReplay()
    {
        string flowPath = Path.Combine(tempDirectory, "emotion_flow.txt");

        EmotionFlowWriter.AppendEmotionFlowEntry(
            flowPath,
            1,
            RecordingPhase.Video,
            "Happiness",
            new DateTime(2026, 5, 26, 14, 30, 0));

        EmotionFlowWriter.AppendEmotionFlowEntry(
            flowPath,
            2,
            RecordingPhase.Video,
            "Happiness",
            new DateTime(2026, 5, 26, 14, 35, 0));

        string[] lines = File.ReadAllLines(flowPath);

        Assert.That(lines[1], Is.EqualTo("1. Video - Happiness - 2026-05-26 14:30:00"));
        Assert.That(lines[2], Is.EqualTo("2. Video - Happiness - 2026-05-26 14:35:00"));
    }

    [Test]
    public void AppendEmotionFlowEntry_DoesNotOverwriteExistingHeader()
    {
        string flowPath = Path.Combine(tempDirectory, "emotion_flow.txt");

        EmotionFlowWriter.AppendEmotionFlowEntry(
            flowPath,
            1,
            RecordingPhase.Video,
            "Happiness",
            new DateTime(2026, 5, 26, 14, 30, 0));

        EmotionFlowWriter.AppendEmotionFlowEntry(
            flowPath,
            2,
            RecordingPhase.Video,
            "Sadness",
            new DateTime(2026, 5, 26, 14, 31, 0));

        string[] lines = File.ReadAllLines(flowPath);

        // Exactly 3 lines: 1 header + 2 entries (not 2 headers)
        Assert.That(lines.Length, Is.EqualTo(3));
        Assert.That(lines[0], Is.EqualTo("Emotion flow for Video"));
    }

    [Test]
    public void AppendEmotionFlowEntry_ActingPhase_UsesCorrectLabel()
    {
        string flowPath = Path.Combine(tempDirectory, "emotion_flow.txt");

        EmotionFlowWriter.AppendEmotionFlowEntry(
            flowPath,
            1,
            RecordingPhase.Acting,
            "Anger",
            new DateTime(2026, 5, 26, 9, 0, 0));

        string[] lines = File.ReadAllLines(flowPath);

        Assert.That(lines[0], Is.EqualTo("Emotion flow for Acting"));
        Assert.That(lines[1], Is.EqualTo("1. Acting - Anger - 2026-05-26 09:00:00"));
    }
}