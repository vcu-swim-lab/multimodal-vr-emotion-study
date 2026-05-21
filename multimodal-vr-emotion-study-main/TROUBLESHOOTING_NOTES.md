# Quest Build Troubleshooting Notes

Project: Multimodal VR Emotion Study
Device observed in logs: Meta Quest Pro
Unity version observed: 6000.0.33f1
Runtime observed: Oculus OpenXR runtime

## Goal

Build and run the Unity VR emotion study app on a standalone Meta Quest headset. The app should launch, show the waiting/next screen, then play the emotion videos from the project.

## Problems Encountered

### 1. APK launched to a black screen / failed immediately

Initial Android logs showed the app activity could not be found:

```text
ClassNotFoundException: Didn't find class "com.unity3d.player.UnityPlayerGameActivity"
```

The APK manifest was launching `UnityPlayerGameActivity`, but the generated dex contained `UnityPlayerActivity` / `UnityPlayerForGameActivity` instead. This pointed to a mismatch in Unity Android application entry settings.

Fix applied in `ProjectSettings/ProjectSettings.asset`:

```yaml
androidApplicationEntry: 2
```

### 2. Custom Android manifest setting was enabled without a useful manifest

`useCustomMainManifest` was enabled, but `Assets/Plugins/Android` did not contain a custom manifest that was needed for this project. This could cause Unity to build with stale or mismatched Android launch settings.

Fix applied in `ProjectSettings/ProjectSettings.asset`:

```yaml
useCustomMainManifest: 0
```

### 3. Input System runtime exception

After the launch issue was fixed, Unity reported an input mismatch:

```text
InvalidOperationException: You are trying to read Input using UnityEngine.Input class,
but active Input handling is Input System package
```

The scene still had older UI input components that use `UnityEngine.Input`, while the project was set to only use the newer Input System.

Temporary compatibility fix applied in `ProjectSettings/ProjectSettings.asset`:

```yaml
activeInputHandler: 2
```

This means "Both" input systems are enabled. Unity warns this is not ideal on Android, but it allows the existing UI flow to run without replacing all legacy input components.

Longer-term cleaner fix: replace the old `StandaloneInputModule` with an `InputSystemUIInputModule` and update the UI input setup fully to the new Input System.

### 4. App launched but stayed dark

Later logs showed that the app was no longer crashing. It reached OpenXR session states and submitted frames:

```text
XR_SESSION_STATE_VISIBLE -> XR_SESSION_STATE_FOCUSED
Fully drawn com.alimert.emotiongathertool
FPS=72/72
```

This suggested the issue was now inside the Unity scene rather than Android startup.

Scene inspection found that the root UI canvas containing the `NextButton` was active, but its transform scale was zero:

```yaml
m_Name: Canvas
m_LocalScale: {x: 0, y: 0, z: 0}
```

The video coroutine waits for the user to press `Next` before playing the first video, so a hidden/unusable Next button can make the app appear dark forever.

Fix applied in `Assets/Scenes/Main.unity`:

```yaml
m_LocalScale: {x: 1, y: 1, z: 1}
```

### 5. Unsupported Vulkan dynamic resolution warning

Quest logs showed:

```text
Vulkan Dynamic Resolution is not supported on your current build version.
Ensure you are on Unity 2021+ with Oculus XR plugin v3.3.0+
```

The scene had OVR dynamic resolution enabled. Since the runtime rejected it, it was disabled.

Fix applied in `Assets/Scenes/Main.unity`:

```yaml
propertyPath: enableDynamicResolution
value: 0
```

### 6. Next button dependency on headset

The original flow required the UI `NextButton` to be clicked. On a standalone headset, depending only on a screen-space UI click can leave the user unable to advance.

Fix applied in `Assets/Scripts/Videos.cs`:

- Keep the original `NextButton` click behavior.
- Also allow advancing with:
  - Quest controller primary button
  - Quest controller trigger button
  - Space
  - Enter
  - Mouse click

This makes the experiment flow easier to test both in the editor and on the headset.

### 7. Videos were stored under `Assets/Videos/Emotions`

The emotion videos exist here in the Unity project:

```text
Assets/Videos/Emotions
```

However, on Android/Quest builds, this folder is not available as a normal filesystem path at runtime. The script used `VideoPlayer.url`, so it needs a real runtime-readable URL/path.

Fix applied:

- Copied the emotion MP4 files to:

```text
Assets/StreamingAssets/Videos/Emotions
```

- Updated `Assets/Scripts/Videos.cs` to use:

```csharp
Application.streamingAssetsPath
```

instead of:

```csharp
Application.dataPath
```

This allows the APK to include the files in a place that `VideoPlayer.url` can read on Quest.

## Remaining Known Issues

### Missing AuraCam script

The logs still show:

```text
The referenced script on this Behaviour (Game Object 'AuraCam') is missing!
```

Scene inspection found a missing script GUID on `AuraCam`:

```yaml
guid: 4446ac56c0cb7144cbe6d4650488fca0
```

Only `ViewpointCamera.cs` is present for the other AuraCam script reference. The missing component may have been a recording or face-expression capture helper. It should be restored if the research workflow depends on AuraCam recording or emotion/face capture output.

### Missing intro and acting prompt videos

`Videos.cs` still expects:

```text
Assets/StreamingAssets/Videos/Intro.mp4
Assets/StreamingAssets/Videos/EmotionTextVids/*.mp4
```

At the time of this note, the emotion videos are present, but the intro video and acting prompt videos were not found. If those phases are still required, the files should be added under `StreamingAssets` or the script should skip those phases when files are missing.

### Screen-space UI in VR

The current fix makes the canvas visible and adds controller button fallback. A more VR-native solution would be a world-space canvas with XR ray interaction, but the fallback input is enough for testing the current build flow.

## Current Build Checklist

1. Open the project in Unity 6000.0.33f1.
2. Confirm Android target is selected.
3. Confirm Meta/OpenXR settings are still enabled.
4. Confirm these settings remain:

```yaml
androidApplicationEntry: 2
useCustomMainManifest: 0
activeInputHandler: 2
```

5. Confirm emotion videos exist in:

```text
Assets/StreamingAssets/Videos/Emotions
```

6. Build and install the APK on Quest.
7. Launch the app.
8. Press A/X, trigger, Space, Enter, mouse click, or the UI Next button to advance.
9. Watch logcat for Unity errors related to missing video URLs, missing scripts, or VideoPlayer startup.

## Summary

The main startup problem was caused by Android launch activity settings. After that was fixed, the remaining dark-screen behavior was caused by runtime scene and asset-loading issues: the app was waiting for input while the canvas was scaled to zero, and the video files were being loaded from an editor-only `Assets` path instead of Android-readable `StreamingAssets`.
## Update: Android VideoPlayer Path Hardening

A later log still showed many Quest system errors, but no clear Unity `VideoPlayer` success or failure line. Most repeated errors were from the headset system, network authentication, audio services, passthrough/MRSS, or Meta services being offline:

```text
Unable to resolve graph.oculus.com
network_status=NO_ACTIVE_NETWORK
Could not acquire render data locks
GameManager not available
```

These are noisy and not necessarily the reason the emotion videos remain dark.

The important app-side risk is that Android `StreamingAssets` paths can resolve to a `jar:file://...apk!/assets/...` URL. Some native playback paths do not handle that as reliably as a real filesystem path. To make playback more robust, `Videos.cs` now copies each emotion MP4 from:

```text
Application.streamingAssetsPath/Videos/Emotions
```

to:

```text
Application.persistentDataPath/Videos/Emotions
```

on first launch, then plays from `persistentDataPath`.

The script now logs with a `[Videos]` prefix for:

- waiting for first input
- copying each video
- preparing each video
- playback start
- playback finish
- missing files
- VideoPlayer errors

After rebuilding, the most useful logcat filter is:

```text
Unity [Videos]
```

or simply search the device log for:

```text
[Videos]
```

If no `[Videos]` lines appear, the problem is earlier than the video script. If `[Videos] Failed to copy` appears, the APK did not include the MP4s correctly. If `[Videos] Failed to prepare` appears, the copied file path or video codec/container is the next suspect.

## Update: First Video Auto-Start

The app was still dark immediately after launch. The scene logic originally waited for a first `Next`/controller input before playing any video, so a dark or empty first screen could still happen even if the videos were packaged correctly.

`Videos.cs` was updated so the first emotion video starts automatically after a one-second delay. The `Next` button / controller trigger behavior is still used between videos.

For debugging, the `[Videos]` diagnostic messages were temporarily changed to `Debug.LogError` for major milestones so they appear even when logcat output is filtered to error-level lines. Important expected lines after launch are:

```text
[Videos] Auto-starting first emotion video in 1 second.
[Videos] Preparing <Emotion>: <path>
[Videos] Playing <Emotion>
```

If these lines do not appear after rebuilding and reinstalling, the APK being tested likely does not include the latest script changes or Unity is not reaching the `Videos` component.

## Update: Between-Video Auto-Advance

A later log confirmed the emotion videos were successfully copied and the first video played to completion:

```text
[Videos] Copied ...
[Videos] Auto-starting first emotion video in 1 second.
[Videos] Preparing Fear: .../Fear.mp4
[Videos] Playing Fear
[Videos] Finished Fear
```

This showed that packaging and playback were working. The remaining issue was experiment flow: after each video, `Videos.cs` waited for `AwaitButtonClick()`. On the standalone headset, that waiting state looked like the app was stuck after the first video.

`Videos.cs` was updated so `AwaitButtonClick()` still accepts the Next button/controller input, but also automatically advances after 2 seconds:

```csharp
private const float AutoAdvanceDelaySeconds = 2f;
```

This means the full emotion video sequence can run on the Quest without requiring visible UI interaction between every clip.

## Final Verification: Emotion Video Sequence Works

A later headset run confirmed the complete first phase works. The log showed all seven emotion videos copied from `StreamingAssets` to persistent storage, then played one after another with two-second auto-advance gaps.

Observed successful sequence:

```text
[Videos] Auto-starting first emotion video in 1 second.
[Videos] Preparing Sadness ...
[Videos] Playing Sadness
[Videos] Finished Sadness
[Videos] Advancing to next video.
[Videos] Preparing Neutral ...
[Videos] Playing Neutral
[Videos] Finished Neutral
[Videos] Advancing to next video.
[Videos] Preparing Surprise ...
[Videos] Playing Surprise
[Videos] Finished Surprise
[Videos] Advancing to next video.
[Videos] Preparing Anger ...
[Videos] Playing Anger
[Videos] Finished Anger
[Videos] Advancing to next video.
[Videos] Preparing Happiness ...
[Videos] Playing Happiness
[Videos] Finished Happiness
[Videos] Advancing to next video.
[Videos] Preparing Fear ...
[Videos] Playing Fear
[Videos] Finished Fear
[Videos] Advancing to next video.
[Videos] Preparing Disgust ...
[Videos] Playing Disgust
[Videos] Finished Disgust
```

This confirms:

- the APK contains the MP4 files,
- Android can copy them from the APK `StreamingAssets` location,
- playback from `Application.persistentDataPath` works on Quest,
- the first video auto-start works,
- between-video auto-advance works,
- all seven emotion videos are playable on the headset.

The only remaining warnings after this successful run were:

```text
[Videos] Intro video not found, skipping acting intro
[Videos] Acting prompt videos not found, skipping prompt phase
```

Those warnings are expected until `Intro.mp4` and the `EmotionTextVids` MP4 files are added to the build and copied with the same persistent-storage strategy.

## Recording Output Location

The README states that facial-expression experiment records are saved under the project-root folder:

```text
VideoRecordings/
```

Each run is expected to create a timestamped subfolder, for example:

```text
VideoRecordings/20260515-1103/
```

Observed file naming pattern from existing recordings:

```text
video_<Emotion>.mp4
weights_<Emotion>.csv
```

Meaning:

- `video_<Emotion>.mp4` is the recorded avatar/face video for that emotion.
- `weights_<Emotion>.csv` is the facial-expression / blendshape weight data for that emotion.

Example from an existing folder:

```text
VideoRecordings/20260515-1103/video_Fear.mp4
VideoRecordings/20260515-1103/weights_Fear.csv
```

Important caveat: the current project has a missing script attached to `AuraCam`:

```text
The referenced script on this Behaviour (Game Object 'AuraCam') is missing!
```

Scene inspection showed the missing component has fields that look like the recorder/capture script:

```yaml
faceExpressions: {fileID: 10}
renderTexture: ...
videoComponent: ...
```

This missing script is likely responsible for creating the `video_*.mp4` and `weights_*.csv` files. Existing old recording folders are present, but new Quest APK runs may not save facial-expression records until that missing recorder script is restored or rewritten.
