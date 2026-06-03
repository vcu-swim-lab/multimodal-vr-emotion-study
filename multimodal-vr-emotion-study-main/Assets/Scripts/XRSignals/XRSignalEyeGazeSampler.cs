using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

internal sealed class XRSignalEyeGazeSampler
{
    private readonly List<InputDevice> eyeTrackingDevices = new();
    private OVRPlugin.EyeGazesState eyeGazesState;

    public bool TryGetEyeGazeSample(
        Transform eyeGazeTransform,
        Vector3 fallbackOrigin,
        out Vector3 origin,
        out Vector3 direction,
        out Quaternion rotation)
    {
        if (eyeGazeTransform != null)
        {
            origin = eyeGazeTransform.position;
            direction = eyeGazeTransform.forward;
            rotation = eyeGazeTransform.rotation;
            return true;
        }

        if (TryGetMetaEyeGazeSample(fallbackOrigin, out origin, out direction, out rotation))
        {
            return true;
        }

        return TryGetUnityEyeGazeSample(fallbackOrigin, out origin, out direction, out rotation);
    }

    private bool TryGetUnityEyeGazeSample(
        Vector3 fallbackOrigin,
        out Vector3 origin,
        out Vector3 direction,
        out Quaternion rotation)
    {
        eyeTrackingDevices.Clear();
        InputDevices.GetDevicesWithCharacteristics(InputDeviceCharacteristics.EyeTracking, eyeTrackingDevices);
        foreach (InputDevice device in eyeTrackingDevices)
        {
            if (!device.TryGetFeatureValue(CommonUsages.eyesData, out Eyes eyes))
            {
                continue;
            }

            origin = fallbackOrigin;
            if (eyes.TryGetLeftEyePosition(out Vector3 leftEyePosition) &&
                eyes.TryGetRightEyePosition(out Vector3 rightEyePosition))
            {
                origin = (leftEyePosition + rightEyePosition) * 0.5f;
            }

            if (eyes.TryGetFixationPoint(out Vector3 fixationPoint))
            {
                Vector3 fixationDirection = fixationPoint - origin;
                if (fixationDirection.sqrMagnitude > Mathf.Epsilon)
                {
                    direction = fixationDirection.normalized;
                    rotation = Quaternion.LookRotation(direction);
                    return true;
                }
            }

            if (eyes.TryGetLeftEyeRotation(out Quaternion leftEyeRotation) &&
                eyes.TryGetRightEyeRotation(out Quaternion rightEyeRotation))
            {
                rotation = Quaternion.Slerp(leftEyeRotation, rightEyeRotation, 0.5f);
                direction = rotation * Vector3.forward;
                return true;
            }
        }

        origin = Vector3.zero;
        direction = Vector3.zero;
        rotation = Quaternion.identity;
        return false;
    }

    private bool TryGetMetaEyeGazeSample(
        Vector3 fallbackOrigin,
        out Vector3 origin,
        out Vector3 direction,
        out Quaternion rotation)
    {
        origin = Vector3.zero;
        direction = Vector3.zero;
        rotation = Quaternion.identity;

        if (!OVRPlugin.eyeTrackingSupported)
        {
            return false;
        }

        if (!OVRPlugin.eyeTrackingEnabled && !OVRPlugin.StartEyeTracking())
        {
            return false;
        }

        if (!OVRPlugin.GetEyeGazesState(OVRPlugin.Step.Render, -1, ref eyeGazesState) ||
            eyeGazesState.EyeGazes == null ||
            eyeGazesState.EyeGazes.Length < (int)OVRPlugin.Eye.Count)
        {
            return false;
        }

        OVRPlugin.EyeGazeState leftEye = eyeGazesState.EyeGazes[(int)OVRPlugin.Eye.Left];
        OVRPlugin.EyeGazeState rightEye = eyeGazesState.EyeGazes[(int)OVRPlugin.Eye.Right];

        if (!leftEye.IsValid && !rightEye.IsValid)
        {
            return false;
        }

        if (leftEye.IsValid && rightEye.IsValid)
        {
            OVRPose leftPose = leftEye.Pose.ToOVRPose().ToWorldSpacePose(Camera.main);
            OVRPose rightPose = rightEye.Pose.ToOVRPose().ToWorldSpacePose(Camera.main);
            origin = (leftPose.position + rightPose.position) * 0.5f;
            rotation = Quaternion.Slerp(leftPose.orientation, rightPose.orientation, 0.5f);
        }
        else
        {
            OVRPlugin.EyeGazeState validEye = leftEye.IsValid ? leftEye : rightEye;
            OVRPose pose = validEye.Pose.ToOVRPose().ToWorldSpacePose(Camera.main);
            origin = pose.position;
            rotation = pose.orientation;
        }

        if (origin == Vector3.zero)
        {
            origin = fallbackOrigin;
        }

        direction = rotation * Vector3.forward;
        return true;
    }
}
