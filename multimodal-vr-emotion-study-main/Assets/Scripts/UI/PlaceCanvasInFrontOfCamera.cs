using System.Collections;
using UnityEngine;

public class PlaceCanvasInFrontOfCamera : MonoBehaviour
{
    [SerializeField] private Transform centerEyeAnchor;
    [SerializeField] private float distance = 0.65f;
    [SerializeField] private float heightOffset = 0f;

    private IEnumerator Start()
    {
        yield return null;
        yield return new WaitForEndOfFrame();

        PlaceInFrontOfHead();
    }

    private void PlaceInFrontOfHead()
    {
        Vector3 forward = centerEyeAnchor.forward;
        forward.y = 0f;
        forward.Normalize();

        transform.position =
            centerEyeAnchor.position +
            forward * distance +
            Vector3.up * heightOffset;

        transform.rotation = Quaternion.LookRotation(forward, Vector3.up);
    }
}