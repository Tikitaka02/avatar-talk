# avatar-talk

Browser-based motion capture that retargets your webcam movements onto a 3D
avatar — built step by step as a YouTube / blog series. Everything runs
client-side: no server, no installs, and no video ever leaves your machine.

**Current state — Phase 3:** the avatar moves when you move. Upper-body pose
tracking ([MediaPipe](https://ai.google.dev/edge/mediapipe) PoseLandmarker, lite
model, GPU delegate) is retargeted onto a VRM rig rendered with three.js and
[@pixiv/three-vrm](https://github.com/pixiv/three-vrm) — arms, torso and head,
smoothed, with expression controls and drag-and-drop avatar swapping.

## Roadmap

| Phase | Episode | Tag | Status |
|-------|---------|-----|--------|
| 1 | Webcam + pose landmark overlay | `v1-landmarks` | ✅ done |
| 2 | Loading a 3D VRM avatar (three.js + @pixiv/three-vrm) | `v2-avatar` | ✅ done |
| 3 | Retargeting: landmarks → bone rotations | `v3-retargeting` | ✅ done |
| 4 | Better smoothing, face expressions, hands, finger tracking | — | next |

Each episode is a git tag, so you can check out exactly the code from any
video: `git checkout v1-landmarks`.

## Running it

```bash
npm install
npm run dev
```

Open http://localhost:5173 and allow camera access. You should see yourself
with a green/pink skeleton tracking your face, torso, and arms at ~30 fps,
next to a 3D avatar.

**Bring your own avatar.** No `.vrm` is committed (they are large, and their
licences are the author's), so the 3D pane starts empty. Drag any VRM file
onto it — free ones are available on [VRoid Hub](https://hub.vroid.com/), or
make your own in VRoid Studio. To have one load automatically, drop it in
`public/avatar/` and point `DEFAULT_VRM` in `src/main.ts` at it.

## Design notes (phase 3)

- **Positions in, rotations out.** A skeleton cannot be moved to a point, only
  rotated, so each landmark becomes an angle: take the bone's rest direction in
  the normalized rig, rotate it onto the tracked direction, then divide out the
  accumulated rotation of the chain above it so the result is relative to the
  parent. The forearm gets this for free — its parent already includes the upper
  arm, so what remains is the elbow bend.
- **Two coordinate systems.** MediaPipe is metric with y down and z growing away
  from the camera; VRM is y up and +z forward. Flipping y and z is a 180° turn
  about x. Flipping x too mirrors the pose, which is why right-side landmarks
  drive the avatar's left bones.
- **Depth is a guess.** A single camera cannot measure z, so the model infers it
  and drifts badly — a symmetric pose can report one elbow twice as far forward
  as the other. `Z_TRUST` shrinks z so limbs stay near the silhouette the camera
  actually saw.
- **Strictly increasing timestamps.** MediaPipe treats a repeated timestamp as
  fatal: the graph errors and *every later frame fails for the session*. Two
  frames can share a millisecond, so the timestamp is forced forward, inference
  is wrapped, and `droppedFrames` makes degraded tracking visible.
- **Smoothing between camera frames.** The webcam yields ~30 poses a second
  while the display refreshes faster, so the last pose is re-applied every frame
  and bones ease toward it at a frame-rate-independent rate. Limbs that leave
  the frame drift home instead of freezing.

## Design notes (phase 2)

- **Standardized bones are the whole point.** VRM defines a fixed humanoid
  bone map, so `leftUpperArm` means the same thing in every avatar. Phase 3's
  retargeting will therefore work with any VRM, not just one rig.
- **Auto-framing.** The camera composes each avatar from chest height to the
  top of its bounding box, so a tall hat or spiky hair is never clipped
  regardless of the model's height.
- **Rest pose.** VRM files are authored in a T-pose; the arms are lowered into
  a relaxed A-pose on load so the avatar looks natural before tracking drives
  it.
- **One loop, two halves.** Tracking only advances on new webcam frames while
  the 3D scene redraws every frame. The loop starts *before* the camera is
  requested, so a slow permission prompt never freezes the avatar.

## Design notes (phase 1)

- **Upper body only.** Leg tracking from a desk webcam is unreliable, so
  landmarks 25–32 are never drawn or used.
- **Visibility filtering.** Landmarks below a 0.5 confidence score are
  skipped — this stops "ghost limbs" when a hand leaves the frame.
- **Canvas/video resolution sync.** The overlay canvas matches the video's
  intrinsic resolution each frame so normalized landmark coordinates line up
  with the pixels underneath.
- **Duplicate-frame skip.** Inference only runs when `video.currentTime`
  advances, since the display refreshes faster than the webcam delivers.

## Credits

Built in collaboration with Claude Code — the code was written by AI through
file operations, with the process documented for the video series.
