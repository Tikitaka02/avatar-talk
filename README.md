# avatar-talk

Browser-based motion capture that retargets your webcam movements onto a 3D
avatar — built step by step as a YouTube / blog series. Everything runs
client-side: no server, no installs, and no video ever leaves your machine.

**Current state — Phase 2:** real-time upper-body pose tracking with a skeleton
overlay ([MediaPipe](https://ai.google.dev/edge/mediapipe) PoseLandmarker, lite
model, GPU delegate) beside a 3D VRM avatar rendered with three.js and
[@pixiv/three-vrm](https://github.com/pixiv/three-vrm) — idle breathing,
expression controls, and drag-and-drop avatar swapping.

## Roadmap

| Phase | Episode | Tag | Status |
|-------|---------|-----|--------|
| 1 | Webcam + pose landmark overlay | `v1-landmarks` | ✅ done |
| 2 | Loading a 3D VRM avatar (three.js + @pixiv/three-vrm) | `v2-avatar` | ✅ done |
| 3 | Retargeting: landmarks → bone rotations | — | next |
| 4 | Smoothing, face expressions, hands | — | planned |

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
