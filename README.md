# avatar-talk

Browser-based motion capture that retargets your webcam movements onto a 3D
avatar — built step by step as a YouTube / blog series. Everything runs
client-side: no server, no installs, and no video ever leaves your machine.

**Current state — Phase 1:** real-time upper-body pose tracking with a
skeleton overlay, using [MediaPipe](https://ai.google.dev/edge/mediapipe)
PoseLandmarker (lite model, GPU delegate) in ~160 lines of TypeScript.

## Roadmap

| Phase | Episode | Tag | Status |
|-------|---------|-----|--------|
| 1 | Webcam + pose landmark overlay | `v1-landmarks` | ✅ done |
| 2 | Loading a 3D VRM avatar (three.js + @pixiv/three-vrm) | — | next |
| 3 | Retargeting: landmarks → bone rotations | — | planned |
| 4 | Smoothing, face expressions, hands | — | planned |

Each episode is a git tag, so you can check out exactly the code from any
video: `git checkout v1-landmarks`.

## Running it

```bash
npm install
npm run dev
```

Open http://localhost:5173 and allow camera access. You should see yourself
with a green/pink skeleton tracking your face, torso, and arms at ~30 fps.

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
