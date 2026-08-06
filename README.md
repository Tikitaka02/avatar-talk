# avatar-talk

Browser-based motion capture that retargets your webcam movements onto a 3D
avatar — built step by step as a YouTube / blog series. Everything runs
client-side: no server, no installs, and no video ever leaves your machine.

**Current state — Phase 4:** one [MediaPipe](https://ai.google.dev/edge/mediapipe)
HolisticLandmarker — body, face and hands in a single inference — drives a VRM
rig rendered with three.js and
[@pixiv/three-vrm](https://github.com/pixiv/three-vrm). Arms, torso and head
follow your body; your expressions drive the avatar's face; your fingers curl
its fingers and your wrists rotate its wrists. Landmarks are filtered with a
One Euro filter before retargeting.

## Roadmap

| Phase | Episode | Tag | Status |
|-------|---------|-----|--------|
| 1 | Webcam + pose landmark overlay | `v1-landmarks` | ✅ done |
| 2 | Loading a 3D VRM avatar (three.js + @pixiv/three-vrm) | `v2-avatar` | ✅ done |
| 3 | Retargeting: landmarks → bone rotations | `v3-retargeting` | ✅ done |
| 4 | One Euro filtering, face expressions, finger tracking | `v4-face-hands` | ✅ done |

Each episode is a git tag, so you can check out exactly the code from any
video: `git checkout v1-landmarks`.

## Running it

```bash
npm install
npm run dev
```

Open http://localhost:5173 and allow camera access. On the left you get your
webcam with a green/pink skeleton over your face, torso and arms, plus 21 points
on each hand; on the right, an avatar doing the same thing. Raise a hand and the avatar raises
the hand on the same side of the screen — it behaves like a mirror. Blink,
smile or open your mouth and its face follows; curl your fingers and its
fingers curl. Drag on the 3D pane to orbit the camera, tick **tracking readout**
to see the blendshapes and the VRM expressions they drive, and hold an
expression button to override the face by hand.

**Bring your own avatar.** No `.vrm` is committed (they are large, and their
licences are the author's), so the 3D pane starts empty. Drag any VRM file
onto it — free ones are available on [VRoid Hub](https://hub.vroid.com/), or
make your own in VRoid Studio. To have one load automatically, drop it in
`public/avatar/` and point `DEFAULT_VRM` in `src/main.ts` at it.

Needs a browser with WebGL2 and camera access on `localhost` or HTTPS. The pose
model and its WebAssembly runtime are fetched from a CDN on first load, so that
much needs a connection — the video itself never leaves the machine.

## The code

| File | What it does |
|------|--------------|
| [`src/tracking.ts`](src/tracking.ts) | Webcam capture and one HolisticLandmarker inference returning body, face and hands together, plus the overlay drawing. |
| [`src/space.ts`](src/space.ts) | Shared MediaPipe→VRM coordinate conversion and bone-chain maths. |
| [`src/hands.ts`](src/hands.ts) | Finger curl and wrist rotation from the hand landmarks. |
| [`src/filter.ts`](src/filter.ts) | One Euro filter, plus banks of them for landmark streams and named scalars. |
| [`src/expression.ts`](src/expression.ts) | Blendshapes → VRM expressions. |
| [`src/retarget.ts`](src/retarget.ts) | Landmarks → bone rotations: coordinate conversion, mirroring, parent-space solve. |
| [`src/avatar.ts`](src/avatar.ts) | three.js scene, VRM loading, auto-framing, rest pose, idle breathing. |
| [`src/main.ts`](src/main.ts) | Wires them together in one render loop. |

## Design notes (phase 4)

- **Filter the input, not the output.** Phase 3 smoothed bone rotations at the
  end of the pipeline; this filters landmarks at the start. Jitter is a property
  of the measurement, so cleaning it at the source means every angle derived
  afterwards inherits the fix. Measured on identical footage, frame-to-frame
  change in the avatar pane fell from 2.67 to 1.92 — about 28% less movement.
- **One Euro over a plain low-pass.** A fixed cutoff forces a choice between
  jitter at rest and lag in motion. One Euro raises its own cutoff with speed,
  so it filters hard when you hold still and gets out of the way when you move.
- **Smiling makes you blink.** Raised cheeks push the eye-blink blendshapes up,
  so a grinning avatar blinks nonstop. Subtracting `cheekSquint` fixes it.
- **Bend only, no splay.** Finger curl is the angle between the two segments
  meeting at each joint. Sideways splay is discarded: far noisier than bend, and
  nearly invisible on a stylised hand.
- **Hands come pre-labelled.** Holistic returns `leftHandLandmarks` and
  `rightHandLandmarks` separately, so nothing has to infer which hand is which.
  The person's right drives the avatar's left, matching the mirrored body.
- **Wrist rotation needs the palm.** A shoulder-to-elbow direction can never
  recover roll. Two vectors across the hand — along the fingers and across the
  knuckles — define its orientation, and the rotation carrying the rest pose's
  pair onto the measured pair is the wrist. It is damped and capped, because
  palm facing is mostly a depth signal and depth is the weakest axis.
- **Blendshapes force the CPU delegate.** Holistic's blendshape sub-model uses
  ops the GPU delegate does not implement (`DEQUANTIZE`, `STRIDED_SLICE`); ask
  for face expressions on GPU and the graph fails to open, which surfaces only
  as tracking that never starts.

## Measured cost

Numbers from this machine, headless Chromium, 960×669 clip, so treat them as
relative rather than absolute:

| Setup | Throughput |
|-------|-----------|
| Three separate landmarkers (pose + face + hands), GPU | ~9 fps |
| HolisticLandmarker, GPU, no blendshapes | ~8 fps |
| HolisticLandmarker, CPU, with blendshapes (current) | ~7 fps |

Holistic is not the speed win its shared detection stage suggests — it is
slightly slower here. What it buys is one model download instead of three,
one inference call, and hands labelled by the model. Full-body tracking with
expressions is simply expensive in a browser; none of these reach 30 fps.

## Known limitations

- **Twist only at the wrist.** The hand solves its full orientation from the
  palm, so wrist roll works. The upper arm and forearm still come from a single
  direction each, which says nothing about roll, so rotating your forearm
  without moving your hand does not turn the avatar's arm.
- **It is not fast.** Full-body tracking with expressions costs roughly 7 fps
  here (see Measured cost). Dropping face blendshapes or hands buys some back.
- **Expressions depend on the avatar.** Only the VRM expressions a model
  actually defines can be driven; the mapping skips whatever is missing.
- **Depth stays approximate** even with `Z_TRUST` — motion toward and away from
  the camera is the weakest axis, and always will be with one lens.
- **Upper body only**, by design: legs are unreliable from a desk webcam.

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
