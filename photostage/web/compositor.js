"use strict";

const VERTEX_SHADER = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = vec2(aPos.x * 0.5 + 0.5, 0.5 - aPos.y * 0.5);
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

// The legacy photo rig draws the photo untouched except where the person moves:
// 1. Body: the photo is sampled at p - weight(p) * sway, a smooth warp whose weight (the server's body field) is 1 on the
//    head and shoulders and fades to 0 at the hips and towards the outline, so the person breathes and sways as one piece.
// 2. Hands: each cut-out hand turns about its wrist and slides: its own pixels are taken from where it has moved from, and
//    where it has left, a plate of the cloth or skin around it shows through (see hand() for the exact blend; the server
//    renders the same formula to judge a hand before it is allowed to move).
// 3. Head: the rectangle FlashHead animates, placed in the warped picture. FlashHead supplies only the motion and the coarse
//    colour: the photo's own fine detail is read from the position each pixel came from (the server's per-frame flow guide)
//    and added to FlashHead's low-passed frame. Where the guide says the warped photo does not explain the frame (open
//    mouth, blinks) FlashHead's own pixels are kept.
// While the idle loop and a live reply cross-fade, both renders are first moved onto the pose in between, using the same
// guides (each says where its pixels sit on the photo), so a head that is held a little differently in the two does not
// show twice.
// 4. Body clips (clips.js), when the avatar has them: a few seconds of video of the body from under the chin down, drawn over
//    the photo only where it differs from it, so that everything that does not move keeps the photo's own pixels. The head of
//    step 3 is then placed on the clip's neck and shoulders (see follow()) and covers the clip's own head. With no clip
//    playing (uClipMix is 0) none of this is evaluated and the picture is exactly the one of steps 1 to 3.
// 5. Full-avatar videos instead own the whole head and body. Only the facial interior is reenacted during speech, with
//    the renderer's global face pose removed and the video's tracked pose applied. The optional head mode instead masks
//    the video's head and reuses step 3, anchored at the collar, for both idle and speech.
// 6. Whole pictures (setPicture): a renderer that dubs the speech into the clips themselves (MuseTalk) streams the full
//    avatar, head and body in one image, which is drawn over the whole picture in the clips' tone (or as it is, for
//    pictures of the photo). Nothing is placed or fitted: the clips the player shows before and after are its own frames.
const FRAGMENT_COMMON = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uPhoto, uDetail, uBias, uSrcA, uGuideA, uSrcB, uGuideB, uField, uAlpha0, uPlate0, uAlpha1, uPlate1, uClip;
uniform vec2 uSize, uSway, uHands, uMatte, uNeck, uMouthCenter;
uniform vec4 uCrop, uFeather, uBox0, uBox1, uPose0, uPose1, uMove0, uMove1, uClipBox, uClipEdge, uFollow, uHeadMap, uFace, uMouth;
uniform vec3 uClipBias;
uniform mat3 uClipTone, uLiveFace, uVideoFace;
uniform float uMix, uHead, uCubic, uDetailOn, uLod, uMorph, uClipMix, uClipLod, uJoin, uVideoBase, uFaceRegistered, uMouthOnly, uFullHead;
uniform float uFullPicture, uPictureTone;
in vec2 vUv;
out vec4 outColor;

const float FLASHHEAD = 512.0;
const float FLOW_LEVELS = 4.0;
const float BIAS_LEVELS = 8.0;
const float LOW_PASS_LOD = 1.5;
// How far (in clip pixels) the clip is compared with the photo around a pixel, so that a moving edge is covered with its surroundings.
const float MATTE_REACH = 2.0;
const vec2 CROSS[5] = vec2[5](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, -1.0));

vec3 catmullRom(sampler2D tex, vec2 uv) {
  vec2 size = vec2(textureSize(tex, 0));
  vec2 pos = uv * size;
  vec2 center = floor(pos - 0.5) + 0.5;
  vec2 f = pos - center;
  vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  vec2 w3 = f * f * (-0.5 + 0.5 * f);
  vec2 w12 = w1 + w2;
  vec2 p0 = (center - 1.0) / size;
  vec2 p3 = (center + 2.0) / size;
  vec2 p12 = (center + w2 / w12) / size;
  vec3 c = textureLod(tex, vec2(p0.x, p0.y), 0.0).rgb * (w0.x * w0.y);
  c += textureLod(tex, vec2(p12.x, p0.y), 0.0).rgb * (w12.x * w0.y);
  c += textureLod(tex, vec2(p3.x, p0.y), 0.0).rgb * (w3.x * w0.y);
  c += textureLod(tex, vec2(p0.x, p12.y), 0.0).rgb * (w0.x * w12.y);
  c += textureLod(tex, vec2(p12.x, p12.y), 0.0).rgb * (w12.x * w12.y);
  c += textureLod(tex, vec2(p3.x, p12.y), 0.0).rgb * (w3.x * w12.y);
  c += textureLod(tex, vec2(p0.x, p3.y), 0.0).rgb * (w0.x * w3.y);
  c += textureLod(tex, vec2(p12.x, p3.y), 0.0).rgb * (w12.x * w3.y);
  c += textureLod(tex, vec2(p3.x, p3.y), 0.0).rgb * (w3.x * w3.y);
  return clamp(c, 0.0, 1.0);
}

vec2 flowOf(vec4 g) {
  return (g.rg * 255.0 - 128.0) / (FLOW_LEVELS * FLASHHEAD);
}

vec3 head(sampler2D frame, sampler2D guide, vec2 q) {
  vec3 pixels = uCubic > 0.5 ? catmullRom(frame, q) : textureLod(frame, q, uLod).rgb;
  if (uDetailOn < 0.5) return pixels;
  vec4 g = textureLod(guide, q, 0.0);
  vec2 flow = flowOf(g);
  vec3 low = textureLod(frame, q, LOW_PASS_LOD).rgb;
  float fine = catmullRom(uDetail, q + flow).r - 128.0 / 255.0;
  return low + g.b * fine + (1.0 - g.b) * (pixels - low);
}

// Register the collar and patch edges to the reference pose before blending them. Alpha alone would draw the moving
// collar over the stationary one twice. The inner face keeps all of the renderer's motion and expressions.
vec2 joined(sampler2D guide, vec2 q, float keep) {
  vec2 at = q;
  for (int i = 0; i < 3; i++) at = q - (1.0 - keep) * flowOf(textureLod(guide, at, 0.0));
  return at;
}

// A hand turned about its wrist and slid, in photo pixels (p is where the output pixel sits in the picture before any
// hand moves). A hand's box holds its alpha and the plate behind it; pose is (wrist x, wrist y, axis x, axis y) and move is
// (ramp, turn, slide x, slide y). The motion builds up from the wrist over the ramp, so the wrist and the arm stay put.
//   moved   = where the hand is now, taken from the position it has moved from;
//   bare    = how much of the hand's old place it has left;
//   result  = moved * the hand's own pixels + (1 - moved) * (the picture, with the plate showing where it is bare).
// With no motion moved equals rest, bare is 0 and the result is the picture itself, exactly.
vec3 hand(vec3 color, vec2 p, vec2 dx, vec2 dy, sampler2D alphaTex, sampler2D plateTex, vec4 box, vec4 pose, vec4 move) {
  vec2 size = box.zw - box.xy;
  float reach = length(move.zw) + abs(move.y) * length(size) + 2.0;
  if (any(lessThan(p, box.xy - reach)) || any(greaterThan(p, box.zw + reach))) return color;
  float weight = smoothstep(0.0, move.x, dot(p - pose.xy, pose.zw));
  vec2 r = p - pose.xy - move.zw;
  float c = cos(move.y), s = sin(move.y);
  vec2 source = p + weight * (pose.xy + vec2(c * r.x + s * r.y, c * r.y - s * r.x) - p);
  vec2 scale = uSize / size;
  vec2 at = (p - box.xy) / size;
  float rest = textureGrad(alphaTex, at, dx * scale, dy * scale).r;
  float moved = textureGrad(alphaTex, (source - box.xy) / size, dx * scale, dy * scale).r;
  float bare = clamp((rest - moved) / max(rest, 1e-3), 0.0, 1.0);
  vec3 plate = textureGrad(plateTex, at, dx * scale, dy * scale).rgb;
  vec3 layer = textureGrad(uPhoto, source / uSize, dx, dy).rgb;
  return moved * layer + (1.0 - moved) * (color + bare * (plate - color));
}

// The clip's colours put right: the photo's own tone, as the fit of the clip's first frame to the photo made it.
vec3 clipTone(vec3 c) {
  return clamp(uClipTone * c + uClipBias, 0.0, 1.0);
}

// Where the head is drawn. It follows the clip's neck and shoulders, which the clip's track gives as a turn and a slide about
// a pivot below the chin (uHeadMap is the inverse: cos and sin over the scale, then the slide): wholly from the chin down,
// not at all at the top of the patch, so the head sits on the neck and the top of it stays on the photo.
vec2 follow(vec2 p) {
  float k = uClipMix * (uVideoBase > 0.5 ? 1.0 : smoothstep(uFollow.x, uFollow.y, p.y));
  vec2 d = p - uFollow.zw - uHeadMap.zw;
  vec2 back = uFollow.zw + vec2(uHeadMap.x * d.x + uHeadMap.y * d.y, uHeadMap.x * d.y - uHeadMap.y * d.x);
  return p + k * (back - p);
}

// 1 above the chin and 0 below it: the clip is not drawn where it would show through the head.
float aboveChin(float y) {
  float soft = 0.08 * (uFollow.y - uFollow.x);
  return 1.0 - smoothstep(uFollow.y - soft, uFollow.y + soft, y);
}

// The clip over the colour given. p0 is where the pixel sits on the photo before any warp (the clip carries its own motion).
// The clip is drawn where it differs from the photo (moved, from the largest difference at a few points around the pixel, the
// clip being compared with the photo at the clip's own sharpness); where it does not, the pixel stays the photo's own. It fades
// out at the edges of the box it covers, except at the picture's border, and by keep and the clip's weight.
vec3 clipOver(vec3 color, vec2 p0, float keep) {
  vec2 size = uClipBox.zw - uClipBox.xy;
  vec2 at = p0 - uClipBox.xy;
  if (any(lessThan(at, vec2(0.0))) || any(greaterThan(at, size))) return color;
  vec2 low = vec2(uClipEdge.x > 0.0 ? smoothstep(0.0, uClipEdge.x, at.x) : 1.0,
                  uClipEdge.y > 0.0 ? smoothstep(0.0, uClipEdge.y, at.y) : 1.0);
  vec2 high = vec2(uClipEdge.z > 0.0 ? smoothstep(0.0, uClipEdge.z, size.x - at.x) : 1.0,
                   uClipEdge.w > 0.0 ? smoothstep(0.0, uClipEdge.w, size.y - at.y) : 1.0);
  float amount = low.x * low.y * high.x * high.y * uClipMix * keep;
  if (amount <= 0.0) return color;
  vec2 uv = at / size;
  float moved = 1.0;
  if (uMatte.x >= 0.0) {
    vec2 texel = 1.0 / vec2(textureSize(uClip, 0));
    float most = 0.0;
    for (int i = 0; i < 5; i++) {
      vec2 o = CROSS[i] * MATTE_REACH * texel;
      vec3 seen = clipTone(textureLod(uClip, uv + o, 0.0).rgb);
      vec3 photo = textureLod(uPhoto, (p0 + o * size) / uSize, uClipLod).rgb;
      vec3 gap = abs(seen - photo);
      most = max(most, max(gap.r, max(gap.g, gap.b)));
    }
    moved = smoothstep(uMatte.x, uMatte.y, most);
  }
  if (moved <= 0.0) return color;
  return mix(color, clipTone(catmullRom(uClip, uv)), moved * amount);
}
`;

// Whole pictures do not need the head/hand warps. A separate shader keeps that work out of software GPU playback too.
const PICTURE_SHADER = FRAGMENT_COMMON + `
void main() {
  vec3 picture = vec3(0.0);
  if (uMix > 0.0) {
    picture = texture(uSrcB, vUv).rgb;
    if (uPictureTone > 0.5) picture = clipTone(picture);
    if (uMix >= 1.0) {
      outColor = vec4(picture, 1.0);
      return;
    }
  }
  vec3 under = texture(uPhoto, vUv).rgb;
  if (uClipMix > 0.0) {
    if (uVideoBase > 0.5 && all(equal(uClipBox, vec4(0.0, 0.0, uSize)))) {
      under = mix(under, clipTone(texture(uClip, vUv).rgb), uClipMix);
    } else {
      under = clipOver(under, vUv * uSize, 1.0);
    }
  }
  outColor = vec4(mix(under, picture, uMix), 1.0);
}`;

const FRAGMENT_SHADER = FRAGMENT_COMMON + `
void main() {
  vec2 dx = dFdx(vUv), dy = dFdy(vUv);
  vec2 p0 = vUv * uSize;
  vec2 p = p0 - texture(uField, vUv).r * uSway;
  vec3 base = textureGrad(uPhoto, p / uSize, dx, dy).rgb;
  if (uHands.x > 0.5) base = hand(base, p, dx, dy, uAlpha0, uPlate0, uBox0, uPose0, uMove0);
  if (uHands.y > 0.5) base = hand(base, p, dx, dy, uAlpha1, uPlate1, uBox1, uPose1, uMove1);
  bool clip = uClipMix > 0.0;
  if (uFullPicture > 0.5) {
    vec3 under = clip ? clipOver(base, p0, 1.0) : base;
    if (uMix <= 0.0) {
      outColor = vec4(under, 1.0);
      return;
    }
    vec3 picture = catmullRom(uSrcB, vUv);
    if (uPictureTone > 0.5) picture = clipTone(picture);
    outColor = vec4(mix(under, picture, uMix), 1.0);
    return;
  }
  vec2 h = clip ? follow(p) : p;
  vec2 q = (h - uCrop.xy) / (uCrop.zw - uCrop.xy);
  if ((uHead <= 0.0 && uFullHead < 0.5) || q.x < 0.0 || q.y < 0.0 || q.x > 1.0 || q.y > 1.0) {
    outColor = vec4(clip ? clipOver(base, p0, 1.0) : base, 1.0);
    return;
  }
  if (uVideoBase > 0.5 && uFullHead < 0.5) {
    vec3 video = clip ? clipOver(base, p0, 1.0) : base;
    if (uFaceRegistered > 0.5) {
      q = (uVideoFace * vec3((p0 - uCrop.xy) / (uCrop.zw - uCrop.xy), 1.0)).xy;
      h = uCrop.xy + q * (uCrop.zw - uCrop.xy);
    }
    vec2 center = (uFace.xy + uFace.zw) * 0.5;
    vec2 radius = (uFace.zw - uFace.xy) * 0.5;
    float mask = 1.0 - smoothstep(0.78, 1.0, length((h - center) / radius));
    if (uMouthOnly > 0.5) {
      vec2 offset = q - uMouthCenter;
      vec2 reach = vec2((uMouth.z - uMouth.x) * 0.5,
                       offset.y < 0.0 ? uMouthCenter.y - uMouth.y : uMouth.w - uMouthCenter.y);
      mask = 1.0 - smoothstep(0.78, 1.0, length(offset / reach));
    }
    float amount = mask * uHead * uMix;
    if (amount <= 0.0) {
      outColor = vec4(video, 1.0);
      return;
    }
    // The video's pose owns the head outline, hair, neck and shoulders. Remove only the live renderer's global face
    // pose; its non-rigid mouth, eyelids and expression remain. Idle uses the video's own face, including its blinks.
    vec2 at = (uLiveFace * vec3(q, 1.0)).xy;
    vec3 facial = head(uSrcB, uGuideB, at);
    facial += (textureLod(uBias, q, 0.0).rgb * 255.0 - 128.0) / (BIAS_LEVELS * 255.0);
    outColor = vec4(mix(video, clamp(facial, 0.0, 1.0), amount), 1.0);
    return;
  }
  vec2 inner = h - uCrop.xy, outer = uCrop.zw - h;
  float w = (uFeather.x > 0.0 ? smoothstep(0.0, uFeather.x, inner.x) : 1.0)
          * (uFeather.y > 0.0 ? smoothstep(0.0, uFeather.y, inner.y) : 1.0)
          * (uFeather.z > 0.0 ? smoothstep(0.0, uFeather.z, outer.x) : 1.0)
          * (uFeather.w > 0.0 ? smoothstep(0.0, uFeather.w, outer.y) : 1.0);
  float keep = mix(1.0, w * (1.0 - smoothstep(uNeck.x, uNeck.y, h.y)), uJoin);
  vec3 face;
  if (uMix <= 0.0) {
    face = head(uSrcA, uGuideA, joined(uGuideA, q, keep));
  } else if (uMix >= 1.0) {
    face = head(uSrcB, uGuideB, joined(uGuideB, q, keep));
  } else {
    // A pixel that sits at q in B sits shift further along in A: the flows both point at the photo.
    vec2 shift = uMorph * keep * (flowOf(textureLod(uGuideB, q, 0.0)) - flowOf(textureLod(uGuideA, q, 0.0)));
    face = mix(head(uSrcA, uGuideA, joined(uGuideA, q + uMix * shift, keep)),
               head(uSrcB, uGuideB, joined(uGuideB, q - (1.0 - uMix) * shift, keep)), uMix);
  }
  face += (textureLod(uBias, q, 0.0).rgb * 255.0 - 128.0) / (BIAS_LEVELS * 255.0);
  float cover = w * (uFullHead > 0.5 ? 1.0 : uHead);
  vec3 under = clip ? clipOver(base, p0, 1.0 - cover * aboveChin(h.y)) : base;
  outColor = vec4(mix(under, clamp(face, 0.0, 1.0), w * uHead), 1.0);
}`;

const MAX_CANVAS_SIDE = 1920;
const FLASHHEAD_SIZE = 512;
const MAX_HANDS = 2;
const SAMPLERS = {
  photo: "uPhoto", detail: "uDetail", bias: "uBias",
  idle: "uSrcA", idleGuide: "uGuideA", live: "uSrcB", liveGuide: "uGuideB", field: "uField",
  alpha0: "uAlpha0", plate0: "uPlate0", alpha1: "uAlpha1", plate1: "uPlate1", clip: "uClip",
};
const UNITS = Object.keys(SAMPLERS);
// Frames are mip-mapped for the low-pass and for downscaling; the data layers are not.
const MIPPED = new Set(["photo", "idle", "live"]);
// Zero flow, full trust, zero detail: what a layer holds when the avatar has none.
let neutral = null;
const neutralLayer = () => neutral || (neutral = new ImageData(new Uint8ClampedArray([128, 128, 255, 255]), 1, 1));
const clearLayer = () => new ImageData(new Uint8ClampedArray(4), 1, 1);

function faceAffine(points, weights = points.map(() => 1)) {
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  const mean = [0, 0, 0, 0];
  points.forEach((p, i) => {
    for (let k = 0; k < 4; k++) mean[k] += p[k] * weights[i] / total;
  });
  let xx = 0, xy = 0, yy = 0, ux = 0, uy = 0, vx = 0, vy = 0;
  points.forEach((p, i) => {
    const x = p[0] - mean[0], y = p[1] - mean[1], u = p[2] - mean[2], v = p[3] - mean[3], w = weights[i];
    xx += w*x*x; xy += w*x*y; yy += w*y*y; ux += w*u*x; uy += w*u*y; vx += w*v*x; vy += w*v*y;
  });
  const det = xx*yy - xy*xy;
  if (!(det > 1e-12)) return null;
  const a = (ux*yy-uy*xy)/det, b = (uy*xx-ux*xy)/det;
  const c = (vx*yy-vy*xy)/det, d = (vy*xx-vx*xy)/det;
  const fit = [a, b, mean[2]-a*mean[0]-b*mean[1], c, d, mean[3]-c*mean[0]-d*mean[1]];
  return fit.every(Number.isFinite) && a*d-b*c > 0.1 ? fit : null;
}

const faceResidual = (fit, p) => Math.hypot(fit[0]*p[0]+fit[1]*p[1]+fit[2]-p[2],
                                           fit[3]*p[0]+fit[4]*p[1]+fit[5]-p[3]);

// Use the video's fixed eye-corner/nose/cheek anchors on the live side too. Sampling a changing set of
// high-confidence pixels across the whole face lets expression and occlusion move the estimated head pose.
function anchoredFacePose(data, side, anchors) {
  const flow = (u, v) => {
    if (u < 0 || u > 1 || v < 0 || v > 1) return null;
    const x = Math.max(0, Math.min(side - 1, u * side - 0.5));
    const y = Math.max(0, Math.min(side - 1, v * side - 0.5));
    const x0 = Math.floor(x), y0 = Math.floor(y), x1 = Math.min(side - 1, x0 + 1), y1 = Math.min(side - 1, y0 + 1);
    const channel = (k) => {
      const top = data[(y0*side+x0)*4+k] * (1-(x-x0)) + data[(y0*side+x1)*4+k] * (x-x0);
      const bottom = data[(y1*side+x0)*4+k] * (1-(x-x0)) + data[(y1*side+x1)*4+k] * (x-x0);
      return (top * (1-(y-y0)) + bottom * (y-y0) - 128) / 2048;
    };
    return [channel(0), channel(1)];
  };
  const points = [];
  for (const [x, y] of anchors) {
    let u = x, v = y;
    // The guide maps rendered -> reference. Invert it at each fixed reference anchor.
    for (let i = 0; i < 8; i++) {
      const delta = flow(u, v);
      if (!delta) return null;
      u = x - delta[0]; v = y - delta[1];
    }
    points.push([x, y, u, v]);
  }
  const median = (values) => {
    const ordered = values.slice().sort((a, b) => a - b), half = Math.floor(ordered.length / 2);
    return ordered.length % 2 ? ordered[half] : (ordered[half - 1] + ordered[half]) / 2;
  };
  let weights = points.map(() => 1), fit = null;
  for (let pass = 0; pass < 6; pass++) {
    fit = faceAffine(points, weights);
    if (!fit) return null;
    const errors = points.map((p) => faceResidual(fit, p)), center = median(errors);
    const spread = Math.max(1.4826 * median(errors.map((e) => Math.abs(e - center))), 0.5 / FLASHHEAD_SIZE);
    weights = errors.map((e) => Math.min(1, 1.345 * spread / Math.max(e, 1e-12)));
  }
  return fit;
}

// Older packs without reference anchors retain their original dense-flow registration.
function fitFacePose(data, side, face, anchors = null) {
  if (anchors) return anchoredFacePose(data, side, anchors);
  let points = [];
  for (let y = 0; y < side; y += 3) for (let x = 0; x < side; x += 3) {
    const px = (x + 0.5) / side, py = (y + 0.5) / side, i = (y * side + x) * 4;
    if (px < face[0] || px > face[2] || py < face[1] || py > face[3] || data[i + 2] < 180) continue;
    points.push([px + (data[i] - 128) / 2048, py + (data[i + 1] - 128) / 2048, px, py]);
  }
  let fit = null;
  for (let pass = 0; pass < 3; pass++) {
    if (points.length < 8) return null;
    fit = faceAffine(points);
    if (!fit) return null;
    const error = (p) => faceResidual(fit, p);
    const errors = points.map(error).sort((a, b) => a - b);
    const limit = Math.max(1.5 / FLASHHEAD_SIZE, 3 * errors[Math.floor(errors.length / 2)]);
    points = points.filter((p) => error(p) <= limit);
  }
  return fit && fit.every(Number.isFinite) && fit[0]*fit[4]-fit[1]*fit[3] > 0.1 ? fit : null;
}

const FACE_IDENTITY = [1, 0, 0, 0, 1, 0];
function inverseFace(p) {
  const det = p[0]*p[4]-p[1]*p[3];
  if (!(det > 0.1)) throw new Error("The video face registration is singular.");
  return [p[4]/det, -p[1]/det, (p[1]*p[5]-p[4]*p[2])/det,
          -p[3]/det, p[0]/det, (p[3]*p[2]-p[0]*p[5])/det];
}

class Compositor {
  constructor(canvas, tuning = new URLSearchParams(), wholePictures = false) {
    this.canvas = canvas;
    this.wholePictures = wholePictures;
    this.tuning = tuning;
    this.morph = tuning.get("morph") !== "0";
    // The cross-fade can only follow the pose where both slots hold a real guide.
    this.guided = { idle: false, live: false };
    this.hands = [];
    this.size = [1, 1];
    this.clipSpec = null;
    this.clipSize = null;
    this.clipReady = false;
    this.picture = null;
    const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error("This browser does not support WebGL 2, which the avatar renderer needs.");
    this.gl = gl;
    const program = gl.createProgram();
    for (const [type, source] of [[gl.VERTEX_SHADER, VERTEX_SHADER],
      [gl.FRAGMENT_SHADER, wholePictures ? PICTURE_SHADER : FRAGMENT_SHADER]]) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      gl.attachShader(program, shader);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    gl.useProgram(program);
    this.program = program;
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const position = gl.getAttribLocation(program, "aPos");
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
    this.textures = {};
    UNITS.forEach((name, unit) => {
      this.textures[name] = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, this.textures[name]);
      // Where the body's warp asks for a pixel beyond the photo's edge (a person cut by the frame) the photo is mirrored
      // there, which reads as more of the same where a repeated border row would be a streak.
      const wrap = name === "photo" ? gl.MIRRORED_REPEAT : gl.CLAMP_TO_EDGE;
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER,
                       MIPPED.has(name) && !(wholePictures && name === "live") ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
      gl.uniform1i(gl.getUniformLocation(program, SAMPLERS[name]), unit);
    });
    const locations = new Map();
    this.uniform = (name) => {
      if (!locations.has(name)) locations.set(name, gl.getUniformLocation(program, name));
      return locations.get(name);
    };
  }

  upload(name, source) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + UNITS.indexOf(name));
    gl.bindTexture(gl.TEXTURE_2D, this.textures[name]);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    if (MIPPED.has(name) && !(this.wholePictures && name === "live")) gl.generateMipmap(gl.TEXTURE_2D);
  }

  // `body` is the avatar's body rig: {rig: body.json, field, hands: [{alpha, plate}]} with decoded images, or null for none.
  load({ photo, rig, ref, detail, bias, body }) {
    const gl = this.gl;
    const [width, height] = [photo.width, photo.height];
    const fit = Math.min(1, MAX_CANVAS_SIDE / Math.max(width, height));
    this.canvas.width = Math.round(width * fit);
    this.canvas.height = Math.round(height * fit);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    this.size = [width, height];
    this.upload("photo", photo);
    this.upload("bias", bias);
    this.upload("detail", detail || neutralLayer());
    // The clip's texture is complete from the start, and no clip is drawn until its first frame is in.
    this.upload("clip", clearLayer());
    this.clipSpec = null;
    this.clipSize = null;
    this.clipReady = false;
    // Until real frames arrive both slots show the neutral reference.
    for (const slot of ["idle", "live"]) this.setFrame(slot, ref, null);
    const [x1, y1, x2, y2] = rig.crop;
    const crop = Math.max(x2 - x1, y2 - y1);
    const feather = 0.12 * Math.min(x2 - x1, y2 - y1);
    const set = (name, ...values) => gl[`uniform${values.length}f`](this.uniform(name), ...values);
    set("uSize", width, height);
    set("uCrop", x1, y1, x2, y2);
    const face = rig.face || rig.crop;
    this.faceBounds = [(face[0] - x1) / (x2 - x1), (face[1] - y1) / (y2 - y1),
                       (face[2] - x1) / (x2 - x1), (face[3] - y1) / (y2 - y1)];
    set("uFace", ...face);
    this.setFaceMatrix("uLiveFace", FACE_IDENTITY);
    this.setFaceMatrix("uVideoFace", FACE_IDENTITY);
    set("uFaceRegistered", 0);
    set("uMouthOnly", 0);
    set("uFullHead", 0);
    set("uVideoBase", 0);
    this.setPicture(null);
    this.setClipTone();
    // Edges on the image border have nothing to blend into, so they stay hard.
    set("uFeather", x1 > 0 ? feather : 0, y1 > 0 ? feather : 0, x2 < width ? feather : 0, y2 < height ? feather : 0);
    const chin = rig.face ? rig.face[3] : y2;
    set("uNeck", chin, Math.max(chin + 1, y2 - feather * 0.25));
    set("uJoin", this.tuning.get("join") === "0" ? 0 : 1);
    set("uDetailOn", detail && rig.detail && this.tuning.get("detail") !== "0" ? 1 : 0);
    set("uCubic", crop > FLASHHEAD_SIZE * 1.05 ? 1 : 0);
    set("uLod", Math.max(0, Math.log2(FLASHHEAD_SIZE / crop)));
    this.loadBody(body);
  }

  loadBody(body) {
    const set = (name, ...values) => this.gl[`uniform${values.length}f`](this.uniform(name), ...values);
    const spec = body && body.rig;
    this.upload("field", body && body.field && spec.field ? body.field : clearLayer());
    this.hands = [];
    const on = [0, 0];
    for (let k = 0; k < MAX_HANDS; k++) {
      const hand = spec && spec.hands && spec.hands[k];
      const layers = body && body.hands && body.hands[k];
      const usable = Boolean(hand && layers);
      this.upload(`alpha${k}`, usable ? layers.alpha : clearLayer());
      this.upload(`plate${k}`, usable ? layers.plate : clearLayer());
      if (!usable) continue;
      on[k] = 1;
      set(`uBox${k}`, ...hand.box);
      set(`uPose${k}`, hand.pivot[0], hand.pivot[1], hand.axis[0], hand.axis[1]);
      this.hands.push({ index: k, ramp: hand.ramp });
    }
    set("uHands", on[0], on[1]);
    this.setPose(null);
  }

  setFrame(slot, image, guide) {
    if (this.wholePictures && slot === "idle") return;
    if (this.wholePictures && slot === "live" && this.picture) this.fitPicture(image.width, image.height);
    this.upload(slot, image);
    this.upload(`${slot}Guide`, guide || neutralLayer());
    this.guided[slot] = Boolean(guide);
    if (slot === "live" && !this.picture && this.clipSpec && this.clipSpec.mode === "avatar" && this.clipSpec.faceMode !== "head") {
      let pose = null;
      if (guide) {
        if (!this.poseReader) {
          const canvas = document.createElement("canvas");
          canvas.width = canvas.height = 128;
          this.poseReader = canvas.getContext("2d", { willReadFrequently: true });
        }
        this.poseReader.drawImage(guide, 0, 0, 128, 128);
        pose = fitFacePose(this.poseReader.getImageData(0, 0, 128, 128).data, 128,
                           this.faceBounds, this.clipSpec.faceAnchors);
      }
      if (!pose && !this.poseWarning) console.warn("The live face pose could not be tracked; holding its last registered pose.");
      this.poseWarning = !pose;
      if (pose) this.setFaceMatrix("uLiveFace", pose);
    }
  }

  setFaceMatrix(name, p) {
    this.gl.uniformMatrix3fv(this.uniform(name), false, [p[0],p[3],0,p[1],p[4],0,p[2],p[5],1]);
  }

  // What the live frames are: null for a head to place (FlashHead, LiveTalk), "clips" for whole pictures made from the clips
  // (drawn in their tone), "photo" for whole pictures of the photo.
  setPicture(kind) {
    if (kind !== null && kind !== "clips" && kind !== "photo") throw new Error(`Unknown live picture: ${kind}`);
    this.picture = kind;
    this.gl.uniform1f(this.uniform("uFullPicture"), kind ? 1 : 0);
    this.gl.uniform1f(this.uniform("uPictureTone"), kind === "clips" ? 1 : 0);
  }

  // A whole picture already contains the final face. GPU upscaling to the photo's resolution adds no facial detail.
  fitPicture(width, height) {
    if (!(width > 0 && height > 0)) return;
    const scale = Math.min(1, width / this.size[0], height / this.size[1],
                           MAX_CANVAS_SIDE / Math.max(...this.size));
    const w = Math.round(this.size[0] * scale), h = Math.round(this.size[1] * scale);
    if (this.canvas.width === w && this.canvas.height === h) return;
    this.canvas.width = w;
    this.canvas.height = h;
    this.gl.viewport(0, 0, w, h);
  }

  // What the body does now, in photo pixels and radians: {sway: [x, y], hands: [{angle, x, y}, ...]}. Nothing is at rest.
  setPose(pose) {
    const gl = this.gl;
    const sway = pose && pose.sway || [0, 0];
    gl.uniform2f(this.uniform("uSway"), sway[0], sway[1]);
    for (const { index, ramp } of this.hands) {
      const move = pose && pose.hands && pose.hands[index] || { angle: 0, x: 0, y: 0 };
      gl.uniform4f(this.uniform(`uMove${index}`), ramp, move.angle, move.x, move.y);
    }
  }

  // The avatar's body clips (see clips.js): `spec` is the pack's manifest, whose box is the part of the photo they show, and
  // whose follow and pivot place the head on the clip's neck. Nothing is drawn until setClipFrame has given a first frame.
  loadClips(spec) {
    const gl = this.gl;
    const [width, height] = this.size;
    const [x1, y1, x2, y2] = spec.box;
    // The box fades out at its edges, except where the edge is the picture's own border.
    const feather = 0.05 * Math.min(x2 - x1, y2 - y1);
    const edge = (gap, extent) => (gap > 0.015 * extent ? feather : 0);
    gl.uniform4f(this.uniform("uClipBox"), x1, y1, x2, y2);
    gl.uniform4f(this.uniform("uClipEdge"), edge(x1, width), edge(y1, height), edge(width - x2, width), edge(height - y2, height));
    gl.uniform4f(this.uniform("uFollow"), spec.follow[0], spec.follow[1], spec.pivot[0], spec.pivot[1]);
    gl.uniform1f(this.uniform("uVideoBase"), spec.mode === "avatar" ? 1 : 0);
    gl.uniform1f(this.uniform("uFaceRegistered"), spec.mode === "avatar" && spec.faceMode !== "head" && spec.facePose ? 1 : 0);
    gl.uniform1f(this.uniform("uMouthOnly"), spec.faceMode === "mouth" ? 1 : 0);
    gl.uniform1f(this.uniform("uFullHead"), spec.faceMode === "head" ? 1 : 0);
    if (spec.faceMode === "mouth") {
      gl.uniform4f(this.uniform("uMouth"), ...spec.mouth.box);
      gl.uniform2f(this.uniform("uMouthCenter"), ...spec.mouth.center);
    }
    // ?matte=off draws the clip whole, ?matte=0.04,0.1 sets where the clip starts and ends to replace the photo.
    const matte = this.tuning.get("matte");
    const [low, high] = matte === "off" || spec.mode === "avatar" ? [-1, -1] : (matte || "0.05,0.12").split(",").map(Number);
    gl.uniform2f(this.uniform("uMatte"), Number.isFinite(low) ? low : 0.05, Number.isFinite(high) ? high : 0.12);
    this.clipSpec = spec;
    this.clipSize = null;
    this.clipReady = false;
    this.setClipTone();
    this.setClip(null);
  }

  // The next picture of the playing clip: a video or a canvas.
  setClipFrame(source) {
    const width = source.displayWidth || source.videoWidth || source.width;
    const height = source.displayHeight || source.videoHeight || source.height;
    if (!this.clipSpec || !(width > 0 && height > 0)) return;
    if (this.wholePictures && this.clipSpec.mode === "avatar") this.fitPicture(width, height);
    this.upload("clip", source);
    if (!this.clipSize || this.clipSize[0] !== width || this.clipSize[1] !== height) {
      this.clipSize = [width, height];
      // The clip is compared with the photo as sharp as the clip is: the photo's mip level that has the clip's pixel size.
      const box = this.clipSpec.box;
      this.gl.uniform1f(this.uniform("uClipLod"), Math.max(0, Math.log2((box[2] - box[0]) / width)));
    }
    this.clipReady = true;
  }

  // The colours that turn the clip into the photo's: photo = matrix (3x3, by rows) * clip + offset; none is no change.
  setClipTone(matrix = [1, 0, 0, 0, 1, 0, 0, 0, 1], offset = [0, 0, 0]) {
    const columns = new Float32Array(9);
    for (let row = 0; row < 3; row++) for (let column = 0; column < 3; column++) columns[column * 3 + row] = matrix[row * 3 + column];
    this.gl.uniformMatrix3fv(this.uniform("uClipTone"), false, columns);
    this.gl.uniform3f(this.uniform("uClipBias"), offset[0], offset[1], offset[2]);
  }

  // How much of the clip is drawn (0-1) and where its neck and shoulders are now against the photo: {scale, turn (radians),
  // x, y (photo pixels)}, which the head follows. Zero draws no clip, and the picture is then the body's own.
  setClip(state) {
    const mix = state && this.clipReady && state.mix > 0 ? Math.min(1, state.mix) : 0;
    this.gl.uniform1f(this.uniform("uClipMix"), mix);
    if (this.clipSpec && this.clipSpec.faceMode !== "head" && this.clipSpec.facePose) {
      const home = this.clipSpec.facePose, pose = state && state.face || home;
      this.setFaceMatrix("uVideoFace", inverseFace(home.map((v, i) => v + mix * (pose[i] - v))));
    }
    if (!mix) return;
    const head = state.head || { scale: 1, turn: 0, x: 0, y: 0 };
    const scale = head.scale > 0 ? head.scale : 1;
    this.gl.uniform4f(this.uniform("uHeadMap"), Math.cos(head.turn) / scale, Math.sin(head.turn) / scale, head.x, head.y);
  }

  clearClips() {
    this.setClip(null);
    this.gl.uniform1f(this.uniform("uVideoBase"), 0);
    this.gl.uniform1f(this.uniform("uFaceRegistered"), 0);
    this.gl.uniform1f(this.uniform("uMouthOnly"), 0);
    this.gl.uniform1f(this.uniform("uFullHead"), 0);
    this.setClipTone();
    this.clipSpec = null;
    this.clipReady = false;
  }

  draw(mix, head) {
    this.gl.uniform1f(this.uniform("uMix"), mix);
    this.gl.uniform1f(this.uniform("uHead"), head);
    this.gl.uniform1f(this.uniform("uMorph"), this.morph && this.guided.idle && this.guided.live ? 1 : 0);
    this.gl.drawArrays(this.gl.TRIANGLES, 0, 3);
  }
}
