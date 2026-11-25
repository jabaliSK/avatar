"""Renders the real compositor in headless Chromium and prints how it cross-faded two synthetic head frames, and what it
drew of a body clip (see CLIP_SCRIPT).

The frames hold the same white bar 16 px apart, and their flow guides say so. The tests run it with the first python3
on the PATH. It exits with SKIP_STATUS when that interpreter cannot import Playwright or launch Chromium, so that the
tests skip instead of fail.
"""
import json
import sys
from pathlib import Path

SKIP_STATUS = 3
COMPOSITOR = Path(__file__).resolve().parents[1] / "photostage/web/compositor.js"

SCRIPT = """() => {
  const SIZE = 512, MIXES = [0, 0.25, 0.5, 0.75, 1];
  const paint = (width, height, fill) => {
    const canvas = document.createElement("canvas");
    canvas.width = width; canvas.height = height;
    fill(canvas.getContext("2d"), width, height);
    return canvas;
  };
  const flat = (width, height, color) => paint(width, height, (g, w, h) => { g.fillStyle = color; g.fillRect(0, 0, w, h); });
  const bar = (left) => paint(SIZE, SIZE, (g) => {
    g.fillStyle = "#000"; g.fillRect(0, 0, SIZE, SIZE);
    g.fillStyle = "#fff"; g.fillRect(left, 0, 32, SIZE);
  });
  // A guide stores the flow as 128 + 4 * pixels of the 512 px frame, and the confidence in blue.
  const guide = (flow) => flat(128, 128, `rgb(${128 + 4 * flow}, 128, 255)`);
  const reader = paint(SIZE, SIZE, () => {}).getContext("2d", { willReadFrequently: true });
  const build = (query) => {
    const target = document.createElement("canvas");
    const compositor = new Compositor(target, new URLSearchParams(query));
    compositor.load({
      photo: flat(SIZE, SIZE, "#808080"), rig: { crop: [0, 0, SIZE, SIZE], detail: null }, ref: bar(0), detail: null,
      bias: flat(1, 1, "rgb(128, 128, 128)"),
    });
    return { compositor, target };
  };
  // Where the whole bar is, and how many pixels are neither bar nor background: a head drawn twice is half bright.
  const summary = (row) => {
    let left = -1, right = -1, ghost = 0;
    for (let x = 0; x < SIZE; x++) {
      const value = row[x * 4];
      if (value >= 224) { if (left < 0) left = x; right = x; } else if (value > 32) ghost++;
    }
    return { left, right, ghost };
  };
  // The photo has the bar at 200. A frame that holds it elsewhere has a guide that says how far to move it back, and a
  // frame without a guide is taken to be where the photo is.
  const frame = (left, flow) => ({ left, guide: flow === null ? null : guide(flow) });
  const home = frame(200, 0), away = frame(216, -16);
  const fade = ({ compositor, target }, mix, idle, live) => {
    compositor.setFrame("idle", bar(idle.left), idle.guide);
    compositor.setFrame("live", bar(live.left), live.guide);
    compositor.draw(mix, 1, 0);
    reader.drawImage(target, 0, 0);
    return summary(reader.getImageData(0, SIZE / 2, SIZE, 1).data);
  };
  const on = build(""), off = build("morph=0");
  const result = { aligned: {} };
  for (const mix of MIXES) result.aligned[mix] = fade(on, mix, home, away);
  result.switchedOff = fade(off, 0.5, home, away);
  result.unguidedIdle = fade(on, 0.5, frame(200, null), away);
  result.unguidedLive = fade(on, 0.5, away, frame(200, null));
  return result;
}"""

# The clip layer. The photo is grey (128) with a patch of 1 px checks; the head patch (crop 128,48 to 384,304, feathered
# 30.72 px) holds an orange face with a white line down its middle; the clip covers the photo from y 150 down to the bottom
# edge (its top edge fades over 25.6 px, the others are the picture's own border) at half the photo's resolution, with
# the chin at y 250 and the neck's pivot at (256, 420).
CLIP_SCRIPT = """() => {
  const W = 512, H = 768;
  const paint = (width, height, fill) => {
    const canvas = document.createElement("canvas");
    canvas.width = width; canvas.height = height;
    fill(canvas.getContext("2d"), width, height);
    return canvas;
  };
  const flat = (width, height, color) => paint(width, height, (g, w, h) => { g.fillStyle = color; g.fillRect(0, 0, w, h); });
  const photo = paint(W, H, (g) => {
    g.fillStyle = "#808080"; g.fillRect(0, 0, W, H);
    const checks = g.createImageData(192, 120);
    for (let y = 0; y < 120; y++) for (let x = 0; x < 192; x++) {
      const at = (y * 192 + x) * 4, value = (x + y) & 1 ? 255 : 0;
      checks.data[at] = checks.data[at + 1] = checks.data[at + 2] = value; checks.data[at + 3] = 255;
    }
    g.putImageData(checks, 32, 400);
  });
  const face = paint(512, 512, (g) => {
    g.fillStyle = "rgb(224, 128, 64)"; g.fillRect(0, 0, 512, 512);
    g.fillStyle = "#fff"; g.fillRect(254, 0, 4, 512);
  });
  const clips = {
    grey: flat(256, 309, "rgb(128, 128, 128)"),
    red: flat(256, 309, "rgb(255, 0, 0)"),
    square: paint(256, 309, (g) => { g.fillStyle = "rgb(128, 128, 128)"; g.fillRect(0, 0, 256, 309);
      g.fillStyle = "rgb(255, 0, 0)"; g.fillRect(150, 150, 50, 50); }),
    colour: flat(256, 309, "rgb(50, 100, 200)"),
  };
  const reader = paint(W, H, () => {}).getContext("2d", { willReadFrequently: true });
  const build = ({ clip = null, query = "", mix = 1, head = null, tone = null }) => {
    const target = document.createElement("canvas");
    const compositor = new Compositor(target, new URLSearchParams(query));
    compositor.load({ photo, rig: { crop: [128, 48, 384, 304], detail: null }, ref: face, detail: null,
                      bias: flat(1, 1, "rgb(128, 128, 128)") });
    if (clip) {
      compositor.loadClips({ box: [0, 150, 512, 768], follow: [48, 250], pivot: [256, 420] });
      if (tone) compositor.setClipTone(tone.matrix, tone.offset);
      compositor.setClipFrame(clips[clip]);
      compositor.setClip({ mix, head });
    }
    compositor.draw(0, 1);
    reader.drawImage(target, 0, 0);
    const data = reader.getImageData(0, 0, W, H).data;
    compositor.gl.getExtension("WEBGL_lose_context").loseContext();
    return { data, at: (x, y) => Array.from(data.slice((y * W + x) * 4, (y * W + x) * 4 + 3)) };
  };
  // Where the face's white line is on a row: the green of the face and of the photo is 128 and the line's is 255.
  const line = ({ data }, y) => {
    let total = 0, moment = 0;
    for (let x = 64; x < 448; x++) {
      const weight = Math.min(1, Math.max(0, (data[(y * W + x) * 4 + 1] - 128) / 127));
      total += weight; moment += weight * (x + 0.5);
    }
    return moment / total;
  };
  const checkered = ({ data }) => {
    let hard = 0, count = 0;
    for (let y = 410; y < 500; y++) for (let x = 40; x < 200; x++) {
      count++;
      if (Math.abs(data[(y * W + x) * 4] - 128) > 100) hard++;
    }
    return hard / count;
  };
  const same = (a, b) => { let n = 0; for (let i = 0; i < a.data.length; i++) if (a.data[i] !== b.data[i]) n++; return n; };
  const plain = build({}), red = build({ clip: "red" });
  const result = {
    // No clip drawn: the picture is the photo's and the head's, whatever the clip holds.
    off: same(plain, build({ clip: "red", mix: 0 })),
    far: red.at(450, 650), above: red.at(450, 100), border: [red.at(0, 650), red.at(450, 767), red.at(511, 650)],
    top: red.at(450, 162), face: red.at(300, 200), sideBand: red.at(140, 200), bottomBand: red.at(300, 290), below: red.at(256, 330),
    photoFar: plain.at(450, 650),
  };
  const square = build({ clip: "square" });
  result.square = { inside: square.at(350, 500), outside: square.at(350, 600), half: build({ clip: "square", mix: 0.5 }).at(350, 500) };
  result.sharp = { kept: checkered(build({ clip: "grey" })), whole: checkered(build({ clip: "grey", query: "matte=off" })),
                   photo: checkered(plain) };
  const moved = (head) => { const picture = build({ clip: "grey", head }); return { top: line(picture, 70), chin: line(picture, 260) }; };
  result.follow = { rest: moved(null), slid: moved({ scale: 1, turn: 0, x: 20, y: 0 }), turned: moved({ scale: 1, turn: 0.1, x: 0, y: 0 }),
                    half: (() => { const picture = build({ clip: "grey", mix: 0.5, head: { scale: 1, turn: 0, x: 20, y: 0 } });
                                   return { chin: line(picture, 260) }; })() };
  result.tone = build({ clip: "colour", tone: { matrix: [0, 1, 0, 0, 0, 1, 1, 0, 0], offset: [0.1, 0, 0] } }).at(450, 650);
  return result;
}"""

JOIN_SCRIPT = """() => {
  const W = 768, C = 512;
  const paint = (size, fill) => {
    const c = document.createElement("canvas"); c.width = c.height = size;
    fill(c.getContext("2d")); return c;
  };
  const photo = paint(W, (g) => {
    for (let x = 0; x < W; x += 16) {
      g.fillStyle = x % 32 ? "#fff" : "#000"; g.fillRect(x, 0, 16, W);
    }
  });
  const frame = (shift) => paint(C, (g) => g.drawImage(photo, 128 - shift, 128, C, C, 0, 0, C, C));
  const guide = (shift) => paint(128, (g) => {
    g.fillStyle = `rgb(${128 - 4 * shift}, 128, 255)`; g.fillRect(0, 0, 128, 128);
  });
  const bias = paint(1, (g) => { g.fillStyle = "rgb(128,128,128)"; g.fillRect(0, 0, 1, 1); });
  const reader = paint(W, () => {}).getContext("2d", {willReadFrequently: true});
  const render = (query, mix, guided = true) => {
    const target = document.createElement("canvas"), c = new Compositor(target, new URLSearchParams(query));
    c.load({photo, rig: {crop: [128,128,640,640], face: [256,256,512,512], detail: null}, ref: frame(0), bias, detail: null});
    c.setFrame("idle", frame(0), guided ? guide(0) : null);
    c.setFrame("live", frame(16), guided ? guide(16) : null);
    c.draw(mix, 1); reader.drawImage(target, 0, 0);
    const data = reader.getImageData(0, 0, W, W).data;
    c.gl.getExtension("WEBGL_lose_context").loseContext(); return data;
  };
  const error = (a, b, box) => {
    let sum = 0, count = 0;
    for (let y = box[1]; y < box[3]; y++) for (let x = box[0]; x < box[2]; x++) {
      sum += Math.abs(a[(y*W+x)*4] - b[(y*W+x)*4]); count++;
    }
    return sum / count;
  };
  const reference = render("", 0), result = [];
  for (const mix of [0.5, 1]) {
    const on = render("", mix), off = render("join=0", mix);
    result.push({mix, collar: error(on, reference, [220,628,548,635]),
      unjoinedCollar: error(off, reference, [220,628,548,635]),
      face: error(on, off, [256,256,512,480]), outside: error(on, reference, [0,0,100,W])});
  }
  return {frames: result, unguided: error(render("", 1, false), render("join=0", 1, false), [0,0,W,W])};
}"""

AVATAR_SCRIPT = """() => {
  const W = 512, H = 768;
  const anchors = [[.3,.35],[.4,.35],[.6,.35],[.7,.35],[.5,.4],[.5,.43],[.5,.46],
                   [.5,.49],[.5,.52],[.5,.55],[.5,.58],[.26,.5],[.74,.5]];
  const paint = (w, h, fill) => {
    const c = document.createElement("canvas"); c.width=w; c.height=h;
    fill(c.getContext("2d")); return c;
  };
  const flat = (w,h,color) => paint(w,h,g => {g.fillStyle=color;g.fillRect(0,0,w,h);});
  const video = flat(W,H,"rgb(50,100,150)");
  const live = (shift) => paint(512,512,g => {
    g.fillStyle="rgb(224,128,64)";g.fillRect(0,0,512,512);
    g.fillStyle="#fff";g.fillRect(254+shift,0,4,512);
  });
  const guide = (shift) => flat(128,128,`rgb(${128-4*shift},128,255)`);
  const reader = paint(W,H,()=>{}).getContext("2d",{willReadFrequently:true});
  const render = (mix, shift, head, clear = false, registered = false, mouthOnly = false, anchored = false, wholeHead = false) => {
    const target=document.createElement("canvas"), c=new Compositor(target);
    c.load({photo:flat(W,H,"#333"),rig:{crop:[128,48,384,304],face:[180,100,332,250],detail:null},
      ref:live(0),detail:null,bias:flat(1,1,"rgb(128,128,128)")});
    c.loadClips({mode:"avatar",box:[0,0,W,H],follow:[100,250],pivot:[256,175],
      facePose:registered ? [1,0,0.02,0,1,-0.01] : null,
      faceAnchors:anchored ? anchors : null,
      faceMode:wholeHead ? "head" : mouthOnly ? "mouth" : "full",mouth:{box:[0.36,0.55,0.64,0.8],center:[0.5,0.65]}});
    const snapshot = anchored && typeof VideoFrame === "function" ? new VideoFrame(video,{timestamp:0}) : video;
    c.setClipFrame(snapshot);
    if (snapshot !== video) snapshot.close();
    c.setClip({mix:1,head,face:registered ? [0.9,0.05,0.07,0.02,1,-0.01] : null});
    c.setFrame("live",live(shift),guide(shift));
    if (clear) c.clearClips();
    c.draw(mix,1);
    reader.drawImage(target,0,0);
    const data=reader.getImageData(0,0,W,H).data;
    const at=(x,y)=>Array.from(data.slice((y*W+x)*4,(y*W+x)*4+3));
    let total=0,moment=0;
    for(let x=200;x<320;x++) {
      const weight=Math.max(0,data[(175*W+x)*4+1]-128);
      total+=weight;moment+=weight*(x+0.5);
    }
    c.gl.getExtension("WEBGL_lose_context").loseContext();
    return {face:at(280,175),eyes:at(236,150),cheek:at(196,185),mouth:at(270,218),
      hair:at(240,80),neck:at(256,280),body:at(256,500),background:at(50,175),
      line:total ? moment/total : null};
  };
  const expected = [0.96,0.06,0.01,-0.025,1.02,-0.01];
  const data = new Uint8ClampedArray(128*128*4), [a,b,tx,c,d,ty] = expected, norm=a*d-b*c;
  for(let y=0;y<128;y++) for(let x=0;x<128;x++) {
    const px=(x+0.5)/128,py=(y+0.5)/128,u=px-tx,v=py-ty,i=(y*128+x)*4;
    data[i]=128+2048*((d*u-b*v)/norm-px);
    data[i+1]=128+2048*((a*v-c*u)/norm-py);
    data[i+2]=255;data[i+3]=255;
    if(x>50&&x<80&&y>75&&y<95) {data[i]+=32;data[i+1]-=32;}
  }
  const changingExpressions = [];
  for (const shift of [-12, 0, 12]) {
    const changing = data.slice();
    for (let y=0; y<128; y++) for (let x=0; x<128; x++) {
      if (y<34 || y>85) {
        const i=(y*128+x)*4;
        changing[i] += 4*shift;
        changing[i+2] = shift < 0 ? 170 : 255;
      }
    }
    changingExpressions.push({
      anchored:fitFacePose(changing,128,[0.2,0.2,0.8,0.8],anchors),
      legacy:fitFacePose(changing,128,[0.2,0.2,0.8,0.8]),
    });
  }
  const dubbed = (() => {
    const target = document.createElement("canvas"), c = new Compositor(target, new URLSearchParams(), true);
    c.load({photo:flat(W,H,"#333"),rig:{crop:[128,48,384,304],face:[180,100,332,250],detail:null},
      ref:live(0),detail:null,bias:flat(1,1,"rgb(128,128,128)")});
    c.setFrame("idle", live(16), guide(16));
    c.setPicture("photo");
    const sample = mix => {
      c.draw(mix,1);
      reader.drawImage(target,0,0);
      return [175,500].map(y => Array.from(reader.getImageData(256,y,1,1).data).slice(0,3));
    };
    const result = {initial:sample(0)};
    c.loadClips({mode:"avatar",box:[0,0,W,H],follow:[100,250],pivot:[256,175],
      facePose:[1,0,0,0,1,0],faceMode:"full"});
    c.setClipFrame(video);
    c.setClip({mix:1});
    c.setClipTone([0,1,0,0,0,1,1,0,0]);
    c.setPicture("clips");
    c.setFrame("live",paint(W,H,g => {
      g.fillStyle="rgb(224,128,64)";g.fillRect(0,0,W,H/2);
      g.fillStyle="rgb(64,160,224)";g.fillRect(0,H/2,W,H/2);
    }),null);
    result.idle=sample(0);
    result.speaking=sample(1);
    result.fade=sample(0.5);
    result.poseWarning=Boolean(c.poseWarning);
    c.setPicture("photo");
    result.photo=sample(1);
    c.gl.getExtension("WEBGL_lose_context").loseContext();
    return result;
  })();
  return {idle:render(0,16,null),talk:render(1,16,null),
    carried:render(1,16,{scale:1,turn:0,x:20,y:0}),cleared:render(0,16,null,true),
    registered:render(1,16,{scale:1,turn:0,x:20,y:0},false,true),
    anchored:render(1,16,{scale:1,turn:0,x:20,y:0},false,true,false,true),
    mouthOnly:render(1,16,null,false,false,true),
    mouthIdle:render(0,16,null,false,false,true),
    wholeIdle:render(0,16,null,false,false,false,false,true),
    wholeTalk:render(1,16,null,false,false,false,false,true),
    wholeCarried:render(1,16,{scale:1,turn:0,x:20,y:0},false,false,false,false,true),
    changingExpressions,dubbed,
    singularAnchors:fitFacePose(data,128,[0.2,0.2,0.8,0.8],Array(13).fill([0.5,0.5])),
    fitted:fitFacePose(data,128,[0.2,0.2,0.8,0.8]),expected};
}"""


def main():
    try:
        from playwright.sync_api import Error, sync_playwright
    except ImportError:
        return SKIP_STATUS
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(args=[
                "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        except Error:
            return SKIP_STATUS
        page = browser.new_page()
        page.set_content("<!doctype html><title>compositor</title>")
        page.add_script_tag(path=str(COMPOSITOR))
        result = page.evaluate(SCRIPT)
        result["clips"] = page.evaluate(CLIP_SCRIPT)
        result["join"] = page.evaluate(JOIN_SCRIPT)
        result["avatar"] = page.evaluate(AVATAR_SCRIPT)
        print(json.dumps(result))
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
