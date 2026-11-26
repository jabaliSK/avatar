// Line icons for the interface, drawn on a 24 x 24 grid and coloured by the text colour around them.
const SVG_NS = "http://www.w3.org/2000/svg";

const PATHS = {
  plus: "M12 5v14M5 12h14",
  search: "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM20 20l-4.2-4.2",
  open: "M8 5.5v13l11-6.5z",
  link: "M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1",
  sliders: "M4 7h8M16 7h4M4 17h4M12 17h8M12 7a2 2 0 1 0 4 0 2 2 0 1 0-4 0M8 17a2 2 0 1 0 4 0 2 2 0 1 0-4 0",
  trash: "M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 11v5M14 11v5",
  close: "M6 6l12 12M18 6L6 18",
  mic: "M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3zM6 11a6 6 0 0 0 12 0M12 17v4M9 21h6",
  micoff: "M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3zM6 11a6 6 0 0 0 12 0M12 17v4M9 21h6M4 4l16 16",
  stop: "M7 7h10v10H7z",
  pause: "M9 5v14M15 5v14",
  send: "M21 3L3 10.5l7 3 3 7zM10 13.5L21 3",
  upload: "M12 16V4M7 9l5-5 5 5M5 20h14",
  bolt: "M13 3L5 14h6l-1 7 8-11h-6z",
  layers: "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5",
  check: "M5 12.5l4.5 4.5L19 7.5",
  alert: "M12 4L2.5 20h19zM12 10v4.5M12 17.5v.01",
  info: "M12 4a8 8 0 1 0 0 16 8 8 0 0 0 0-16zM12 11v5M12 8v.01",
  refresh: "M20 5v5h-5M4 19v-5h5M18.4 9A7 7 0 0 0 6.2 6.8M5.6 15A7 7 0 0 0 17.8 17.2",
  back: "M19 12H5M11 6l-6 6 6 6",
  cpu: "M7 7h10v10H7zM10 10h4v4h-4zM9 3v4M15 3v4M9 17v4M15 17v4M3 9h4M3 15h4M17 9h4M17 15h4",
  disk: "M4 14h16M4 14l2.5-8h11L20 14v4a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1zM8 16.5h.01M12 16.5h.01",
  spark: "M12 3l2 6 6 2-6 2-2 6-2-6-6-2 6-2z",
  clock: "M12 4a8 8 0 1 0 0 16 8 8 0 0 0 0-16zM12 8v4.5l3 1.5",
  film: "M4 4h16v16H4zM8 4v16M16 4v16M4 9h4M16 9h4M4 15h4M16 15h4",
  image: "M4 5h16v14H4zM4 17l5-5 4 4 3-3 4 4M9 9.5a1 1 0 1 0 .01 0",
  person: "M12 4a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM4.5 20c.5-4 3.5-6 7.5-6s7 2 7.5 6",
  copy: "M9 9h11v11H9zM5 15V4h11",
  lock: "M7 11h10v9H7zM9 11V8a3 3 0 0 1 6 0v3",
  wave: "M4 12h2M8 8v8M12 5v14M16 8v8M20 12h-2",
  text: "M4 6h16M4 12h10M4 18h13",
  hex: "M12 3l8 4.5v9L12 21l-8-4.5v-9z",
  upgrade: "M12 20V8M6 13l6-6 6 6M5 4h14",
};

export function icon(name, size = 18) {
  const svg = document.createElementNS(SVG_NS, "svg");
  for (const [key, value] of Object.entries({
    viewBox: "0 0 24 24", width: size, height: size, fill: "none", stroke: "currentColor", "stroke-width": 1.8,
    "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true", focusable: "false", class: "icon",
  })) {
    svg.setAttribute(key, String(value));
  }
  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute("d", PATHS[name] || PATHS.hex);
  svg.append(path);
  return svg;
}
