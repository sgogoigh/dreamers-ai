// Single source of truth for the brand mark. The SAME markup is written to
// app/icon.svg (the browser-tab favicon) and rendered into the page heading,
// so the tab icon and the heading icon are guaranteed identical.
export const ICON_SVG = `<svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <defs><linearGradient id="dg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#a78bfa"/><stop offset=".5" stop-color="#22d3ee"/><stop offset="1" stop-color="#f472b6"/>
  </linearGradient></defs>
  <rect x="7" y="25" width="50" height="31" rx="8" fill="url(#dg)"/>
  <rect x="7" y="14" width="50" height="13" rx="4" fill="url(#dg)"/>
  <g fill="#05060d" opacity=".8">
    <path d="M13 14h7l-7 13h-6z"/><path d="M26 14h7l-7 13h-7z"/><path d="M39 14h7l-7 13h-7z"/>
  </g>
  <path d="M28 35l13 6.5-13 6.5z" fill="#05060d" opacity=".85"/>
  <path d="M50 5l1.8 4.7L56.5 11l-4.7 1.8L50 17.5l-1.8-4.7L43.5 11l4.7-1.3z" fill="#fff"/>
</svg>`;
