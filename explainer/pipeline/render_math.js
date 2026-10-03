// Replace $$...$$ (display) and \(...\) (inline) TeX in the HTML before <script> with MathJax SVG.
// The SVG glyphs use currentColor, so equations follow the page theme. Usage: node render_math.js in.html out.html
const fs = require('fs');
const { mathjax } = require('mathjax-full/js/mathjax.js');
const { TeX } = require('mathjax-full/js/input/tex.js');
const { SVG } = require('mathjax-full/js/output/svg.js');
const { liteAdaptor } = require('mathjax-full/js/adaptors/liteAdaptor.js');
const { RegisterHTMLHandler } = require('mathjax-full/js/handlers/html.js');
const { AllPackages } = require('mathjax-full/js/input/tex/AllPackages.js');
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const doc = mathjax.document('', { InputJax: new TeX({ packages: AllPackages }), OutputJax: new SVG({ fontCache: 'local' }) });
const esc = s => s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
function tex(src, display) {
  const node = doc.convert(src, { display });
  const svg = adaptor.innerHTML(node);          // the <svg> inside <mjx-container>
  if (/merror|data-mjx-error/.test(svg)) throw new Error('TeX error in: ' + src);
  return `<span class="math" role="img" aria-label="${esc(src)}">${svg.replace('<svg ', '<svg aria-hidden="true" focusable="false" ')}</span>`;
}
const [inp, out] = process.argv.slice(2);   // template in, page skeleton with math out
const html = fs.readFileSync(inp, 'utf8');
const cut = html.indexOf('<script>');
let head = html.slice(0, cut);
let n = 0;
head = head.replace(/\$\$([\s\S]+?)\$\$/g, (m, s) => (n++, `<div class="math-d">${tex(s.trim(), true)}</div>`));
head = head.replace(/\\\(([\s\S]+?)\\\)/g, (m, s) => (n++, tex(s.trim(), false)));
fs.writeFileSync(out, head + html.slice(cut));
console.log('rendered', n, 'equations');
