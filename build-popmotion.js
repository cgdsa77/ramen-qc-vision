// 使用本地 popmotion-master 打包为单文件，输出到 web/js/popmotion.bundle.js。
// 首次使用前请先在 popmotion-master 根目录执行：npm install（或 yarn） 与 npm run build（或 yarn build），
// 以生成 packages/popmotion/dist。然后在本项目根目录执行：npm install && npm run build:popmotion。
const esbuild = require('esbuild');
const path = require('path');
const fs = require('fs');

const outDir = path.join(__dirname, 'web', 'js');
const outFile = path.join(outDir, 'popmotion.bundle.js');

if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

esbuild.build({
  entryPoints: [path.join(__dirname, 'web', 'popmotion-entry.js')],
  bundle: true,
  format: 'iife',
  outfile: outFile,
  minify: true,
}).then(() => {
  console.log('Popmotion 已打包到:', outFile);
}).catch((e) => {
  console.error('打包失败:', e);
  process.exit(1);
});
