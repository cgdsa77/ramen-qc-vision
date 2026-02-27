// 供 build-popmotion.js 打包用，将 popmotion 挂到 window.popmotion
import { animate, easeInOut, spring } from 'popmotion';
window.popmotion = { animate, easeInOut, spring };
