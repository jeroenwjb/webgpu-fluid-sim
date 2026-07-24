import './style.css'

const canvas = document.querySelector<HTMLCanvasElement>('#fluid-canvas')!
const fallback = document.querySelector<HTMLDivElement>('#webgpu-fallback')!

if (!navigator.gpu) {
  canvas.hidden = true
  fallback.hidden = false
} else {
  console.log('WebGPU is available. Device init comes in the next milestone.')
}
