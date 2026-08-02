import { defineConfig } from 'vite'

export default defineConfig({
  // Asset URLs are baked in at build time, so this has to match the serving path.
  // GitHub project pages serve from /<repo>/; BASE_PATH covers hosting it anywhere else.
  base: process.env.BASE_PATH ?? (process.env.GITHUB_ACTIONS ? '/webgpu-fluid-sim/' : '/'),
})
