import { defineConfig } from 'vite'

export default defineConfig({
  // Project pages serve from /<repo>/, not the domain root.
  base: process.env.GITHUB_ACTIONS ? '/webgpu-fluid-sim/' : '/',
})
