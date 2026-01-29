import { defineConfig } from 'astro/config';
import icon from 'astro-icon';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  // Replace with your actual GitHub URL later
  site: 'https://joanmutale.github.io',
  base: 'ds-portfolio',
  integrations: [icon()],
  vite: {
    plugins: [tailwindcss()],
  },
});