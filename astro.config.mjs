import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';
import icon from 'astro-icon';

export default defineConfig({
  site: 'https://jmutale5.github.io',
  base: '/data-science-portfolio',
  integrations: [
    tailwind(), 
    icon()
  ],
});
