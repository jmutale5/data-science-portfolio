import { defineConfig } from 'astro/config';
import icon from 'astro-icon';
import tailwind from '@astrojs/tailwind';

export default defineConfig({
  site: 'https://jmutale5.github.io',
  base: '/data-science-portfolio',
  integrations: [
    icon(), 
    tailwind({
      applyBaseStyles: false,
    })
  ],
});
