import { defineConfig } from 'astro/config';
import icon from 'astro-icon';
import tailwind from '@astrojs/tailwind';

export default defineConfig({
  site: 'https://jmutale5.github.io',
  base: '/data-science-portfolio', // This is the key for formatting!
  integrations: [
    icon(), 
    tailwind({
      applyBaseStyles: false,
    })
  ],
});
