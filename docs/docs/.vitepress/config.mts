import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Torch Lens Maker",
  description: "Differentiable geometric optics in PyTorch",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Documentation', link: '/' },
      { text: 'About', link: '/about' }
    ],

    logo: '/logos/tlmlogo_black150.png',

    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Welcome', link: '/'},
          { text: 'Design Overview', link: '/design-overview' },
          { text: 'Installation', link: '/installation' },
        ]
      },
      {
        text: 'Examples',
        items: [
          { text: 'Pink Floyd', link: '/examples/pink_floyd'},
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/victorpoughon/torchlensmaker' }
    ],

    search: {
      provider: 'local'
    }
  },

  cleanUrls: true,

  markdown: {
    math: true,
    defaultHighlightLang: "python",
    theme : {
      light : 'github-light',
      dark: 'github-dark',
  }}
})
