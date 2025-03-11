import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Torch Lens Maker",
  description: "Differentiable geometric optics in PyTorch",
  head: [['link', { rel: 'icon', href: '/logos/tlmlogo_black130_margin.png' }]],
  themeConfig: {
    logo: '/logos/tlmlogo_black150.png',

    nav: [
      { text: 'Documentation', link: '/' },
      { text: 'About', link: '/about' }
    ],

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
